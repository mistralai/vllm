# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from queue import Queue
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import ReqMeta


class CacheUnavailable(RuntimeError):
    pass


class MooncakeStoreService:
    """Own cache I/O queues; transfer threads retain CUDA buffer ownership."""

    SAVE_START_TIMEOUT_S = 5.0
    MAX_SAVE_QUEUE_SIZE = 256

    @dataclass(frozen=True)
    class SaveConfig:
        max_pending_blocks_ratio: float = 0.1

        def __post_init__(self) -> None:
            ratio = self.max_pending_blocks_ratio
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not 0 <= ratio <= 1
            ):
                raise ValueError("max_pending_blocks_ratio must be between 0 and 1")

        @classmethod
        def from_extra_config(
            cls, extra: Mapping[str, Any]
        ) -> "MooncakeStoreService.SaveConfig":
            return cls(
                extra.get("max_pending_blocks_ratio", cls.max_pending_blocks_ratio)
            )

    class SaveBlockBudget:
        """Charge shared blocks until the last job using them completes."""

        def __init__(self, max_blocks: int) -> None:
            self.max_blocks = max_blocks
            self._blocks: Counter[int] = Counter()

        @property
        def num_blocks(self) -> int:
            return len(self._blocks)

        def acquire(self, blocks: Sequence[int], reserved_blocks: int = 0) -> bool:
            unique = set(blocks)
            new_blocks = sum(block not in self._blocks for block in unique)
            if (
                not unique
                or len(self._blocks) + new_blocks + reserved_blocks > self.max_blocks
            ):
                return False
            self._blocks.update(unique)
            return True

        def release(self, blocks: Sequence[int]) -> None:
            for block in set(blocks):
                assert self._blocks[block] > 0, f"Save block {block} is not held"
                self._blocks[block] -= 1
                if self._blocks[block] == 0:
                    del self._blocks[block]

    class SaveQueue(Queue[ReqMeta | None]):
        def __init__(self) -> None:
            super().__init__(maxsize=MooncakeStoreService.MAX_SAVE_QUEUE_SIZE)

        def put(
            self, item: ReqMeta | None, block: bool = True, timeout: float | None = None
        ) -> None:
            if item is not None:
                # Assigned locally: monotonic clocks on different hosts cannot
                # be compared through scheduler metadata.
                item.save_start_deadline = (
                    time.monotonic() + MooncakeStoreService.SAVE_START_TIMEOUT_S
                )
            super().put(item, block, timeout)

        def expire(self) -> list[ReqMeta]:
            expired: list[ReqMeta] = []
            with self.mutex:
                now = time.monotonic()
                while self.queue:
                    item = self.queue[0]
                    if (
                        item is None
                        or item.save_start_deadline is None
                        or item.save_start_deadline > now
                    ):
                        break
                    self._get()
                    expired.append(item)
                    assert self.unfinished_tasks > 0, "Queued save has no task charge"
                    self.unfinished_tasks -= 1
                if expired:
                    self.not_full.notify_all()
                    if self.unfinished_tasks == 0:
                        self.all_tasks_done.notify_all()
            return expired

    def __init__(self, store: Any, config: SaveConfig) -> None:
        self.store = store
        self.save_queue = self.SaveQueue()
        self.load_queue: queue.Queue[ReqMeta] = queue.Queue()

    @staticmethod
    def _validate_results(results: list[int], count: int) -> list[int]:
        if len(results) != count:
            raise CacheUnavailable(
                f"Mooncake returned {len(results)} results for {count} keys"
            )
        return results

    def check_save_exists(self, keys: list[str]) -> list[int]:
        return self._validate_results(self.store.batch_is_exist(keys), len(keys))

    def put(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
        replicate_config: Any,
    ) -> list[int]:
        return self._validate_results(
            self.store.batch_put_from_multi_buffers(
                keys, addrs, sizes, replicate_config
            ),
            len(keys),
        )

    def get(
        self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]
    ) -> list[int]:
        return self._validate_results(
            self.store.batch_get_into_multi_buffers(keys, addrs, sizes), len(keys)
        )

    def lookup_exists(
        self, keys: list[str], exists_fn: Callable[[list[str]], list[int]]
    ) -> list[int]:
        results = self._validate_results(exists_fn(keys), len(keys))
        if any(result < 0 for result in results):
            raise CacheUnavailable("Mooncake lookup returned SDK errors")
        return results

    def expire_saves(self, finish: Callable[[ReqMeta, str], None]) -> None:
        for request in self.save_queue.expire():
            finish(request, "start_timeout")

    def submit_save(
        self, request: ReqMeta, finish: Callable[[ReqMeta, str], None]
    ) -> None:
        try:
            self.save_queue.put_nowait(request)
        except queue.Full:
            finish(request, "queue_overflow")
