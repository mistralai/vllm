# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from queue import Queue
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (
    ReqMeta,
)


@dataclass(frozen=True)
class SaveConfig:
    max_pending_blocks_ratio: float = 0.1
    save_start_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        ratio = self.max_pending_blocks_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or not 0 <= ratio <= 1
        ):
            raise ValueError("max_pending_blocks_ratio must be between 0 and 1")
        timeout = self.save_start_timeout_s
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("save_start_timeout_s must be finite and positive")

    @classmethod
    def from_extra_config(cls, extra: Mapping[str, Any]) -> "SaveConfig":
        return cls(**{f.name: extra[f.name] for f in fields(cls) if f.name in extra})


class SaveBlockBudget:
    """Charge shared blocks until the last job using them completes."""

    def __init__(self, max_blocks: int) -> None:
        self.max_blocks = max_blocks
        self._blocks: Counter[int] = Counter()

    def acquire(self, blocks: Sequence[int]) -> bool:
        unique = set(blocks)
        new_blocks = sum(block not in self._blocks for block in unique)
        if not unique or len(self._blocks) + new_blocks > self.max_blocks:
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
    def __init__(self, config: SaveConfig) -> None:
        super().__init__()
        self.start_timeout_s = config.save_start_timeout_s

    def put(
        self, item: ReqMeta | None, block: bool = True, timeout: float | None = None
    ) -> None:
        if item is not None:
            # Assigned locally: monotonic clocks on different hosts cannot
            # be compared through scheduler metadata.
            item.save_start_deadline = time.monotonic() + self.start_timeout_s
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
