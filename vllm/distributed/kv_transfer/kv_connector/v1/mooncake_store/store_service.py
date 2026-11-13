# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from collections.abc import Callable
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import ReqMeta


class CacheUnavailable(RuntimeError):
    pass


class MooncakeStoreService:
    """Own cache I/O queues; transfer threads retain CUDA buffer ownership."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.save_queue: queue.Queue[ReqMeta | None] = queue.Queue()
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
