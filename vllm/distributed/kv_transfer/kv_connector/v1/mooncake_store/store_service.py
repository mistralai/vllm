# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from collections.abc import Callable
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import ReqMeta


class MooncakeStoreService:
    """Own cache I/O queues; transfer threads retain CUDA buffer ownership."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.save_queue: queue.Queue[ReqMeta | None] = queue.Queue()
        self.load_queue: queue.Queue[ReqMeta] = queue.Queue()

    def check_save_exists(self, keys: list[str]) -> list[int]:
        return self.store.batch_is_exist(keys)

    def put(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
        replicate_config: Any,
    ) -> list[int]:
        return self.store.batch_put_from_multi_buffers(
            keys, addrs, sizes, replicate_config
        )

    def get(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
    ) -> list[int]:
        return self.store.batch_get_into_multi_buffers(keys, addrs, sizes)

    def lookup_exists(
        self,
        keys: list[str],
        exists_fn: Callable[[list[str]], list[int]],
    ) -> list[int]:
        return exists_fn(keys)
