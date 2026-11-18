# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import get_max_offload_tokens


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (None, None),
        ({}, None),
        ({"max_offload_tokens": None}, None),
        ({"max_offload_tokens": 0}, 0),
        ({"max_offload_tokens": 16}, 16),
        ({"max_offload_tokens": False}, None),
        ({"max_offload_tokens": True}, None),
        ({"max_offload_tokens": -1}, None),
        ({"max_offload_tokens": "0"}, None),
        ({"max_offload_tokens": 0.0}, None),
    ],
)
def test_request_offload_limit_preserves_zero_without_coercing_invalid_values(
    params: dict[str, Any] | None, expected: int | None
) -> None:
    assert get_max_offload_tokens(params) == expected
