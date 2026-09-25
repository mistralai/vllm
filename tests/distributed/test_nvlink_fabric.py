# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

import vllm.v1.attention.ops.cp_common as cp_common
from vllm.distributed.device_communicators import nvlink_fabric


def _fabric_block(
    state: str = "Completed",
    status: str = "Success",
    cluster_uuid: str = "bd77041e-6cd1-4b2a-81ed-f13c81d31e11",
) -> str:
    return "\n".join(
        [
            "Fabric",
            f"    State       : {state}",
            f"    Status      : {status}",
            "    CliqueId    : 32766",
            f"    ClusterUUID : {cluster_uuid}",
        ]
    )


def test_parse_completed_fabric_key() -> None:
    assert nvlink_fabric._parse_nvidia_smi_fabric_key(_fabric_block()) == (
        "bd77041e6cd14b2a81edf13c81d31e11",
        32766,
    )


@pytest.mark.parametrize(
    "output",
    [
        _fabric_block(state="Not Started"),
        _fabric_block(status="Failure"),
        _fabric_block(cluster_uuid="00000000-0000-0000-0000-000000000000"),
        "GPU 00000000:06:00.0",
    ],
)
def test_parse_rejects_unusable_fabric(output: str) -> None:
    assert nvlink_fabric._parse_nvidia_smi_fabric_key(output) is None


@pytest.mark.parametrize(
    ("gathered_keys", "expected"),
    [
        ([("fabric-a", 32766), ("fabric-a", 32766)], True),
        ([("fabric-a", 32766), ("fabric-b", 32766)], False),
        ([("fabric-a", 32766), None], False),
    ],
)
def test_cross_node_nvlink_requires_one_shared_fabric(
    monkeypatch,
    gathered_keys: list[tuple[str, int] | None],
    expected: bool,
) -> None:
    process_group = object()
    monkeypatch.setattr(nvlink_fabric.dist, "is_available", lambda: True)
    monkeypatch.setattr(nvlink_fabric.dist, "is_initialized", lambda: True)

    def get_world_size(*, group):
        assert group is process_group
        return len(gathered_keys)

    monkeypatch.setattr(
        nvlink_fabric.dist,
        "get_world_size",
        get_world_size,
    )

    def all_gather_object(output, _local_key, *, group):
        assert group is process_group
        output[:] = gathered_keys

    monkeypatch.setattr(nvlink_fabric.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        nvlink_fabric,
        "_get_local_nvlink_fabric_key",
        lambda: gathered_keys[0],
    )

    assert nvlink_fabric.has_cross_node_nvlink(process_group) is expected


def test_multinode_without_nvlink_skips_symmetric_memory_probe(monkeypatch) -> None:
    group = MagicMock(cpu_group=object())
    cross_node_nvlink = MagicMock(return_value=False)
    symmetric_memory = MagicMock()

    cp_common._symm_mem_spans_group.cache_clear()
    monkeypatch.setattr(cp_common, "symm_mem_available", True)
    monkeypatch.setattr(cp_common, "_group_is_intra_node", lambda _group: False)
    monkeypatch.setattr(cp_common, "has_cross_node_nvlink", cross_node_nvlink)
    monkeypatch.setattr(cp_common, "symm_mem", symmetric_memory)

    assert not cp_common._symm_mem_spans_group(group)
    cross_node_nvlink.assert_called_once_with(group.cpu_group)
    symmetric_memory.empty.assert_not_called()
