# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-node NVLink fabric detection via ``nvidia-smi``.

The Python NVML fabric API can return an all-zero structure on GB300 systems,
while ``nvidia-smi -q`` reports the fabric state and identifiers correctly.
"""

import os
import subprocess
from typing import TYPE_CHECKING

import torch.distributed as dist

from vllm.logger import init_logger

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = init_logger(__name__)

_NULL_CLUSTER_UUID_HEX = "0" * 32
_HEX_CHARS = frozenset("0123456789abcdefABCDEF")
_NvlinkFabricKey = tuple[str, int]


def _local_gpu_id_for_nvidia_smi() -> str:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [device.strip() for device in visible.split(",") if device.strip()]
        if local_rank < len(devices):
            return devices[local_rank]
    return str(local_rank)


def _parse_nvidia_smi_fabric_key(output: str) -> _NvlinkFabricKey | None:
    """Parse a healthy ``nvidia-smi -q`` Fabric block."""

    def get_value(names: set[str]) -> str | None:
        for line in output.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = " ".join(key.split()).lower()
            if key in names:
                value = value.strip()
                return value if value and value != "N/A" else None
        return None

    state = get_value({"state", "fabric state"})
    status = get_value({"status", "fabric status"})
    cluster_uuid = get_value({"clusteruuid", "cluster uuid", "fabric cluster uuid"})
    clique_id = get_value({"cliqueid", "clique id", "fabric clique id"})

    if state is None or state.lower() != "completed":
        return None
    if status is not None and status.lower() != "success":
        return None
    if cluster_uuid is None or clique_id is None:
        return None

    uuid_hex = "".join(char for char in cluster_uuid if char in _HEX_CHARS).lower()
    if len(uuid_hex) < 32:
        return None
    uuid_hex = uuid_hex[:32]
    if uuid_hex == _NULL_CLUSTER_UUID_HEX:
        return None

    try:
        clique = int(clique_id, 0)
    except ValueError:
        return None

    return (uuid_hex, clique)


def _get_local_nvlink_fabric_key() -> _NvlinkFabricKey | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-i", _local_gpu_id_for_nvidia_smi()],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as error:
        logger.debug("nvidia-smi fabric detection failed: %s", error)
        return None

    if result.returncode != 0:
        logger.debug("nvidia-smi failed: %s", result.stderr.strip())
        return None

    return _parse_nvidia_smi_fabric_key(result.stdout)


def has_cross_node_nvlink(group: "ProcessGroup") -> bool:
    """Return whether all ranks report the same valid NVLink fabric.

    Detection fails closed when fabric information is unavailable or incomplete.
    The caller is responsible for establishing that the group crosses nodes.
    """
    if not dist.is_available() or not dist.is_initialized():
        return False

    world_size = dist.get_world_size(group=group)
    if world_size <= 1:
        return False

    fabric_keys: list[_NvlinkFabricKey | None] = [None] * world_size
    dist.all_gather_object(
        fabric_keys,
        _get_local_nvlink_fabric_key(),
        group=group,
    )
    valid_keys = [key for key in fabric_keys if key is not None]
    if len(valid_keys) != world_size:
        return False

    has_shared_fabric = len(set(valid_keys)) == 1
    if has_shared_fabric:
        logger.info_once(
            "Detected multi-node NVLink fabric: %s",
            valid_keys[0],
            scope="local",
        )
    return has_shared_fabric
