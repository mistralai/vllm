# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific (READ) worker-side logic for the NIXL connector."""

import time
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Slack (seconds) subtracted from D's exported block-expiry deadline on the turn-2
# readback, absorbing clock-offset error and read latency.
_KV_BLOCKS_EXPIRY_SAFETY_MARGIN = 5.0


class NixlPullConnectorWorker(NixlBaseConnectorWorker):
    """Pull-specific (READ) worker logic."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids, self._physical_blocks_per_logical_kv_block
            )
            assert meta.remote is not None
            # Remote block IDs are kept logical here; expanded in
            # _read_blocks_for_req using the remote engine's phys ratio.
            remote_engine_id = meta.remote.engine_id
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            # always store metadata for failure recovery
            self._recving_metadata[req_id] = meta
            if remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_nixl_handshake(req_id, remote_engine_id, meta)
                        continue

            # Handshake already completed, start async read xfer.
            self._read_blocks_for_req(req_id, meta)

        # Start transfers for requests whose handshakes have now finished.
        while not self._ready_requests.empty():
            self._read_blocks_for_req(*self._ready_requests.get_nowait())

        if self.pcp_rank > 0:
            return

        # Keep around the requests that have been part of a batch. This is
        # needed because async scheduling pushes the misalignment between the
        # moment in which requests expiration is set (P side) and the moment in
        # which blocks are read from D. As P can now more easily lag behind D
        # while processing the next batch, we make sure to only set an
        # expiration for requests that have not been read from D yet.
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)

        # Remove all requests that are not to be processed (eg aborted).
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            # We should never get an abort after setting an expiry timer
            assert req_id not in self._reqs_to_send

        # Add to requests that are waiting to be read and track expiration.
        # Deadlines are stamped with the scheduler process's perf_counter,
        # which is not comparable to ours when the worker runs in another
        # process on another node (perf_counter epochs differ by boot time).
        # Rebase the remaining TTL onto our clock; broadcast latency only
        # lengthens the lease, which is the safe direction. A cross-node
        # epoch gap larger than the TTL otherwise expires the lease on
        # arrival and the blocks are freed before D reads them.
        now_local = time.perf_counter()
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                if metadata.scheduler_clock:
                    expiration_time = now_local + (
                        expiration_time - metadata.scheduler_clock
                    )
                self._reqs_to_send[req_id] = expiration_time

        # Send heartbeats to P-side engines to keep KV blocks alive while
        # requests sit in the D scheduler WAITING queue.
        self._send_heartbeats(metadata)

    def _is_turn2_read_expired(self, meta: ReqMeta) -> bool:
        """Whether D's cached blocks for this turn-2 readback have (nearly) expired."""
        assert meta.remote is not None
        blocks_expiry_time = meta.remote.blocks_expiry_time
        # Deadline may be absent (router may not forward it) -> read as usual.
        if blocks_expiry_time is None or not meta.local_physical_block_ids:
            return False
        clock_offset = self._engine_clock_offset[meta.remote.engine_id]
        deadline = blocks_expiry_time - clock_offset
        return time.perf_counter() + _KV_BLOCKS_EXPIRY_SAFETY_MARGIN >= deadline

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        # Update last activity from this remote. Mind that cleanup is done on main
        # thread (this one), so we don't race on this structure.
        self._engine_last_active[engine_id] = time.perf_counter()

        if self._bidirectional_kv_xfer_enabled and self._is_turn2_read_expired(meta):
            logger.warning(
                "Declining expired remote read for %s from engine %s.",
                req_id,
                engine_id,
            )
            self.xfer_stats.record_kv_expired_req()
            self._handle_failed_transfer(req_id, None)
            return

        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        local_block_ids = meta.local_physical_block_ids
        remote_region_groups = self.dst_region_group_ids[engine_id]
        local_region_groups = self.region_group_ids or remote_region_groups
        groups_differ = local_region_groups != remote_region_groups
        if groups_differ:
            if not self.use_mla or self._has_mamba:
                raise NotImplementedError(
                    "Different NIXL cache-group layouts are only supported for "
                    "pure MLA models"
                )
            assert len(plan.all_source_ranks) == 1
            # Remote ids arrive at the producer's logical granularity; expand
            # them to its kernel pages (the group-matched branch does the
            # same) so per-region counts pair with our kernel-granularity ids.
            meta.remote.block_ids = self._logical_to_kernel_block_ids(
                meta.remote.block_ids,
                remote_info.remote_physical_blocks_per_logical,
            )
            remote_by_region = self._block_ids_by_region(
                meta.remote.block_ids, remote_region_groups
            )
            read_specs = [
                ReadSpec(
                    remote_rank=plan.all_source_ranks[0],
                    local_block_ids=self._block_ids_by_region(
                        local_block_ids, local_region_groups
                    ),
                    remote_block_ids=remote_by_region,
                    block_ids_by_region=True,
                )
            ]
        else:
            meta.remote.block_ids = self._logical_to_kernel_block_ids(
                meta.remote.block_ids,
                remote_info.remote_physical_blocks_per_logical,
            )
            remote_block_ids = meta.remote.block_ids
            num_groups = len(local_block_ids)
            read_specs = [
                ReadSpec(
                    remote_rank=rank,
                    local_block_ids=[
                        list(local_block_ids[g])
                        if rank in plan.source_ranks_per_group[g]
                        else []
                        for g in range(num_groups)
                    ],
                    remote_block_ids=[
                        list(remote_block_ids[g])
                        if rank in plan.source_ranks_per_group[g]
                        else []
                        for g in range(num_groups)
                    ],
                )
                for rank in plan.all_source_ranks
            ]

        # D may have to perform multiple reads from different remote ranks.
        # Pure MLA reads once because its cache is replicated. Hybrid
        # MLA+SSM still needs one read per SSM source rank.
        if self.use_mla and tp_ratio < 0 and not self._has_mamba:
            assert len(read_specs) == 1

        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _read_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            # Get side handles.
            if tp_ratio < 0 and (not self.use_mla or len(read_specs) > 1):
                # Remote tp_size > local tp_size: we must perform multiple
                # reads. Get the memory chunk onto which we will write to.
                split_key = (tp_ratio, remote_block_size)
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[split_key][i]
            else:
                # Single read from remote, we write to the whole memory region.
                # Also handle remote block size different from local block size.
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            # Destination handle: remote_engine_id -> remote_rank -> handle.
            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            # Once a read routes the request to failure reporting, the
            # scheduler may free and reuse its blocks, so no sibling READs
            # may be posted (and P must not be notified).
            if not self._read_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
            ):
                return

        if self.use_mla and tp_ratio < 0 and len(read_specs) == 1:
            # ..but we still need to notify the other remote ranks that we
            # have the blocks we need so they can update the request state.
            notif_id = f"{meta.remote.request_id}:{self.world_size}".encode()
            remote_agents = self._remote_agents[meta.remote.engine_id]
            for rank_to_notify, agent in remote_agents.items():
                if rank_to_notify != (0, read_specs[0].remote_rank):
                    self.nixl_wrapper.send_notif(agent, notif_msg=notif_id)

    def _read_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ) -> bool:
        """
        Post a READ point-to-point xfer request from a single local worker to
        a single remote worker.

        Returns True when the read was posted (or was unnecessary), False
        when the request was routed to failure reporting — the caller must
        not post further transfers for it.
        """
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            if read_spec.block_ids_by_region:
                raise NotImplementedError(
                    "Region-mapped NIXL transfers require matching physical block sizes"
                )
            local_block_ids, remote_block_ids = (
                self._map_block_ids_for_block_size_ratio(
                    local_block_ids, remote_block_ids, block_size_ratio
                )
            )
        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes.

        # Number of D TP workers that will read from dst P. Propagate info
        # on notification so that dst worker can wait before freeing blocks.
        notif_id = f"{remote_request_id}:{self.world_size}".encode()

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        if len(local_block_ids) == 0:
            # A full prefix cache hit is indicated with an empty list.
            agent_name = self._remote_agents[dst_engine_id][(0, remote_rank)]
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    msg="P worker blocks will be freed after timeout. "
                    "This may indicate network issues.",
                    req_id=request_id,
                    error=e,
                    dst_engine_id=dst_engine_id,
                    remote_rank=remote_rank,
                    remote_agent_name=agent_name,
                )
                self.xfer_stats.record_failed_notification()
            return True

        if read_spec.block_ids_by_region:
            local_block_ids, remote_block_ids = self._apply_prefix_caching_by_region(
                decode_block_ids=local_block_ids,
                prefill_block_ids=remote_block_ids,
            )
        else:
            assert (
                len(remote_block_ids)
                == len(local_block_ids)
                == len(self.kv_cache_config.transfer_groups)
            )
            local_block_ids, remote_block_ids = self._apply_prefix_caching(
                decode_block_ids=local_block_ids,
                prefill_block_ids=remote_block_ids,
                decode_physical_per_logical=(
                    self._physical_blocks_per_logical_kv_block
                ),
                prefill_physical_per_logical=(
                    remote_info.remote_physical_blocks_per_logical
                ),
            )

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.

        # Region-group layout of the local side; entries of the block-id
        # lists are regions for region-mapped specs, cache groups otherwise.
        local_region_group_ids = (
            list(range(self.num_regions))
            if read_spec.block_ids_by_region
            else (self.region_group_ids or None)
        )

        # Get descs ids.
        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
            region_num_blocks=(self.dst_region_num_blocks.get(dst_engine_id) or None),
            region_group_ids=(
                list(range(self.num_regions))
                if read_spec.block_ids_by_region
                else (self.dst_region_group_ids.get(dst_engine_id) or None)
            ),
        )

        if self._mixed_mem_types:
            # Mixed-memory registrations span several descriptor lists, so
            # one READ is posted per memory type; the flat local descriptor
            # ids below are never used.
            handle = None
            try:
                self._read_blocks_by_mem_type(
                    read_spec=read_spec,
                    request_id=request_id,
                    dst_engine_id=dst_engine_id,
                    remote_block_descs_ids=remote_block_descs_ids,
                    local_region_group_ids=local_region_group_ids,
                    local_vram_handle=local_xfer_side_handle,
                    remote_xfer_side_handle=remote_xfer_side_handle,
                    block_size_ratio=block_size_ratio,
                    local_block_size_key=remote_info.remote_block_size,
                    notif_id=notif_id,
                )
                return True
            except Exception as e:
                # mark all (logical) blocks for this request as invalid
                self._log_failure(
                    failure_type="transfer_setup_failed",
                    req_id=request_id,
                    msg="Marking blocks as invalid",
                    error=e,
                    dst_engine_id=dst_engine_id,
                    remote_rank=remote_rank,
                )
                self._handle_failed_transfer(request_id, handle)
                return False

        if self.dst_xfer_side_handles_by_mem_type.get(dst_engine_id, {}).get(
            remote_rank
        ):
            # The remote spans memory types while this side does not: its
            # descriptor lists are split by type and cannot be addressed with
            # one flat dlist.
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Marking blocks as invalid",
                error=NotImplementedError(
                    "Reading a mixed-memory remote from a single-memory-type "
                    "local registration is not supported."
                ),
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            self._handle_failed_transfer(request_id, None)
            return False

        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
            region_num_blocks=(self.dst_region_num_blocks.get(self.engine_id) or None),
            region_group_ids=local_region_group_ids,
        )

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        # Prepare transfer with Nixl.
        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )

            # Begin async xfer.
            self.nixl_wrapper.transfer(handle)

            # Use handle to check completion in future step().
            self._recving_transfers[request_id].append(handle)
            return True
        except Exception as e:
            # mark all (logical) blocks for this request as invalid
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Marking blocks as invalid",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            self._handle_failed_transfer(request_id, handle)
            return False

    def _read_blocks_by_mem_type(
        self,
        read_spec: ReadSpec,
        request_id: str,
        dst_engine_id: str,
        remote_block_descs_ids: np.ndarray,
        local_region_group_ids: list[int] | None,
        local_vram_handle: int,
        remote_xfer_side_handle: int,
        block_size_ratio: int | None,
        local_block_size_key: int,
        notif_id: bytes,
    ) -> None:
        """Post one READ per memory type over a mixed-memory registration.

        Local descriptor lists are split by memory type at registration, so
        descriptor ids are computed per partition while the remote side keeps
        its flat ids, sliced at the partitions' entry segments. A remote that
        also spans memory types (e.g. a HiSparse DRAM host pool) splits its
        dlists by type too, so each local partition is further paired by the
        remote regions' type. P is freed only once every partition's READ
        completes, hence the deferred notification.
        """
        local_block_ids = read_spec.local_block_ids
        region_group_ids = local_region_group_ids
        if region_group_ids is None:
            region_group_ids = self.region_group_ids
        if not region_group_ids and self.num_regions == 1:
            region_group_ids = [0]
        group_arr = np.asarray(region_group_ids)
        partitions = self._mem_type_partitions(region_group_ids, len(local_block_ids))

        # Flat local and remote descriptor ids pair positionally, so each
        # entry owns a contiguous segment of both arrays.
        segments: list[slice] = []
        start = 0
        for entry, blocks in enumerate(local_block_ids):
            length = len(blocks) * int((group_arr == entry).sum())
            segments.append(slice(start, start + length))
            start += length
        assert start == len(remote_block_descs_ids)

        remote_handles_by_mem_type = self.dst_xfer_side_handles_by_mem_type.get(
            dst_engine_id, {}
        ).get(read_spec.remote_rank)
        region_mapped = bool(read_spec.block_ids_by_region)

        reads: list[tuple[int, np.ndarray, int, np.ndarray]] = []
        for mem_type, entries in partitions.items():
            # Entries of one local memory type may pair with remote regions of
            # different types; each (local, remote) type pair reads against its
            # own pair of dlists.
            by_remote_type: dict[str | None, list[int]] = {}
            for entry in entries:
                remote_mem_type = self._remote_entry_mem_type(
                    dst_engine_id, entry, region_mapped
                )
                if (
                    remote_handles_by_mem_type is not None
                    and remote_mem_type not in remote_handles_by_mem_type
                ):
                    raise RuntimeError(
                        f"No remote dlist for entry {entry} memory type "
                        f"{remote_mem_type}; available: "
                        f"{sorted(remote_handles_by_mem_type)}."
                    )
                by_remote_type.setdefault(remote_mem_type, []).append(entry)

            for remote_mem_type, typed_entries in by_remote_type.items():
                remote_handle = (
                    remote_handles_by_mem_type[remote_mem_type]
                    if remote_handles_by_mem_type is not None
                    else remote_xfer_side_handle
                )
                # The local dlist for mem_type holds every local region of
                # that type in region order, so ids are computed over the
                # full set; regions of entries outside typed_entries occupy
                # dlist slots but contribute no ids (sentinel group id).
                type_regions = [
                    r
                    for r in range(len(self.region_mem_types))
                    if self.region_mem_types[r] == mem_type
                ]
                entry_pos = {entry: pos for pos, entry in enumerate(typed_entries)}
                local_ids = self._compute_desc_ids(
                    block_ids=[local_block_ids[i] for i in typed_entries],
                    dst_num_blocks=self.dst_num_blocks[self.engine_id],
                    block_size_ratio=block_size_ratio,
                    physical_blocks_per_logical=(
                        self._physical_blocks_per_logical_kv_block
                    ),
                    region_num_blocks=[self.region_num_blocks[r] for r in type_regions],
                    region_group_ids=[
                        entry_pos.get(region_group_ids[r], len(typed_entries))
                        for r in type_regions
                    ],
                )
                if len(local_ids) == 0:
                    continue
                if remote_handles_by_mem_type is not None:
                    # The remote dlist for this type holds that type's regions
                    # in region order, so ids are computed against it rather
                    # than sliced from the flat region-major sequence.
                    remote_ids = self._remote_desc_ids_for_type(
                        dst_engine_id=dst_engine_id,
                        mem_type=remote_mem_type,
                        entries=typed_entries,
                        region_mapped=region_mapped,
                        remote_block_ids=read_spec.remote_block_ids,
                    )
                else:
                    remote_ids = np.concatenate(
                        [remote_block_descs_ids[segments[i]] for i in typed_entries]
                    )
                assert len(local_ids) == len(remote_ids)
                if mem_type == "DRAM":
                    local_handle = self._dram_src_handles_by_block_size[
                        local_block_size_key
                    ]
                else:
                    local_handle = local_vram_handle
                reads.append((local_handle, local_ids, remote_handle, remote_ids))

        prepped: list[int] = []
        try:
            for local_handle, local_ids, remote_handle, remote_ids in reads:
                prepped.append(
                    self.nixl_wrapper.make_prepped_xfer(
                        "READ",
                        local_handle,
                        local_ids,
                        remote_handle,
                        remote_ids,
                    )
                )
        except Exception:
            for handle in prepped:
                self.nixl_wrapper.release_xfer_handle(handle)
            raise

        notif_agent = self._remote_agents[dst_engine_id][(0, read_spec.remote_rank)]
        self._pending_recv_notifs.setdefault(request_id, []).append(
            (notif_agent, notif_id)
        )
        for i, handle in enumerate(prepped):
            try:
                self.nixl_wrapper.transfer(handle)
            except Exception:
                for unstarted in prepped[i:]:
                    self.nixl_wrapper.release_xfer_handle(unstarted)
                raise
            self._recving_transfers[request_id].append(handle)

    def _remote_entry_mem_type(
        self, dst_engine_id: str, entry: int, region_mapped: bool
    ) -> str | None:
        """Memory type of the remote regions paired with a block-id entry.

        Entries are regions for region-mapped specs and cache groups
        otherwise; the remote side uses the same layout. Returns None when
        the remote did not report per-region memory types or the entry spans
        several.
        """
        dst_mem_types = self.dst_region_mem_types.get(dst_engine_id)
        if not dst_mem_types:
            return None
        if region_mapped:
            return dst_mem_types[entry] if entry < len(dst_mem_types) else None
        dst_group_ids = self.dst_region_group_ids.get(dst_engine_id)
        if not dst_group_ids or len(dst_group_ids) != len(dst_mem_types):
            return None
        types = {
            dst_mem_types[r]
            for r in range(len(dst_mem_types))
            if dst_group_ids[r] == entry
        }
        return types.pop() if len(types) == 1 else None

    def _remote_desc_ids_for_type(
        self,
        dst_engine_id: str,
        mem_type: str,
        entries: list[int],
        region_mapped: bool,
        remote_block_ids: BlockIds,
    ) -> np.ndarray:
        """Descriptor ids into a remote dlist holding one memory type.

        The dlist concatenates the remote regions of ``mem_type`` in region
        order, so ids are computed over that region subset. Regions of
        entries outside ``entries`` still occupy dlist slots but contribute
        no ids; a sentinel group id keeps them out of the block-id lookup.
        """
        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        dst_mem_types = self.dst_region_mem_types.get(dst_engine_id, [])
        dst_region_num_blocks = self.dst_region_num_blocks.get(dst_engine_id, [])
        dst_group_ids = self.dst_region_group_ids.get(dst_engine_id, [])
        entry_index = {entry: pos for pos, entry in enumerate(entries)}
        region_num_blocks: list[int] = []
        region_group_ids: list[int] = []
        for r in range(len(dst_mem_types)):
            if dst_mem_types[r] != mem_type:
                continue
            region_num_blocks.append(dst_region_num_blocks[r])
            group = r if region_mapped else dst_group_ids[r]
            region_group_ids.append(entry_index.get(group, len(entries)))
        return self._compute_desc_ids(
            block_ids=[remote_block_ids[i] for i in entries],
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
            region_num_blocks=region_num_blocks,
            region_group_ids=region_group_ids,
        )

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP scenario), wait
        for all consumers to be done pulling.

        Also handles heartbeat notifications ("HB:req1,req2,...") by
        extending the lease on the referenced requests.
        """
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                msg = notif.decode("utf-8")

                # Handle heartbeat messages from D-side.
                if msg.startswith("HB:"):
                    self._handle_heartbeat(msg[3:])
                    continue

                req_id, tp_size = msg.rsplit(":", 1)
                if (
                    req_id not in self._reqs_to_send
                    and req_id not in self._reqs_to_process
                ):
                    logger.error(
                        "Potentially invalid KV blocks for "
                        "unrecognized request %s were retrieved by "
                        "a decode worker. They may have expired.",
                        req_id,
                    )
                    continue

                # NOTE: `tp_ratio` is the opposite when swapping local<>remote
                n_consumers = int(tp_size)
                tp_ratio = self.transfer_topo.tp_ratio(n_consumers)

                # Number of reads *per producer* to wait for.
                # When remote D TP > local P TP we expect `tp_ratio` reads.
                consumers_per_producer = (
                    -tp_ratio if n_consumers > self.world_size else 1
                )

                self.consumer_notification_counts_by_req[req_id] += 1
                # Wait all consumers (D) to be done reading before freeing.
                if (
                    self.consumer_notification_counts_by_req[req_id]
                    == consumers_per_producer
                ):
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
        return notified_req_ids
