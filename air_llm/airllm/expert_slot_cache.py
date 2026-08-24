"""Fixed-slot expert runtime for memory-constrained MoE inference.

The compatibility expert cache keeps weights attached to individual ``nn.Module`` objects.  That
is portable, but a miss allocates new accelerator tensors and eviction replaces them with ``meta``
parameters.  This module provides the lower-overhead alternative used by the explicit PhiMoE slot
backend: one normalized CPU bank, fixed accelerator storage, and a two-lane pinned transfer pipe.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import math
import time
from typing import Callable, Iterable

import torch
from safetensors import safe_open


ExpertKey = tuple[int, int]

_ALLOWED_SOURCE_DTYPES = {"F16", "BF16"}
_PROJECTIONS = ("w1.weight", "w2.weight", "w3.weight")


def _numel(shape: Iterable[int]) -> int:
    return math.prod(int(dim) for dim in shape)


def _available_host_memory_bytes() -> int | None:
    """Return Linux MemAvailable when present, otherwise leave allocation to PyTorch."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


@dataclass(frozen=True)
class SlotReservation:
    slot: int
    hit: bool
    evicted_key: ExpertKey | None = None


class FixedSlotLRU:
    """Pure bookkeeping for a fixed number of equal-size expert slots."""

    def __init__(self, slot_count: int):
        if slot_count < 1:
            raise ValueError("fixed expert cache requires at least one slot")
        self.slot_count = int(slot_count)
        self._entries: OrderedDict[ExpertKey, int] = OrderedDict()
        self._free_slots = list(range(self.slot_count - 1, -1, -1))

    def lookup(self, key: ExpertKey, *, touch: bool = True) -> int | None:
        slot = self._entries.get(key)
        if slot is not None and touch:
            self._entries.move_to_end(key)
        return slot

    def reserve(
        self,
        key: ExpertKey,
        *,
        protected: Iterable[ExpertKey] = (),
    ) -> SlotReservation | None:
        existing = self.lookup(key)
        if existing is not None:
            return SlotReservation(existing, True)

        if self._free_slots:
            slot = self._free_slots.pop()
            self._entries[key] = slot
            return SlotReservation(slot, False)

        protected_set = set(protected)
        victim = next(
            (candidate for candidate in self._entries if candidate not in protected_set),
            None,
        )
        if victim is None:
            return None
        slot = self._entries.pop(victim)
        self._entries[key] = slot
        return SlotReservation(slot, False, victim)

    def clear(self) -> None:
        self._entries.clear()
        self._free_slots = list(range(self.slot_count - 1, -1, -1))

    @property
    def resident_count(self) -> int:
        return len(self._entries)

    def keys(self) -> tuple[ExpertKey, ...]:
        return tuple(self._entries)


class ExpertHostBank:
    """Three contiguous pageable CPU banks indexed by ``(layer, expert)``."""

    host_reserve_bytes = 2 * 1024 ** 3
    host_preflight_min_bank_bytes = 512 * 1024 ** 2

    def __init__(
        self,
        checkpoint_path,
        layer_names,
        expert_keys,
        expert_param_prefixes,
        dtype: torch.dtype,
    ):
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("slot expert bank supports only float16 or bfloat16 runtime dtype")

        self.dtype = dtype
        self.projection_names = _PROJECTIONS
        self._flat_index: dict[ExpertKey, int] = {}
        self._entries = [
            (layer_idx, expert_idx)
            for layer_idx in sorted(expert_keys)
            for expert_idx in sorted(expert_keys[layer_idx])
        ]
        if not self._entries:
            raise ValueError("slot expert bank found no checkpoint-indexed experts")
        self._flat_index = {key: flat for flat, key in enumerate(self._entries)}

        checkpoint_path = Path(checkpoint_path)
        metadata: dict[str, tuple[int, ...]] = {}
        source_dtypes: set[str] = set()
        layer_records: dict[int, list[tuple[int, dict[str, str]]]] = {}

        for layer_idx, expert_idx in self._entries:
            prefix = expert_param_prefixes[(layer_idx, expert_idx)]
            by_projection = {}
            for tensor_name in expert_keys[layer_idx][expert_idx]:
                expected_prefix = prefix + "."
                if not tensor_name.startswith(expected_prefix):
                    raise ValueError(f"expert tensor {tensor_name!r} is outside {prefix!r}")
                by_projection[tensor_name[len(expected_prefix):]] = tensor_name
            if set(by_projection) != set(_PROJECTIONS):
                raise ValueError(
                    f"PhiMoE expert {layer_idx, expert_idx} must contain exactly "
                    f"{', '.join(_PROJECTIONS)}; found {sorted(by_projection)}"
                )
            layer_records.setdefault(layer_idx, []).append((expert_idx, by_projection))

        # Validate every tensor before committing several GiB of host memory.
        for layer_idx, records in layer_records.items():
            shard = checkpoint_path / f"{layer_names[layer_idx]}.safetensors"
            with safe_open(str(shard), framework="pt") as handle:
                available = set(handle.keys())
                for expert_idx, projections in records:
                    for projection, tensor_name in projections.items():
                        if tensor_name not in available:
                            raise ValueError(
                                f"missing expert tensor {tensor_name!r} in {shard.name}"
                            )
                        tensor_slice = handle.get_slice(tensor_name)
                        shape = tuple(tensor_slice.get_shape())
                        source_dtype = tensor_slice.get_dtype()
                        if source_dtype not in _ALLOWED_SOURCE_DTYPES:
                            raise ValueError(
                                "slot backend requires FP16/BF16 experts; "
                                f"{tensor_name} is {source_dtype}"
                            )
                        source_dtypes.add(source_dtype)
                        expected = metadata.setdefault(projection, shape)
                        if shape != expected:
                            raise ValueError(
                                f"non-uniform {projection} shape for expert "
                                f"{layer_idx, expert_idx}: "
                                f"expected {expected}, found {shape}"
                            )

        dtype_bytes = torch.empty((), dtype=dtype).element_size()
        self.expert_bytes = sum(_numel(metadata[name]) * dtype_bytes for name in _PROJECTIONS)
        self.nbytes = self.expert_bytes * len(self._entries)
        available_host = _available_host_memory_bytes()
        if (
            self.nbytes >= self.host_preflight_min_bank_bytes
            and available_host is not None
            and available_host < self.nbytes + self.host_reserve_bytes
        ):
            raise MemoryError(
                "insufficient host memory for slot expert bank: "
                f"need {self.nbytes / 1024 ** 3:.2f} GiB plus 2 GiB reserve, "
                f"have {available_host / 1024 ** 3:.2f} GiB available"
            )

        started = time.perf_counter()
        try:
            self.banks = {
                name: torch.empty((len(self._entries), *metadata[name]), dtype=dtype)
                for name in _PROJECTIONS
            }
            source_tensor_reads = 0
            with torch.no_grad():
                for layer_idx, records in layer_records.items():
                    shard = checkpoint_path / f"{layer_names[layer_idx]}.safetensors"
                    with safe_open(str(shard), framework="pt") as handle:
                        for expert_idx, projections in records:
                            flat = self._flat_index[(layer_idx, expert_idx)]
                            for projection in _PROJECTIONS:
                                self.banks[projection][flat].copy_(
                                    handle.get_tensor(projections[projection])
                                )
                                source_tensor_reads += 1
        except Exception as exc:
            self.banks = {}
            raise RuntimeError("failed to build the pageable PhiMoE expert bank") from exc

        self.source_dtypes = tuple(sorted(source_dtypes))
        self.source_tensor_reads = source_tensor_reads
        self.load_seconds = time.perf_counter() - started

    def tensors(self, key: ExpertKey) -> tuple[torch.Tensor, ...]:
        try:
            flat = self._flat_index[key]
        except KeyError as exc:
            raise KeyError(f"unknown expert {key}") from exc
        return tuple(self.banks[name][flat] for name in _PROJECTIONS)

    @property
    def expert_count(self) -> int:
        return len(self._entries)


@dataclass
class _ExpertRequest:
    key: ExpertKey
    expert_idx: int
    kind: str
    index: int
    hit: bool
    lane: int | None = None
    ready_event: object | None = None


class ExpertSlotRuntime:
    """Fixed GPU expert slots with a pageable-bank -> pinned-stage -> GPU pipeline."""

    scratch_slots = 2

    def __init__(
        self,
        host_bank: ExpertHostBank,
        capacity_bytes: int,
        device: torch.device,
    ):
        self.host_bank = host_bank
        self.capacity_bytes = int(capacity_bytes)
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("slot expert runtime requires a CUDA-compatible device (CUDA or ROCm)")

        self.expert_bytes = host_bank.expert_bytes
        self.slot_count = self.capacity_bytes // self.expert_bytes
        if self.slot_count < 1:
            expert_mib = self.expert_bytes / 1024 ** 2
            raise ValueError(f"expert_cache_gb is smaller than one {expert_mib:.2f} MiB expert")

        self.lru = FixedSlotLRU(self.slot_count)
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="airllm-expert-stage")
        self._in_execute = False
        self.dense_resident_bytes = 0

        try:
            self.cache_banks = {
                name: torch.empty((self.slot_count, *bank.shape[1:]), dtype=bank.dtype,
                                  device=self.device)
                for name, bank in host_bank.banks.items()
            }
            self.scratch_banks = {
                name: torch.empty((self.scratch_slots, *bank.shape[1:]), dtype=bank.dtype,
                                  device=self.device)
                for name, bank in host_bank.banks.items()
            }
            self.staging_banks = {
                name: torch.empty((self.scratch_slots, *bank.shape[1:]), dtype=bank.dtype,
                                  device="cpu", pin_memory=True)
                for name, bank in host_bank.banks.items()
            }
        except Exception:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self.cache_banks = {}
            self.scratch_banks = {}
            self.staging_banks = {}
            raise

        self._cache_ready = [None] * self.slot_count
        self._cache_last_use = [None] * self.slot_count
        self._scratch_ready = [None] * self.scratch_slots
        self._scratch_last_use = [None] * self.scratch_slots
        self._lane_transfer_done = [None] * self.scratch_slots
        self.reset_stats()

    @property
    def allocated_cache_bytes(self) -> int:
        return self.slot_count * self.expert_bytes

    @property
    def scratch_bytes(self) -> int:
        return self.scratch_slots * self.expert_bytes

    @property
    def pinned_staging_bytes(self) -> int:
        return self.scratch_slots * self.expert_bytes

    @property
    def allocated_device_bytes(self) -> int:
        return self.allocated_cache_bytes + self.scratch_bytes + self.dense_resident_bytes

    def _destination_banks(self, request: _ExpertRequest) -> tuple[torch.Tensor, ...]:
        banks = self.cache_banks if request.kind == "cache" else self.scratch_banks
        return tuple(banks[name][request.index] for name in self.host_bank.projection_names)

    def _destination_last_use(self, request: _ExpertRequest):
        events = self._cache_last_use if request.kind == "cache" else self._scratch_last_use
        return events[request.index]

    def _set_destination_ready(self, request: _ExpertRequest, event) -> None:
        events = self._cache_ready if request.kind == "cache" else self._scratch_ready
        events[request.index] = event

    def _destination_ready(self, request: _ExpertRequest):
        events = self._cache_ready if request.kind == "cache" else self._scratch_ready
        return events[request.index]

    def _record_destination_use(self, request: _ExpertRequest, stream) -> None:
        event = torch.cuda.Event(blocking=False)
        event.record(stream)
        events = self._cache_last_use if request.kind == "cache" else self._scratch_last_use
        events[request.index] = event

    def _fill_staging_lane(self, lane: int, key: ExpertKey, previous_transfer) -> None:
        if previous_transfer is not None:
            previous_transfer.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            for projection, source in zip(self.host_bank.projection_names,
                                          self.host_bank.tensors(key)):
                self.staging_banks[projection][lane].copy_(source)
        self.host_stage_seconds += time.perf_counter() - started
        self.host_stage_copies += 1

    def _submit_stage(self, lane: int, request: _ExpertRequest) -> Future:
        return self._executor.submit(
            self._fill_staging_lane,
            lane,
            request.key,
            self._lane_transfer_done[lane],
        )

    def _enqueue_transfer(self, lane: int, request: _ExpertRequest) -> None:
        with torch.cuda.stream(self.transfer_stream):
            last_use = self._destination_last_use(request)
            if last_use is not None:
                self.transfer_stream.wait_event(last_use)
            for projection, destination in zip(
                self.host_bank.projection_names,
                self._destination_banks(request),
            ):
                destination.copy_(self.staging_banks[projection][lane], non_blocking=True)
            ready = torch.cuda.Event(blocking=False)
            ready.record(self.transfer_stream)
        request.ready_event = ready
        self._set_destination_ready(request, ready)
        self._lane_transfer_done[lane] = ready
        self.h2d_copies += 1
        self.h2d_bytes += self.expert_bytes

    def _plan(self, layer_idx: int, expert_indices: Iterable[int], admit: bool):
        expert_indices = [int(index) for index in expert_indices]
        protected = {(layer_idx, index) for index in expert_indices}
        requests = []
        miss_count = 0
        for expert_idx in expert_indices:
            key = (layer_idx, expert_idx)
            existing = self.lru.lookup(key)
            if existing is not None:
                self.hits += 1
                requests.append(_ExpertRequest(key, expert_idx, "cache", existing, True))
                continue

            self.misses += 1
            reservation = self.lru.reserve(key, protected=protected) if admit else None
            if reservation is not None:
                if reservation.evicted_key is not None:
                    self.evictions += 1
                request = _ExpertRequest(key, expert_idx, "cache", reservation.slot, False)
            else:
                lane = miss_count % self.scratch_slots
                request = _ExpertRequest(key, expert_idx, "scratch", lane, False)
                if admit:
                    self.bypasses += 1
                else:
                    self.prefill_skips += 1
            request.lane = miss_count % self.scratch_slots
            miss_count += 1
            requests.append(request)
        return requests

    def execute(
        self,
        layer_idx: int,
        expert_indices: Iterable[int],
        *,
        admit: bool,
        callback: Callable[[int, tuple[torch.Tensor, ...]], None],
    ) -> None:
        """Execute callbacks in routing order while staging misses two experts ahead."""
        if self._in_execute:
            raise RuntimeError("slot expert runtime is not reentrant")
        self._in_execute = True
        pending: dict[int, Future | None] = {lane: None for lane in range(self.scratch_slots)}
        lane_queues: dict[int, list[_ExpertRequest]] = {
            lane: [] for lane in range(self.scratch_slots)
        }
        lane_positions = {lane: 0 for lane in range(self.scratch_slots)}

        try:
            requests = self._plan(layer_idx, expert_indices, admit)
            for request in requests:
                if not request.hit:
                    lane_queues[request.lane].append(request)

            # Stage and enqueue the first miss assigned to each lane.
            for lane, queue in lane_queues.items():
                if queue:
                    pending[lane] = self._submit_stage(lane, queue[0])
            for lane, queue in lane_queues.items():
                if not queue:
                    continue
                pending[lane].result()
                self._enqueue_transfer(lane, queue[0])
                lane_positions[lane] = 1
                pending[lane] = (
                    self._submit_stage(lane, queue[1]) if len(queue) > 1 else None
                )

            compute_stream = torch.cuda.current_stream(self.device)
            for request in requests:
                ready = request.ready_event if not request.hit else self._destination_ready(request)
                if ready is not None:
                    compute_stream.wait_event(ready)
                callback(request.expert_idx, self._destination_banks(request))
                self._record_destination_use(request, compute_stream)

                if request.hit:
                    continue
                lane = request.lane
                queue = lane_queues[lane]
                position = lane_positions[lane]
                if position >= len(queue):
                    pending[lane] = None
                    continue

                next_request = queue[position]
                pending[lane].result()
                self._enqueue_transfer(lane, next_request)
                lane_positions[lane] = position + 1
                following = position + 1
                pending[lane] = (
                    self._submit_stage(lane, queue[following])
                    if following < len(queue)
                    else None
                )
        except Exception:
            # A caller may catch a failed generation and issue another request. Never leave a
            # logical hit pointing at a partially copied slot in that case.
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass
            self._invalidate()
            raise
        finally:
            for future in pending.values():
                if future is not None:
                    # Any staging failure encountered on the normal path was already raised above.
                    # If the callback failed first, do not replace that exception while draining
                    # background work during cleanup.
                    try:
                        future.result()
                    except Exception:
                        pass
            self._in_execute = False

    def _invalidate(self) -> None:
        self.lru.clear()
        self._cache_ready = [None] * self.slot_count
        self._cache_last_use = [None] * self.slot_count
        self._scratch_ready = [None] * self.scratch_slots
        self._scratch_last_use = [None] * self.scratch_slots
        self._lane_transfer_done = [None] * self.scratch_slots

    def clear(self) -> None:
        torch.cuda.synchronize(self.device)
        self._invalidate()

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.bypasses = 0
        self.prefill_skips = 0
        self.h2d_copies = 0
        self.h2d_bytes = 0
        self.host_stage_copies = 0
        self.host_stage_seconds = 0.0

    def stats(self) -> dict[str, int | float | str | bool]:
        return {
            "enabled": True,
            "backend": "slot",
            "capacity_bytes": self.capacity_bytes,
            "allocated_cache_bytes": self.allocated_cache_bytes,
            "resident_bytes": self.lru.resident_count * self.expert_bytes,
            "resident_experts": self.lru.resident_count,
            "slot_count": self.slot_count,
            "scratch_slots": self.scratch_slots,
            "scratch_bytes": self.scratch_bytes,
            "allocated_device_bytes": self.allocated_device_bytes,
            "dense_resident_bytes": self.dense_resident_bytes,
            "host_bank_bytes": self.host_bank.nbytes,
            "host_bank_experts": self.host_bank.expert_count,
            "host_bank_load_seconds": self.host_bank.load_seconds,
            "host_bank_tensor_reads": self.host_bank.source_tensor_reads,
            "pinned_staging_bytes": self.pinned_staging_bytes,
            "generation_disk_expert_reads": 0,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "bypasses": self.bypasses,
            "prefill_skips": self.prefill_skips,
            "oom_retries": 0,
            "h2d_copies": self.h2d_copies,
            "h2d_bytes": self.h2d_bytes,
            "host_stage_copies": self.host_stage_copies,
            "host_stage_seconds": self.host_stage_seconds,
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
