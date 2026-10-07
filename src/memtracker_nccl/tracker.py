"""Add modeled external allocations to PyTorch's tensor lifetime accounting."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from typing import Any

import torch
from torch.distributed._tools.mem_tracker import MemTracker, _UpdateType

from .allocator_model import AllocatorConfig, CachingAllocatorModel


def _device_key(device: str | torch.device) -> str:
    parsed = torch.device(device)
    # CPU indices do not identify separate physical memory pools. FakeTensor
    # can preserve these indices, although eager CPU tensors canonicalize them.
    return "cpu" if parsed.type == "cpu" else str(parsed)


class ExtendedMemTracker(MemTracker):
    """MemTracker with an external allocation ledger and simultaneous peaks.

    PyTorch's inherited snapshots remain tensor-only. ``report`` combines tensor
    and external bytes at each event, instead of adding independent maxima.
    This uses the private ``_update_snap`` hook; tested with PyTorch 2.14.1.
    External bytes are estimates supplied by the caller, not measured memory.
    """

    def __init__(self, *, record_tensor_events: bool = False,
                 allocator_config: AllocatorConfig | None = None) -> None:
        self._allocator = CachingAllocatorModel(allocator_config) if allocator_config is not None else None
        self._storage_allocations: dict[Any, tuple[str, str, str, str]] = {}
        self._allocation_counter = 0
        self._allocation_domain = ("default", "default")
        self._resident_peaks: dict[str, dict[str, Any]] = {}
        self._external: dict[str, dict[str, Any]] = {}
        self._combined_peaks: dict[str, dict[str, Any]] = {}
        self._tensor_peaks: dict[str, int] = {}
        self._external_peaks: dict[str, int] = {}
        self._events: list[dict[str, Any]] = []
        self._sequence = 0
        self.record_tensor_events = record_tensor_events
        super().__init__()

    def _update_snap(self, u_type: Any, winfo: Any, *args: Any, **kwargs: Any) -> None:
        if self._allocator is not None:
            if u_type in (_UpdateType.ADD, _UpdateType.SIZE):
                device = _device_key(winfo.device)
                pool, stream = self._allocation_domain
                previous = self._storage_allocations.get(winfo)
                if previous is not None:
                    _, device, pool, stream = previous
                self._allocation_counter += 1
                handle = f"storage-{self._allocation_counter}"
                self._allocator.allocate(handle, winfo.size * winfo.element_size,
                                         device=device, pool=pool, stream=stream)
                if previous is not None:
                    # Conservative resize approximation: new backing before old
                    # release. Sample that overlap, which storage totals omit.
                    self._sample("allocator_resize_overlap", record=self.record_tensor_events)
                    self._allocator.free(previous[0])
                self._storage_allocations[winfo] = (handle, device, pool, stream)
            elif u_type == _UpdateType.DEL:
                self._allocator.free(self._storage_allocations.pop(winfo)[0])
        super()._update_snap(u_type, winfo, *args, **kwargs)
        self._sample("tensor_update", record=self.record_tensor_events)

    def _require_allocator(self) -> CachingAllocatorModel:
        if self._allocator is None:
            raise RuntimeError("Pass allocator_config to enable allocator modeling")
        return self._allocator

    @contextmanager
    def allocation_scope(self, *, pool: str = "default", stream: str = "default"):
        """Label newly observed storage; this does not create a real CUDA MemPool."""
        self._require_allocator()
        if not isinstance(pool, str) or not pool or not isinstance(stream, str) or not stream:
            raise ValueError("pool and stream must be nonempty strings")
        previous = self._allocation_domain
        self._allocation_domain = (pool, stream)
        try:
            yield
        finally:
            self._allocation_domain = previous

    def record_tensor_stream(self, tensor: torch.Tensor, completion_token: str) -> None:
        """Require an explicit completion event before this storage can be reused."""
        allocator = self._require_allocator()
        winfo, _ = self._WINFO.get(tensor.untyped_storage(), (None, None))
        if winfo not in self._storage_allocations:
            raise ValueError("Tensor storage is not tracked; call track_external first")
        allocator.record_stream(self._storage_allocations[winfo][0], completion_token)
        self._sample("allocator_record_stream", token=completion_token)

    def complete_stream(self, completion_token: str) -> None:
        self._require_allocator().complete(completion_token)
        self._sample("allocator_complete_stream", token=completion_token)

    def empty_allocator_cache(self, device: str | torch.device | None = None) -> None:
        """Release eligible modeled backing, without synchronizing pending work."""
        self._require_allocator().empty_cache(None if device is None else _device_key(device))
        self._sample("allocator_empty_cache")

    def _resident_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        backing = self._require_allocator().snapshot()
        return {
            device: {
                "reserved_bytes": backing.get(device, {}).get("reserved_bytes", 0),
                "external_bytes": snapshot.get(device, {}).get("external_bytes", 0),
                "external_by_category": snapshot.get(device, {}).get("external_by_category", {}),
                "combined_bytes": backing.get(device, {}).get("reserved_bytes", 0)
                                  + snapshot.get(device, {}).get("external_bytes", 0),
            }
            for device in sorted(set(backing) | set(snapshot))
        }

    def external_alloc(
        self,
        allocation_id: str,
        size_bytes: int,
        device: str | torch.device = "cpu",
        category: str = "NCCL",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not allocation_id or allocation_id in self._external:
            raise ValueError(f"Allocation id must be nonempty and unique: {allocation_id!r}")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError("size_bytes must be a nonnegative integer")
        dev = _device_key(device)
        self._external[allocation_id] = {
            "size_bytes": size_bytes, "device": dev, "category": category,
            "metadata": deepcopy(metadata or {}),
        }
        self._sample("external_alloc", allocation_id=allocation_id,
                     allocation=deepcopy(self._external[allocation_id]))

    def external_free(self, allocation_id: str) -> None:
        if allocation_id not in self._external:
            raise KeyError(f"Unknown or already freed allocation: {allocation_id!r}")
        allocation = self._external.pop(allocation_id)
        self._sample("external_free", allocation_id=allocation_id, allocation=allocation)

    def _snapshot(self) -> dict[str, dict[str, Any]]:
        devices: dict[str, dict[str, Any]] = {}
        for device, stats in self._curr_mem_snap.items():
            device_stats = devices.setdefault(_device_key(device), {
                "tensor_bytes": 0, "external_bytes": 0,
                "external_by_category": {},
            })
            device_stats["tensor_bytes"] += int(stats["Total"])
        for allocation in self._external.values():
            stats = devices.setdefault(allocation["device"], {
                "tensor_bytes": 0, "external_bytes": 0, "external_by_category": {},
            })
            size, category = allocation["size_bytes"], allocation["category"]
            stats["external_bytes"] += size
            stats["external_by_category"][category] = stats["external_by_category"].get(category, 0) + size
        for stats in devices.values():
            stats["combined_bytes"] = stats["tensor_bytes"] + stats["external_bytes"]
        return devices

    def _sample(self, kind: str, *, record: bool = True, **details: Any) -> None:
        self._sequence += 1
        snapshot = self._snapshot()
        for device, stats in snapshot.items():
            self._tensor_peaks[device] = max(self._tensor_peaks.get(device, 0), stats["tensor_bytes"])
            self._external_peaks[device] = max(self._external_peaks.get(device, 0), stats["external_bytes"])
            if device not in self._combined_peaks or stats["combined_bytes"] > self._combined_peaks[device]["combined_bytes"]:
                self._combined_peaks[device] = {**deepcopy(stats), "sequence": self._sequence, "event": kind}
        resident = None
        if self._allocator is not None:
            resident = self._resident_snapshot(snapshot)
            for device, stats in resident.items():
                if device not in self._resident_peaks or stats["combined_bytes"] > self._resident_peaks[device]["combined_bytes"]:
                    self._resident_peaks[device] = {**deepcopy(stats), "sequence": self._sequence, "event": kind}
        if record:
            self._events.append({"sequence": self._sequence, "event": kind,
                                 **details, "devices": deepcopy(snapshot),
                                 **({"resident": deepcopy(resident)} if resident is not None else {})})

    def report(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot, captured independently of future frees."""
        optional = {}
        if self._allocator is not None:
            optional = {
                "allocator": {"model": self._allocator.describe(), "current": self._allocator.snapshot()},
                "resident": {
                    "accounting": "modeled allocator reserved backing plus modeled external allocations; not a measurement",
                    "current": self._resident_snapshot(self._snapshot()), "peak": self._resident_peaks,
                    "assumptions": ["CPU FakeTensor device labels represent hypothetical CUDA allocations.",
                                    "Storage resize modeled as allocate-before-free; streams/pools/completion must be supplied.",
                                    "External entries must be disjoint from modeled allocator backing."],
                },
            }
        return deepcopy({
            **optional,
            "schema_version": 1,
            "torch_version": str(torch.__version__),
            "accounting": "live tensor storage plus modeled external allocations; not reserved GPU memory",
            "current": self._snapshot(),
            "peak": self._combined_peaks,
            "tensor_peak_bytes": self._tensor_peaks,
            "external_peak_bytes": self._external_peaks,
            "live_external_allocations": self._external,
            "events": self._events,
        })
