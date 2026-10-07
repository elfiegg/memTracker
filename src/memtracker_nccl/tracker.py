"""Add modeled external allocations to PyTorch's tensor lifetime accounting."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch.distributed._tools.mem_tracker import MemTracker


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

    def __init__(self, *, record_tensor_events: bool = False) -> None:
        self._external: dict[str, dict[str, Any]] = {}
        self._combined_peaks: dict[str, dict[str, Any]] = {}
        self._tensor_peaks: dict[str, int] = {}
        self._external_peaks: dict[str, int] = {}
        self._events: list[dict[str, Any]] = []
        self._sequence = 0
        self.record_tensor_events = record_tensor_events
        super().__init__()

    def _update_snap(self, *args: Any, **kwargs: Any) -> None:
        super()._update_snap(*args, **kwargs)
        self._sample("tensor_update", record=self.record_tensor_events)

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
        if record:
            self._events.append({"sequence": self._sequence, "event": kind,
                                 **details, "devices": deepcopy(snapshot)})

    def report(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot, captured independently of future frees."""
        return deepcopy({
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
