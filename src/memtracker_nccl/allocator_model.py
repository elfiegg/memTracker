"""A deterministic CPU-only approximation of native caching-allocator backing.

Inputs describe storage lifetimes and explicitly named stream-completion tokens.
This is not a CUDA allocator, timing simulator, capacity check, or OOM predictor.
Offsets are stable synthetic addresses within a segment, not device pointers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

MiB = 2**20
SOURCE = "https://github.com/pytorch/pytorch/blob/v2.8.0/c10/cuda/CUDACachingAllocator.cpp"


def _label(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be nonempty")


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class AllocatorConfig:
    """Source-derived default native rules; nondefault allocator knobs excluded."""

    expandable_segments: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.expandable_segments, bool):
            raise TypeError("expandable_segments must be bool")


@dataclass
class _Allocation:
    allocation_id: str
    requested: int
    size: int
    device: str
    pool: str
    stream: str
    segment: _Segment | None = None
    offset: int = 0
    active: bool = True
    tokens: set[str] = field(default_factory=set)


@dataclass
class _Segment:
    segment_id: int
    device: str
    pool: str
    stream: str
    size_class: str
    extent: int
    chunk_size: int  # Zero denotes a fixed segment.
    mapped_chunks: set[int] = field(default_factory=set)
    blocks: list[_Allocation] = field(default_factory=list)
    _mapped_ranges_cache: list[tuple[int, int]] | None = field(default=None, repr=False)

    @property
    def reserved(self) -> int:
        return len(self.mapped_chunks) * self.chunk_size if self.chunk_size else self.extent

    def free_ranges(self) -> list[tuple[int, int]]:
        """Coalesced virtual gaps, including unmapped pages for expandable segments."""
        ranges = []
        cursor = 0
        for block in sorted(self.blocks, key=lambda item: item.offset):
            if block.offset > cursor:
                ranges.append((cursor, block.offset))
            cursor = block.offset + block.size
        if cursor < self.extent:
            ranges.append((cursor, self.extent))
        return ranges

    def mapped_ranges(self) -> list[tuple[int, int]]:
        if not self.chunk_size:
            return [(0, self.extent)]
        if self._mapped_ranges_cache is not None:
            return self._mapped_ranges_cache
        ranges: list[tuple[int, int]] = []
        for chunk in sorted(self.mapped_chunks):
            start, end = chunk * self.chunk_size, (chunk + 1) * self.chunk_size
            if ranges and ranges[-1][1] == start:
                ranges[-1] = (ranges[-1][0], end)
            else:
                ranges.append((start, end))
        self._mapped_ranges_cache = ranges
        return ranges

    def cached_ranges(self) -> list[tuple[int, int]]:
        mapped = self.mapped_ranges()
        return [(max(start, mapped_start), min(end, mapped_end))
                for start, end in self.free_ranges()
                for mapped_start, mapped_end in mapped
                if max(start, mapped_start) < min(end, mapped_end)]


class CachingAllocatorModel:
    """Best-fit cache with split/coalesced gaps and explicit deferred frees.

    A completion token identifies caller-declared work, not an inferred CUDA
    stream event. ``complete`` clears existing dependencies, even on live blocks;
    unknown tokens are harmless. Use distinct tokens for distinct work epochs.
    Logical allocation ids may be reused after free while old blocks are pending.
    """

    def __init__(self, config: AllocatorConfig | None = None):
        if config is not None and not isinstance(config, AllocatorConfig):
            raise TypeError("config must be AllocatorConfig")
        self.config = config or AllocatorConfig()
        self._live: dict[str, _Allocation] = {}
        self._segments: list[_Segment] = []
        self._devices: set[str] = set()
        self._next_segment_id = 1

    def allocate(self, allocation_id: str, size_bytes: int, *, device: str = "cpu",
                 pool: str = "default", stream: str = "default") -> None:
        for name, value in (("allocation_id", allocation_id), ("device", device),
                            ("pool", pool), ("stream", stream)):
            _label(name, value)
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise TypeError("size_bytes must be an integer")
        if size_bytes < 0:
            raise ValueError("size_bytes must be nonnegative")
        if allocation_id in self._live:
            raise ValueError(f"duplicate allocation id: {allocation_id}")
        size = _round_up(size_bytes, 512)
        block = _Allocation(allocation_id, size_bytes, size, device, pool, stream)
        if size:
            size_class = "small" if size <= MiB else "large"
            matching = [segment for segment in self._segments
                        if (segment.device, segment.pool, segment.stream, segment.size_class)
                        == (device, pool, stream, size_class)]
            # Stable address tie-breaking makes the trace reproducible.
            candidates = [(end-start, segment.segment_id, start, segment)
                          for segment in matching
                          for start, end in segment.cached_ranges() if end-start >= size]
            if candidates:
                _, _, offset, segment = min(candidates, key=lambda entry: entry[:3])
            elif self.config.expandable_segments:
                if matching:
                    segment = matching[0]
                else:
                    segment = self._new_segment(device, pool, stream, size_class, 0,
                                                2*MiB if size_class == "small" else 20*MiB)
                # Remap existing holes first; otherwise grow the free tail. No
                # compaction occurs, including when physical pages were released.
                holes = [(end-start, start) for start, end in segment.free_ranges()
                         if end-start >= size]
                if holes:
                    _, offset = min(holes)
                else:
                    offset = max((item.offset+item.size for item in segment.blocks), default=0)
                segment.extent = max(segment.extent, _round_up(offset+size, segment.chunk_size))
                first = offset // segment.chunk_size
                last = (offset+size-1) // segment.chunk_size
                segment.mapped_chunks.update(range(first, last+1))
                segment._mapped_ranges_cache = None
            else:
                backing = (2*MiB if size <= MiB else
                           20*MiB if size < 10*MiB else _round_up(size, 2*MiB))
                segment = self._new_segment(device, pool, stream, size_class, backing, 0)
                offset = 0
            block.segment, block.offset = segment, offset
            segment.blocks.append(block)
        self._live[allocation_id] = block
        self._devices.add(device)

    def _new_segment(self, device: str, pool: str, stream: str, size_class: str,
                     extent: int, chunk_size: int) -> _Segment:
        segment = _Segment(self._next_segment_id, device, pool, stream,
                           size_class, extent, chunk_size)
        self._next_segment_id += 1
        self._segments.append(segment)
        return segment

    def free(self, allocation_id: str) -> None:
        block = self._get_live(allocation_id)
        del self._live[allocation_id]
        block.active = False
        if not block.tokens and block.segment is not None:
            block.segment.blocks.remove(block)

    def _get_live(self, allocation_id: str) -> _Allocation:
        _label("allocation_id", allocation_id)
        if allocation_id not in self._live:
            raise ValueError(f"unknown live allocation id: {allocation_id}")
        return self._live[allocation_id]

    def record_stream(self, allocation_id: str, token: str) -> None:
        block = self._get_live(allocation_id)
        _label("token", token)
        block.tokens.add(token)

    def complete(self, token: str) -> None:
        _label("token", token)
        for block in self._live.values():
            block.tokens.discard(token)
        for segment in self._segments:
            for block in segment.blocks:
                block.tokens.discard(token)
            segment.blocks[:] = [block for block in segment.blocks if block.active or block.tokens]

    def empty_cache(self, device: str | None = None) -> None:
        """Release eligible backing without implicitly completing pending work.

        Expandable virtual ranges remain available for remapping, including fully
        empty ones. Only mapped physical chunks contribute to reserved bytes.
        """
        if device is not None:
            _label("device", device)
        retained = []
        for segment in self._segments:
            if device is not None and segment.device != device:
                retained.append(segment)
                continue
            if segment.chunk_size:
                occupied_chunks = set()
                for block in segment.blocks:
                    first = block.offset // segment.chunk_size
                    last = (block.offset+block.size-1) // segment.chunk_size
                    occupied_chunks.update(range(first, last+1))
                segment.mapped_chunks.intersection_update(occupied_chunks)
                segment._mapped_ranges_cache = None
                retained.append(segment)
            elif segment.blocks:
                retained.append(segment)
        self._segments = retained

    @staticmethod
    def _totals() -> dict[str, int]:
        return dict(requested_bytes=0, allocated_bytes=0, pending_free_bytes=0,
                    reserved_bytes=0, cached_bytes=0, segment_count=0)

    def reserved_bytes(self) -> int:
        """Total mapped backing without constructing per-block address metadata."""
        return sum(segment.reserved for segment in self._segments)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Detached counters by device, pools, and synthetic address metadata.

        ``allocated_bytes`` counts rounded live blocks; ``pending_free_bytes``
        counts freed but unreusable blocks. ``cached_bytes`` is mapped safe free
        memory. ``segment_count`` counts segments with any physical backing, not
        retained empty expandable virtual ranges. Requested bytes exclude frees.
        """
        result: dict[str, dict[str, Any]] = {
            device: {**self._totals(), "pools": {}, "segments": []}
            for device in sorted(self._devices)
        }
        for block in self._live.values():
            stats = result[block.device]
            pool = stats["pools"].setdefault(block.pool, self._totals())
            for counters in (stats, pool):
                counters["requested_bytes"] += block.requested
                counters["allocated_bytes"] += block.size
        for segment in self._segments:
            stats = result[segment.device]
            pool = stats["pools"].setdefault(segment.pool, self._totals())
            pending = sum(block.size for block in segment.blocks if not block.active)
            occupied = sum(block.size for block in segment.blocks)
            for counters in (stats, pool):
                counters["reserved_bytes"] += segment.reserved
                counters["pending_free_bytes"] += pending
                counters["cached_bytes"] += segment.reserved - occupied
                counters["segment_count"] += int(segment.reserved > 0)
            stats["segments"].append({
                "segment_id": segment.segment_id, "pool": segment.pool,
                "stream": segment.stream, "size_class": segment.size_class,
                "virtual_extent_bytes": segment.extent, "reserved_bytes": segment.reserved,
                "chunk_size_bytes": segment.chunk_size,
                "mapped_ranges": [[start, end] for start, end in segment.mapped_ranges()],
                "cached_ranges": [[start, end] for start, end in segment.cached_ranges()],
                "blocks": [{"allocation_id": block.allocation_id,
                            "offset_bytes": block.offset, "size_bytes": block.size,
                            "requested_bytes": block.requested,
                            "state": "active" if block.active else "pending_free",
                            "tokens": sorted(block.tokens)}
                           for block in sorted(segment.blocks, key=lambda item: item.offset)],
            })
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "confidence": "deterministic source-informed approximation; not GPU calibrated",
            "source": SOURCE,
            "source_pin": "PyTorch v2.8.0 native CUDACachingAllocator (independent of runtime torch)",
            "rules": {
                "request_rounding_bytes": 512, "small_request_max_bytes": MiB,
                "small_segment_bytes": 2*MiB, "medium_request_below_bytes": 10*MiB,
                "medium_segment_bytes": 20*MiB, "large_segment_rounding_bytes": 2*MiB,
                "expandable_small_chunk_bytes": 2*MiB, "expandable_large_chunk_bytes": 20*MiB,
            },
            "assumptions": [
                "Best fit within explicit device, pool, stream and rounded small/large class.",
                "All reusable gaps split at rounded request boundaries; adjacent gaps coalesce.",
                "One expandable virtual range per device/pool/stream/size class; no live relocation.",
                "Expandable mapping first reuses mapped gaps, then holes, then grows the free tail.",
                "Completion tokens are explicit caller inputs; empty_cache does not synchronize them.",
                "Zero-size ids have no backing; segment_count counts physically backed segments.",
            ],
            "excluded": [
                "Exact native splitting thresholds and nondefault allocator configuration knobs.",
                "cudaMallocAsync, CUDA graph lifetime rules, IPC and peer mappings.",
                "CUDA stream/event inference, scheduling, automatic synchronization and GC.",
                "Device capacity, virtual-address limits, allocation failure/retry and OOM prediction.",
                "CUDA context, NCCL and other library allocations; provide them in external ledgers.",
                "Hardware measurements or byte-exact parity with any installed PyTorch build.",
            ],
        }
