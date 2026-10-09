# Allocator approximation

`CachingAllocatorModel` replays explicit storage allocation/free events on CPU.
It models reserved backing independently of the live tensor ledger. The model
neither allocates GPU memory nor predicts OOM. Installed PyTorch version and the
source pin are independent.

The defaults come from [PyTorch v2.8.0 CUDACachingAllocator.cpp](https://github.com/pytorch/pytorch/blob/v2.8.0/c10/cuda/CUDACachingAllocator.cpp):
512-byte request rounding; small requests up to 1 MiB use 2 MiB segments; requests
below 10 MiB use 20 MiB segments; larger backing rounds to 2 MiB. Expandable
physical chunks are 2 MiB for small requests and 20 MiB for large requests.

```python
from memtracker_nccl.allocator_model import AllocatorConfig, CachingAllocatorModel

model = CachingAllocatorModel(AllocatorConfig(expandable_segments=True))
model.allocate("storage-1", 21 * 2**20, device="cuda:0")
model.record_stream("storage-1", "work-1")
model.free("storage-1")
model.empty_cache()             # Pending work still owns the backing.
model.complete("work-1")
model.empty_cache()             # Eligible physical chunks can now be released.
print(model.snapshot())
```

Allocation identity, device, pool, stream, and work tokens are explicit nonempty
strings. No stream timing is inferred. `complete(token)` clears dependencies on
both live and freed allocations. Repeated/unknown completion is harmless; callers
must use distinct tokens for distinct work epochs. A freed logical allocation id
may be reused even while its previous block awaits completion. Zero-byte
allocations retain logical identity but need no backing.

Best-fit reuse stays inside the device, pool, stream, and small/large size class.
Fixed segments are indivisible backing allocations: empty-cache releases only
segments with no live or pending blocks. Expandable segments retain synthetic
virtual ranges and map physical chunks on demand. Empty-cache unmaps only whole
chunks that overlap no live or pending block. Interior holes can be remapped, and
free tails can grow without relocating any occupied offset. Empty-cache never
implicitly completes a work token.

Snapshots report each device and pool separately:

- `requested_bytes`: exact live requests.
- `allocated_bytes`: rounded live requests.
- `pending_free_bytes`: rounded freed blocks awaiting completion.
- `reserved_bytes`: modeled physical backing.
- `cached_bytes`: reusable mapped free bytes.
- `segment_count`: segments with at least some physical backing.

The invariant is `reserved = allocated + pending_free + cached`. `segments`
contains synthetic offsets, occupied blocks, and mapped/cached half-open ranges;
retained empty virtual segments can appear here with zero reserved bytes. Reports
are detached JSON-compatible values.

The simulator splits every free gap at a rounded request boundary. It does not
reproduce native large-block minimum remainders, oversize rules, or nondefault
allocator knobs. Its expandable best-fit/mapping search is simplified. CUDA
malloc-async, graph/IPC ownership, allocator retries, automatic garbage
collection, virtual-address limits, device capacity, and hardware timing are
excluded. External NCCL/context/library allocations belong in separate ledgers.

The deterministic fragmentation test allocates three 16 MiB blocks, frees them,
then allocates three 14 MiB blocks and one 5 MiB block. Fixed segments reserve
68 MiB because three isolated 2 MiB tails cannot satisfy 5 MiB. Expandable backing
reserves 60 MiB. These are model outputs for this schedule, not a prediction of
universal savings; expandable chunk granularity can increase reservation for
other schedules. Additional tests cover deferred frees, multiple dependencies,
completion while live, pool/stream/device isolation, interior unmapping/remapping,
partial chunks, validation before mutation, and pointer stability.
