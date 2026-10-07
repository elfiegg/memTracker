# Allocator and external-memory experiment

## Alignment review (2026-10-07)

Reviewed main at `c87380c`. The current design is aligned with the proposed
hybrid estimator: `ExtendedMemTracker` samples storage and external events on
one timeline; `NcclMemoryModel` already supplies persistent/lazy and temporary
allocations with explicit provenance. Payload tensors are not counted twice.
The K3 experiment correctly describes itself as a partial schedule.

The missing layer is backing allocation: live tensor storage is not allocator
reserved memory. Other CUDA/library memory also needs explicit lifecycle inputs.
No existing measurement validates GPU prediction accuracy.

## Design

Keep the existing report semantics and add an optional allocator report. Replay
storage events into a pure-Python approximation of the native CUDA caching
allocator: rounded requests, small/large pools, best-fit splitting/coalescing,
fixed segments or expandable page-backed ranges, and explicit deferred frees.
Pool/stream labels and completion points are supplied by the caller; no CUDA
execution or inferred timing. Empty-cache releases eligible backing only.

Combine modeled reserved backing with the existing NCCL ledger and separately
declared other-memory components at every event. Never add tensor bytes again
to reserved bytes. Preserve category breakdown and the state at the simultaneous
peak. Keep omitted terms and modeling assumptions visible; do not emit a GPU
fit guarantee or pretend this is a byte-exact implementation of the allocator.

The alternatives are (1) fixed percentage overhead, too insensitive to lifetimes
and fragmentation, and (2) complete CUDA/NCCL emulation, outside this experiment.
The recommended event model extends the existing architecture with inspectable
rules and deterministic tests. This implements the design approved in chat.

## Implementation and verification

- [x] Add `allocator_model.py` and pure CPU tests. Exercise best-fit reuse,
  fixed-segment fragmentation, expandable growth and page release, separate
  streams/pools/devices, rounding, delayed completion, and invalid transitions.
- [x] Integrate storage identity events in `tracker.py`; preserve legacy reports.
  Test aliases, resize, training, external registration, deferred storage reuse,
  and simultaneous reserved-plus-external peaks.
- [x] Add explicit other-memory profiles with provenance and lifecycle control.
  Existing NCCL profiles remain source/assumption based, with their exclusions.
- [x] Run deterministic fragmentation, overlapping collective, and lazy external
  allocation experiments; save JSON and explain numerical results in docs.
- [x] Extend the partial K3 scenario with opt-in allocator/other-memory inputs,
  leaving missing activations and full-training/OOM coverage explicitly unknown.
- [x] Add a guarded CUDA/NCCL calibration runner which records environment,
  phase measurements and snapshots. Validate its CPU-safe CLI and arithmetic.
  Completed two-rank GB200 calibration on Polyphe; both modes matched this
  trace's allocator checkpoints. External residual ownership remains unknown.
- [x] Review spec coverage, then code correctness; run the complete unittest
  suite and reproducible experiment commands. Publication is the final step.

Local test runtime: Python 3.12 with the repository-pinned torch 2.14.1. The
fresh clone and feature branch isolate this work from existing user checkouts.
