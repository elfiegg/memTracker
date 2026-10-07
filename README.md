# memTracker: tensor, allocator and NCCL memory estimation

A proof of concept combining PyTorch `FakeTensorMode` + `MemTracker` with
**modeled caching-allocator backing, NCCL allocations, and other external memory**.
The simulator runs without CUDA or NCCL. A separate optional GPU runner calibrates
selected checkpoints. It does not emulate NCCL or guarantee GPU memory fit.

The core is useful now; accurate NCCL prediction still requires a transport
configuration or calibration data. Unsupported overhead is reported explicitly.
The included Kimi K3 experiment is a **partial parameter/state/communication
estimate**, not a full forward/backward or OOM prediction.

## Install and test

Python 3.11+; tested with Python 3.12 and **PyTorch 2.14.1 CPU**. PyTorch private
APIs are pinned because their hooks can change.

```bash
python -m venv .venv
source .venv/bin/activate
# Linux/Windows: install a CPU wheel (macOS's regular torch wheel is CPU-compatible).
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/fake_collectives_demo.py --output experiments/collectives.json
```

On macOS, replace the CPU-index install command with
`python -m pip install torch==2.14.1`.

The included demo gives these CPU-simulated peaks for a 64 MiB shard and an
eight-rank all-gather. The same two sequential gathers reuse persistent buffers.
These are **scenario calculations**, not measured NCCL footprints.

| Scenario | Tensor peak | Modeled NCCL | Combined peak |
|---|---:|---:|---:|
| NCCL omitted | 576 MiB | 0 MiB | 576 MiB |
| P2P, 4 channels | 576 MiB | 48 MiB | 624 MiB |
| P2P, 16 channels | 576 MiB | 192 MiB | 768 MiB |

Machine-readable output: [experiments/collectives.json](experiments/collectives.json).

## What is implemented

- `ExtendedMemTracker`: an explicit `external_alloc` / `external_free` ledger.
  Samples combined tensor + external memory after **every tensor storage event**
  and external event. It does not add independently observed peaks.
- Optional `AllocatorConfig`: rounded requests, caching, fixed/expandable segments,
  separate pools/streams, and explicit deferred frees. `report()['resident']`
  samples **reserved backing + external memory**, without adding tensors again.
- `OtherMemoryModel`: explicit externally owned components, provenance and lifetimes.
  Context/library bytes have no built-in universal constant.
- `NcclMemoryModel`: communicator-scoped persistent buffers, lazy initialization,
  pool reuse, concurrent temporary allocations, and explicit completion/destruction.
- `FakeCollectiveRunner`: actual PyTorch fake `all_gather`, `reduce_scatter`, and
  `all_reduce` operations, with modeled NCCL allocations alongside them. Inputs
  remain referenced until the caller declares completion with `wait()` or
  `runner.wait_all()`, even if the user drops an individual work handle.
- NCCL transport profiles derived from pinned source, plus a calibration input.
  Payload tensors are already counted by MemTracker and are not added again.

```python
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from memtracker_nccl import ExtendedMemTracker
from memtracker_nccl.fake_collectives import FakeCollectiveRunner
from memtracker_nccl.nccl_model import NcclMemoryModel, p2p_profile

with FakeTensorMode(allow_fallback_kernels=False):
    tracker = ExtendedMemTracker()
    nccl = NcclMemoryModel(tracker)
    # Assumption: four P2P channels, one local send + receive connection/channel.
    nccl.register_communicator("dp", p2p_profile(channels=4), device="cpu")
    collectives = FakeCollectiveRunner(nccl)
    with tracker:
        shard = torch.empty(1024, dtype=torch.bfloat16)
        work = collectives.all_gather(shard, "dp", world_size=8)
        gathered = work.wait()
    nccl.destroy_communicator("dp")
    print(tracker.report()["peak"])
```

The `cpu` label is the **simulated tensor device**, not a claim that NCCL executes
on CPU. This prototype deliberately uses fake CPU tensors and labels the CUDA
execution differences as uncertainty. Assign external bytes to the same device
as the tensors they should be combined with.

For direct integration into another simulator:

```python
tracker.external_alloc("comm-buffer", 64 * 2**20, device="cpu", category="NCCL")
# Execute fake model operations; combine peaks while this allocation is live.
tracker.external_free("comm-buffer")
```

Use `tracker.report()` for combined results. Inherited `display_snapshot()` and
module snapshots remain **tensor-only**. The custom report retains a lifetime
peak across tracker contexts; `reset_mod_stats()` clears only module attribution.
Create a new tracker for a separate scenario. Tensor events are sampled always;
pass `record_tensor_events=True` to retain every event in the JSON timeline.

## Allocator and overhead experiments

```bash
python -m memtracker_nccl.allocator_experiment --output experiments/allocator_overhead.json
python -m memtracker_nccl.k3_experiment --allocator-mode expandable --other-memory-mib 512 \
  --output experiments/k3_512rank_expandable_backing.json
```

Enable backing modeling with
`ExtendedMemTracker(allocator_config=AllocatorConfig(expandable_segments=True))`.
Existing `report()['peak']` remains tensor-plus-external accounting;
`report()['resident']['peak']` is the simultaneous backing-plus-external estimate.
The real CUDA flag is `expandable_segments`, not `enable_segment`. Pool and stream
labels are explicit model inputs; they do not create real CUDA pools or infer
asynchronous completion.

A **two-rank GB200 calibration** matched the fragmentation trace at every sampled
allocator checkpoint: **68 MiB fixed vs 60 MiB expandable** reserved backing.
Other schedules can reserve more with expandable segments. Measured NCCL-associated
residuals changed at setup and first use, then stayed constant across repeated
collectives; they are not attributed exclusively to NCCL or generalized to K3.

See [experiment results, usage and calibration limits](docs/allocator-experiments.md),
[allocator rules](docs/allocator-model.md), and
[GPU measurements](experiments/gb200_calibration.json).

## Kimi K3 experiment

The model inventory follows TorchTitan commit
[`948d65c868c5fa8f0290bcf9e54b69f004721f54`](https://github.com/pytorch/torchtitan/tree/948d65c868c5fa8f0290bcf9e54b69f004721f54/torchtitan/models/kimi_k3).
It describes **2,779,931,738,208 parameter elements**, including all routed experts
and the vision encoder. It does not instantiate or execute the complete model.

```bash
python -m memtracker_nccl.k3_experiment \
  --pp 8 --dp 64 --ep 8 --prefetch 2 \
  --communicators 3 --nccl-mib-per-communicator 128 \
  --output experiments/k3_512rank_partial.json
```

For this **illustrative 512-rank scenario** (PP=8, DP=64, EP=8 overlays DP,
EDP=8; TP=CP=1), the CPU experiment reports:

| Quantity | Result |
|---|---:|
| Largest per-rank parameter/gradient/Adam state | 84.31 GiB |
| Partial peak, including two prefetched layer gathers and router buffers | 101.75 GiB |
| NCCL allowance included in that partial peak | 0.375 GiB |

The NCCL allowance is **assumed: three communicators × 128 MiB**, not calibrated.
The state assumption is FP32 parameters, FP32 gradients and two FP32 Adam moments
(16 bytes/parameter). Real PyTorch fake all-gathers materialize BF16 metadata for
the supplied layer schedule. EP is not multiplied into the world size or applied
as an extra divisor to already DP-sharded state.

**Activations and full training are not modeled.** This number excludes the
KDA/MLA activation graph, EP token dispatch/combine, pipeline scheduling, full
gradient/reduction buffers, optimizer temporaries, and GPU allocator overhead.
An optional `--activation-mib` is an explicit allowance, not an activation
prediction. Batch/sequence length and a training peak remain unknown in the JSON.
The nearly equal contiguous layer partition is an experiment assumption.

Full-model import was attempted on the CPU-only macOS environment but blocked by
TorchTitan's Triton dependency. This result therefore does not imply successful
full K3 FakeTensor execution. See [the K3 report](docs/k3-experiment.md) and
[machine-readable results](experiments/k3_512rank_partial.json).

## Interpretation and limits

This is an **opt-in model and schedule**, not automatic interception of arbitrary
NCCL calls in an existing program. The caller supplies communicator configuration,
collective order, and completion points. It does not infer hardware topology,
NCCL algorithm selection, channels, CUDA stream completion, or pool sharing
between communicators. Temporary bytes default to zero unless explicitly modeled.

A source-derived buffer profile covers only its named transport allocations.
The transport profile itself omits CUDA context/library allocations, allocator
fragmentation, unmodeled NCCL resources, and other transports such as NVLS.
Optional allocator/other-memory models cover only explicitly supplied assumptions. Tensor estimates also inherit
FakeTensor limitations: GPU-specific fused kernels, compiler memory planning,
data-dependent execution, and unsupported custom ops can change real usage.

Tests validate bookkeeping, actual fake collective shapes and tensor lifetimes,
training/backward/optimizer integration, alias handling, overlapping collectives,
persistent pool reuse, allocator backing, and external lifetimes. The small GPU
calibration validates selected allocator checkpoints; it does **not** establish
accuracy for arbitrary workloads or external-memory prediction.
Transport formulas and coverage: [NCCL model notes](docs/nccl-model.md).

Further calibration needs allocation ownership traces and held-out workloads
on fixed NCCL versions/topologies; see the experiment report above.
