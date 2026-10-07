# Allocator, NCCL and other-memory experiments

The existing repository at `c87380c` agrees with the hybrid design: aliases are
tracked by storage, NCCL buffers have explicit lifetimes, and combined peaks are
sampled on a common timeline. The missing component was allocator backing. This
change adds that component without changing the meaning of existing reports.

## Accounting and use

At each event:

```
legacy combined = live tensor storage + external allocations
modeled resident = allocator reserved backing + external allocations
reserved backing = rounded live allocations + pending frees + reusable cache
```

Tensor bytes must not be added to reserved backing. Communication payloads that
use the PyTorch allocator are already represented there. Imported peer mappings
and registrations of existing memory are not new local physical backing.
`report()['peak']` retains legacy semantics; `report()['resident']['peak']`
contains simultaneous backing-plus-external peaks, including their event and
category breakdown. The word resident names a model output, not an NVML reading.

```python
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from memtracker_nccl import ExtendedMemTracker
from memtracker_nccl.allocator_model import AllocatorConfig
from memtracker_nccl.overhead_model import MemoryComponent, OtherMemoryModel

with FakeTensorMode(allow_fallback_kernels=False):
    tracker = ExtendedMemTracker(allocator_config=AllocatorConfig(expandable_segments=True))
    other = OtherMemoryModel(tracker)
    # Illustrative input only; there is no universal CUDA-context constant.
    context = MemoryComponent('context', 256 * 2**20, 'CUDA context',
                              'assumed', 'User scenario allowance; not measured')
    with tracker, other.scope(context):
        with tracker.allocation_scope(pool='communication', stream='comm'):
            payload = torch.empty(32 * 2**20, dtype=torch.uint8)
        tracker.record_tensor_stream(payload, 'collective-1')
        del payload
        tracker.complete_stream('collective-1')
        tracker.empty_allocator_cache()
    print(tracker.report()['resident']['peak'])
```

Pool labels isolate modeled allocations; they do not create a real
`torch.cuda.MemPool`, and PyTorch does not generally allocate all communication
payloads from a special communication pool. A caller can model explicitly
configured pools, but this model does not emulate their custom backing allocator.
The relevant real PyTorch flag is `expandable_segments:True` in
`PYTORCH_ALLOC_CONF`, not `enable_segment`. See the [allocator rules and
exclusions](allocator-model.md). A scope labels newly observed storage; a view
keeps its existing storage's pool. Stream completion is supplied explicitly,
not inferred from FakeTensor execution. `empty_allocator_cache()` does not
synchronize work; the real CUDA API can synchronize pending frees.

`OtherMemoryModel` requires category, byte size, evidence type and provenance.
Use persistent lifetimes for context-like overhead and a scope for externally
owned temporary workspaces. Workspaces allocated through PyTorch must instead
enter the allocator ledger. A device residual already containing NCCL must not
be added to a separate NCCL profile. Omitted components mean unknown, not zero.

## CPU scenarios

```bash
python -m memtracker_nccl.allocator_experiment --output experiments/allocator_overhead.json
```

The fragmentation trace allocates three 16 MiB blocks, frees them, allocates
three 14 MiB blocks, then allocates 5 MiB. Live storage is 47 MiB. Fixed backing
is 68 MiB because the three separate 2 MiB tails cannot satisfy 5 MiB. Expandable
backing is 60 MiB; contiguous free backing can serve that request.

A separate timeline uses a synthetic 4 MiB context, a 3 MiB externally owned
workspace at the large tensor's peak, an 8 MiB persistent NCCL allowance, and
two overlapping 2 MiB collective temporaries. It explicitly releases cached
backing before the later collectives. None of these external inputs is a
measured default.

| Schedule | Fixed backing + external peak | Expandable backing + external peak |
|---|---:|---:|
| NCCL initialized before 25 MiB tensor | 41 MiB | 55 MiB |
| NCCL initialized after tensor/cache release | 33 MiB | 47 MiB |

Expandable backing is larger here because the single 25 MiB tensor maps two
20 MiB chunks, versus 26 MiB fixed backing. The flag reduces some fragmentation
patterns; it is not a universal memory reduction. Lazy initialization changes
which overhead overlaps the peak. Results and full event timelines are in
[allocator_overhead.json](../experiments/allocator_overhead.json).

## GPU calibration

A short run on **2026-10-07**, one exclusively allocated Polyphe node, used two
GB200 GPUs connected through NV18. Runtime: PyTorch `2.15.0.dev20260928+cu130`,
CUDA 13.0, NCCL 2.30.7. CPU tracker tests use the repository's PyTorch 2.14.1;
allocator source rules are independently pinned to v2.8.0. The hardware runner
uses public measurement APIs and does not assume the private MemTracker hook
works on this different version. One fresh torchrun per allocator mode; two
ranks, one GPU per rank; NCCL network plugin disabled. This was not a multinode
or full-model calibration. Inherited environment settings included
`NCCL_MNNVL_CLIQUE_ID=-2`, `NCCL_MNNVL_CROSS_NVLD=1`,
`NCCL_MNNVL_CROSS_CLIQUE=1` and `NCCL_IB_SL=1`; selected settings are preserved
in the artifact. The container's `NCCL_VERSION=2.31.2` environment label differs
from the **loaded** library version 2.30.7 reported by PyTorch; the latter is
used here. Reproduction must match the runtime and settings, not only GPU type.

| Checkpoint | Fixed: predicted / measured reserved | Expandable: predicted / measured reserved |
|---|---:|---:|
| Three 16 MiB tensors | 48 / 48 MiB | 60 / 60 MiB |
| Three 14 MiB tensors reusing cache | 48 / 48 MiB | 60 / 60 MiB |
| Additional 5 MiB tensor | 68 / 68 MiB | 60 / 60 MiB |
| All tensors freed, empty cache | 0 / 0 MiB | 0 / 0 MiB |

Both ranks matched every allocated/reserved checkpoint. This validates this
small trace, not the excluded native allocator policies or arbitrary workloads.

For NCCL, a 32 MiB tensor remained live. The table shows **driver-used minus
PyTorch-reserved** memory. Fixed and expandable runs produced the same residuals.
These values include context, driver, libraries and NCCL; they are not a
measurement of NCCL ownership alone.

| Phase | Rank 0 residual | Rank 1 residual |
|---|---:|---:|
| Payload ready, before process group | 800.375 MiB | 800.375 MiB |
| Process group created | 1218.375 MiB | 1218.375 MiB |
| First 4 KiB all-reduce | 2470.188 MiB | 1778.375 MiB |
| Repeat small / first and repeat 32 MiB all-reduce | unchanged | unchanged |
| Second communicator, after first use | 3100.188 MiB | 2408.375 MiB |
| Both communicators destroyed | 1840.188 MiB | 1148.375 MiB |

Observed consequences:

- Communicator setup and first use both change the residual. The existing
  all-at-first-use NCCL profile is a useful simplified lifecycle, but it does
  not reproduce every observed setup phase.
- A repeated operation need not allocate another full set of persistent buffers.
- Teardown does not return the whole residual to its pre-NCCL value. We have not
  assigned the retained bytes to an owner; runtime/module retention and sampling
  timing are possible contributors requiring separate instrumentation.
- Rank asymmetry matters. Multiplying one constant by communicator count does
  not capture this run's complete lifecycle.
- NVML device usage exceeded the driver-used metric by 700.5625 MiB throughout
  these checkpoints. The runner records both; it does not treat them as
  interchangeable or explain that difference without evidence.

To replay these **observations**, use the full per-phase residual as one measured
external component, with its rank/version/topology provenance, and do not add
another NCCL estimate. To predict unseen runs, separate ownership with allocation
traces or supported NCCL statistics and calibrate on held-out shapes/topologies.
This experiment has not validated prediction of external bytes.

Reproduce on allocated CUDA GPUs with an installed CUDA PyTorch build:

```bash
export NCCL_NET_PLUGIN=none
PYTORCH_ALLOC_CONF=backend:native,expandable_segments:False \
  torchrun --standalone --nproc-per-node=2 scripts/calibrate_cuda.py --output work/gpu/fixed
PYTORCH_ALLOC_CONF=backend:native,expandable_segments:True \
  torchrun --standalone --nproc-per-node=2 scripts/calibrate_cuda.py --output work/gpu/expandable
python -m memtracker_nccl.calibration_analysis work/gpu \
  --output experiments/gb200_calibration.json --topology 'Describe measured placement here'
```

The runner emits rank JSON and memory snapshots at synchronized checkpoints.
These can miss transient asynchronous peaks. The comparison artifact contains
phase measurements and hashes of the original rank reports; cluster paths,
network addresses, and raw NCCL logs are not included in the repository.
See [gb200_calibration.json](../experiments/gb200_calibration.json).

## Optional K3 backing envelope

```bash
python -m memtracker_nccl.k3_experiment --allocator-mode fixed --other-memory-mib 512 \
  --output experiments/k3_512rank_fixed_backing.json
python -m memtracker_nccl.k3_experiment --allocator-mode expandable --other-memory-mib 512 \
  --output experiments/k3_512rank_expandable_backing.json
```

With the existing PP8/DP64/EP8 partial schedule, assumed three 128 MiB NCCL
components, and an **assumed 512 MiB other-memory component**, both backing
models report **109.605 GiB** at the worst stage. The legacy tensor-plus-external
peak is **102.247 GiB** including that allowance. The previous 101.747 GiB result
had no other-memory allowance. Equal backing peaks in this schedule do not mean
the segment layouts or behavior of the two policies are identical.

These remain partial inventory/lifetime envelopes. No full forward/backward,
activation graph, pipeline scheduling, topology discovery, or OOM verdict has
been added. The two-rank GPU residuals are deliberately **not** transplanted to
512 ranks. Detailed reports retain model assumptions and per-stage peaks.
