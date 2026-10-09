# Framework memory capture

The measurement and attribution layer is independent of model classes, layer
counts, parameter names, and fitted workload offsets. TorchTitan supplies phase
boundaries and its effective configuration. PyTorch supplies allocator history,
stack traces, pool identifiers, and current counters.

The implementation follows real allocation/free events, including deferred frees
and expandable-segment mapping. Phase peaks include allocations carried over from
earlier phases. Allocation origin and the phase in which memory peaks are separate.
Private graph pools remain distinguishable. External memory is sampled at phase
boundaries and never silently attributed to NCCL or added to an unrelated peak.

Default capture does not reset PyTorch's peak counters, empty its cache, or synchronize
CUDA by default. Optional synchronized boundaries are a separate diagnostic mode
because they alter overlap. Allocator history and snapshots add CPU overhead.
Incomplete, unsupported, or truncated history must fail attribution explicitly.

Tests cover transient allocations, delayed frees, cache release, graph pools,
OOM censoring, nested phases, exception cleanup, and model-independent FakeTensor
phase tracking. CUDA runtime integration remains unverified until a GPU capture
is executed. Historical aggregate logs cannot be retroactively converted into
allocation traces.

## Use with TorchTitan

Make `memTracker/src` available on `PYTHONPATH` in the existing training container.
Do not install the project's pinned CPU PyTorch dependency over that container's
training runtime. Use the same selected TorchTitan checkout/config as the run:

```sh
python -m memtracker_nccl.torchtitan_capture \
  --output /path/to/capture --max-entries 100000 \
  --container /path/to/training.sqsh -- <existing TorchTitan CLI arguments>
```

Under an existing distributed launcher, replace the `torchtitan.train` module
with `memtracker_nccl.torchtitan_capture`; preserve the launcher rank environment.
It writes `rank-N-capture.json.gz` and `rank-N-analysis.json.gz`, including on
ordinary Python exceptions/OOM. A killed or hung process cannot guarantee a dump.
No job is submitted by this tool. Start with a reduced workload on one or two
allocated GPUs, or instrument an already authorized run.

The adapter checks the `TrainingEngine` interface before patching methods and
restores methods afterward. Captured config, engine source-file hash, observed
Git HEAD/dirty state, observed runtime and explicit run metadata are retained.
The public identity arguments remain one per component. A missing lifecycle
interface is an explicit incompatibility, not a fallback to Kimi assumptions.
Hooks run around initialization, prepare-step, forward/backward and optimizer
calls. Step and accumulation labels align repeated calls. CUDA graph internals
are traced by the allocator; the adapter does not insert checkpoints inside graph
capture or assume a universal number of graph-warmup steps.

The recorder owns PyTorch memory history: run it in a fresh process without
another history recorder. A filled ring buffer invalidates attribution; increase
the entry budget or shorten the run rather than accepting partial peaks.

## Read the report

PyTorch native allocation history records **requested sizes**, while active and
allocated counters use physical block sizes ([allocator source](https://github.com/pytorch/pytorch/blob/v2.8.0/c10/cuda/CUDACachingAllocator.cpp#L1486-L1506)).
The analyzer binds physical block sizes from boundary snapshots to allocation
generations, so reused addresses cannot inherit an earlier allocation's size.
Transient allocations that die between snapshots may have unknown block sizes.
Their requested-byte peaks remain available, but physical peaks stay null with
explicit lower bounds. Rounding a request does not prove the physical block size.

For a standalone experiment with exclusive ownership of peak-counter resets,
`TorchCudaProbe(isolated_peaks=True)` also measures exact allocator phase peaks.
This mode resets counters at boundaries and must not be combined with other
peak-resetting instrumentation or ordinary training-log comparisons. The small
CUDA demo uses it; the TorchTitan adapter does not. These counter peaks do not
identify the allocation stack at the peak instant.

- Each phase has independent requested/allocated/active/reserved/pending-free
  peaks, with unknown physical sizes explicitly marked. Trace-derived active
  peaks include live bytes grouped by allocating phase and call stack; counter
  peaks lack that instantaneous attribution.
- Parameter, gradient, buffer and optimizer `.state` storage is inventoried at
  boundaries. Aliases count once; cross-category aliases stay `shared`. DTensors
  use their local payload without gathering. Private optimizer buffers outside
  `.state` and unnamed kernel storage remain unattributed; stack traces can locate
  their allocation sites. An origin phase is not proof of semantic ownership.
- `active = allocated + pending_free`; `cached = reserved - active`. Cached bytes
  are not automatically fragmentation. Graph-private pools retain separate IDs.
- External deltas use the two actual boundary samples, even if communication
  produces no PyTorch allocator events. The default residual is device-wide.
  `TorchCudaProbe(process_memory_reader=...)` accepts an optional process-specific
  reader; unavailable measurements stay null. Never add an independent external
  maximum to an allocator maximum.
- Reconstructed counters are compared with boundary counters. Unsynchronized
  sampling is non-atomic; differences may reflect races or schema incompatibility.
  Comparison is blocked when these counters disagree.

## Compare an estimate

Use `with tracker.phase("forward_backward", step=1): ...` around the same code
under `ExtendedMemTracker`. Existing model code supplies shapes and lifetimes;
there is no model-name dispatch table or layer-count formula in this layer.
Then run:

```sh
python -m memtracker_nccl.phase_memory rank-0-capture.json.gz \
  --estimate fake-estimate.json --output comparison.json
```

Matching uses phase name, step/accumulation labels and occurrence. Missing phases
stay unknown; failed phases remain censored. Source/config/runtime equality must
still be established before claiming prediction accuracy. Comparison reports are
diagnostics and do not silently fit workload offsets.

`examples/framework_phase_demo.py` executes either an MLP or convolution model
with AdamW, with optional full activation checkpointing. Identical arguments can
be run in `--mode fake` and `--mode cuda` in separate fresh processes. CUDA mode
uses one GPU and creates no distributed group:

```sh
PYTHONPATH=src python examples/framework_phase_demo.py --mode fake \
  --family mlp --full-ac --expandable --output fake-estimate.json
```

## Validation so far

The analyzer exactly reconciles allocated, active and reserved counters in all
56 saved GB200 calibration snapshots (two ranks, fixed/expandable allocators,
14 checkpoints each). This validates **snapshot accounting**, not transient
CUDA capture or full training estimates. Synthetic event tests cover transient
peaks, stream-delayed frees, expandable mappings, graph-pool identity, OOMs and
truncated histories. CPU FakeTensor tests exercise both model families. New live
GPU capture is still required to validate the private history API and whole-model
attribution in the target container. The nine historical workload estimates are
unchanged; aggregate logs cannot supply their missing allocation histories.
