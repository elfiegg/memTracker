# Running the branch

All relative commands below run from the memTracker repository root. Replace
placeholder paths and arguments with inspected inputs. Use a compatible CPU
environment for estimates. For GPU capture, expose `src` through `PYTHONPATH`
inside the existing training container; do not install the repository's pinned
CPU PyTorch dependency over its training runtime.

## Generic estimate

Start from `examples/framework_phase_demo.py`. Its MLP/conv families are test
fixtures only. Build a harness that invokes the target model, optimizer, and
checkpoint behavior under `FakeTensorMode(allow_fallback_kernels=False)`.

Minimal structure, where the uppercase helpers are application integration
points to implement from the actual training code:

```python
from torch._subclasses.fake_tensor import FakeTensorMode
from memtracker_nccl import ExtendedMemTracker
from memtracker_nccl.allocator_model import AllocatorConfig
from memtracker_nccl.report_io import write_report

with FakeTensorMode(allow_fallback_kernels=False):
    tracker = ExtendedMemTracker(
        allocator_config=AllocatorConfig(expandable_segments=True),
        run_metadata=run_metadata,
    )
    tracker.metadata["effective_config"] = effective_config
    with tracker:
        with tracker.phase("model_initialization"):
            model, inputs = BUILD_MODEL_AND_INPUTS(effective_config, device="cpu")
        with tracker.phase("optimizer_initialization"):
            optimizer = BUILD_ACTUAL_OPTIMIZER(model, effective_config)
        for step in range(1, steps + 1):
            tracker.reset_mod_stats()
            with tracker.phase("prepare_step", step=step):
                optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
            with tracker.phase("forward_backward", step=step):
                RUN_ACTUAL_FORWARD_BACKWARD(model, inputs, effective_config)
            with tracker.phase("optimizer", step=step):
                optimizer.step()
    write_report(output_path, tracker.report())
```

This sketch is a single-rank phase boundary pattern, not an implementation of
pipeline parallelism. The helper must preserve actual loss, accumulation,
checkpointing, and intermediate lifetimes; return/drop tensors just as training
does. Distributed execution needs explicit adapters and per-rank scheduling.
Do not silently replace unsupported operators or collectives with zero tensors.

Construct fake modules/tensors directly on CPU. Avoid moving tracked modules
with `.to(...)`, which can conflict with the private tracker's weak references.
For objects constructed before entering the tracker, use
`tracker.track_external(model, optimizer, inputs)`. Call `reset_mod_stats()`
between iterative calls; it clears module attribution, not allocator/session
history. Phase boundaries do not clear cached memory.

Allocator controls are explicit: `allocation_scope(pool=..., stream=...)`,
`record_tensor_stream(tensor, completion_token)`, `complete_stream(token)`, and
`empty_allocator_cache()`. Use them only when supported by execution evidence.
The simulator does not infer CUDA stream completion or create real MemPools.

Use `NcclMemoryModel` and `FakeCollectiveRunner` for supported communicator
scenarios; inspect `examples/fake_collectives_demo.py` and `nccl_model.py` first.
Use `OtherMemoryModel` or `external_alloc`/`external_free` for disjoint external
allocations with source-derived or measured sizes. Record communicator topology,
channels, transport, connection lifecycle, and profile compatibility. Destroying
a communicator and releasing a tensor are different lifetime events. Assign
external bytes to the same simulated device as their associated tensors.

## Small matched FakeTensor/CUDA probe

Run in separate fresh processes, preserving all workload flags:

```sh
PYTHONPATH=src python examples/framework_phase_demo.py \
  --mode fake --family mlp --width 32 --batch 2 --steps 3 \
  --full-ac --expandable --output /tmp/fake-estimate.json.gz
PYTHONPATH=src python examples/framework_phase_demo.py \
  --mode cuda --family mlp --width 32 --batch 2 --steps 3 \
  --full-ac --expandable --output /tmp/cuda-capture.json.gz
PYTHONPATH=src python -m memtracker_nccl.phase_memory \
  /tmp/cuda-capture.json.gz --estimate /tmp/fake-estimate.json.gz \
  --estimated-device cpu --output /tmp/comparison.json
```

CUDA mode uses one GPU without a process group. Run it only on an allocated GPU.
This demo uses exclusive peak-counter resets (`isolated_peaks=True`); do not
reuse that setting alongside normal training logs or another peak-reset owner.
Repeat without `--expandable` to check fixed segments. The demo sets allocator
configuration before CUDA initialization; other allocator knobs are unsupported.

## Real TorchTitan capture

Under the existing distributed launcher, replace module `torchtitan.train` with:

```sh
python -m memtracker_nccl.torchtitan_capture \
  --output /path/capture --max-entries 100000 \
  --container /path/training.sqsh \
  --run-metadata /path/run.json -- <existing TorchTitan CLI arguments>
```

Preserve launcher rank environment, selected checkout, effective config, and
training runtime. The adapter checks `TrainingEngine` lifecycle interfaces and
fails on incompatible code. It does not submit a job. Use an authorized small
probe or instrument an already authorized training run; do not scale to hundreds
of GPUs merely to estimate memory.

Outputs are `rank-N-capture.json.gz` and `rank-N-analysis.json.gz`. The adapter
captures initialization, prepare-step, forward/backward and optimizer phases,
plus source/config/runtime identity and boundary storage ownership. Ordinary
Python errors also attempt a dump; process kills/hangs cannot guarantee one.

Default capture owns allocator history in a fresh process, uses the native
allocator, and does not synchronize, empty cache, or reset peak counters.
`--synchronize` changes overlap and is a separate diagnostic. A filled history
ring invalidates attribution: shorten the run or increase `--max-entries`.

Compare each captured rank with its matching phase-labeled estimate:

```sh
PYTHONPATH=src python -m memtracker_nccl.phase_memory \
  /path/capture/rank-0-capture.json.gz \
  --estimate /path/rank-0-estimate.json.gz --estimated-device cpu \
  --output /path/rank-0-comparison.json
```

Consult repository `docs/framework-memory-capture.md` for the selected revision's
capture limitations and current validation evidence.

## Saved GB200 workloads only

For supported historical recipes, use the source-checked adapter:

```sh
PYTHONPATH=src python -m memtracker_nccl.workload_backtest /path/datapoints.json \
  --run-metadata-manifest experiments/gb200_run_metadata.json \
  --source /path/pinned-torchtitan \
  --pytorch-schedules /path/gpu-pipeline-schedules.py \
  --evidence-dir /path/replay-evidence \
  --output /path/backtest.json --markdown /path/backtest.md
```

Read repository `docs/gb200-workload-reproduction.md` for required source commit,
schedule hash, config support, and coverage. Alternatively use
`experiments/gb200_workload_recipes.json.gz` as input and add
`--observations experiments/gb200_historical_observations.json`.
Do not bypass source/config hash guards or present this adapter as support for
arbitrary models. Preserve the original experiment JSON. Aggregate logs cannot
recover missing allocation histories.
