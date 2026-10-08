# Reproducing the GB200 workload backtest

This backtest generates predictions from saved configurations, then compares
them with observations. It does not fit a communication allowance or activation
constant to the measured peaks. It remains a **partial source-derived reference**,
not a complete CUDA training simulator.

## Run

Use the original supplied JSON:

```sh
PYTHONPATH=src python -m memtracker_nccl.workload_backtest /path/to/datapoints.json \
  --run-metadata-manifest experiments/gb200_run_metadata.json \
  --source /path/to/pinned-torchtitan \
  --pytorch-schedules /path/to/gpu-pipeline-schedules.py \
  --evidence-dir experiments/gb200_replay_evidence \
  --output experiments/gb200_workload_backtest.json \
  --markdown docs/gb200-workload-backtest.md
```

Or replace the first argument with `experiments/gb200_workload_recipes.json.gz`
and add `--observations experiments/gb200_historical_observations.json`. The
portable recipe file retains model geometry, optimizer settings/layouts,
parallelism, training, checkpoint, compilation and loss settings. Initializer
and sharding reprs and unrelated I/O settings are omitted. Original full-config
fingerprints and independent retained-config fingerprints are both checked and
reported. The original Downloads file is unchanged.

The local source must be the pinned reference TorchTitan commit
`b4d5b404bd30dec67f86873203fbd4ba5f457531`, checked out in Git or unpacked using
the existing probe archive/job-record convention. The supplied PyTorch schedule
source has SHA-256
`61c4e103e3bcb17c11c4dc46db138d2c3400b76e3a8b2d4797cf106e51e931da`.
Source helpers and constructors are hash checked before execution. Neither is
downloaded automatically. Tests use PyTorch 2.14.1 CPU; target scheduling source
is from PyTorch 2.15.0.dev20260928+cu130.

## What is modeled

- Saved linear, grouped-linear, normalization, convolution and vision parameter
  shapes; full-model inventory matches the earlier 2,679 parameter entries.
- Saved optimizer patterns, compute layouts and buckets, executed through the
  reference DistMuon planner. Descriptor strings use a restricted AST parser;
  arbitrary configuration code is never evaluated.
- BF16 parameters and optimizer gradients, FP32 full accumulated gradients,
  eager optimizer redistribution buffers, lazy first-step momentum/AdamW states,
  and actual CPU FakeTensor Newton–Schulz/update/AdamW traces.
- Source-executed normal 1F1B and interleaved schedules with saved stage splits,
  microbatch counts, prefetch settings, and attention-residual block sizes.
  Normal 1F1B traces the actual executor: the pinned visualization helper starts
  the last rank at F1 and is unsuitable for a complete allocation replay.
- FullAC checkpoint inputs and one exact attention-residual helper per layer,
  FSDP gathered parameter copies, accumulation rounds, and fixed/expandable
  allocator storage lifetimes. Rounded live bytes are reported separately from
  requested tensor bytes and reserved segments.

For successful debug runs, every PP/DP rank is modeled. For OOM runs, only the
recorded failing rank is modeled. The source mesh order maps rank to PP then DP;
the OOM comparisons do not establish a global maximum over unobserved ranks.
Repeated tensor recipes can reuse predictions, but each run's metadata and
compatibility comparison are attached independently.

The generated report lists SHA-256 hashes for its compressed evidence files:
source optimizer plans/traces, FullAC helper traces, schedules, and first-step or
initialized-state replays. All source probes run locally on CPU. No GPU jobs are
submitted.

## Limits and success criteria

The historical runs name base commit `53a45ee31d260bcaacaa319dae29080c68675a49`.
That is not proof of equivalence to the extracted reference implementation.
Serialized configs also omit Python class tags; notably, router histogram
buffers cannot be established from absent `num_bins` alone. These uncertainties
remain explicit in each probe report.

The reference engine uses two eager optimizer steps before graph capture.
First logged peaks may include model construction or earlier startup activity
that this replay omits. The initialized-state replay starts from a fresh
allocator; it is not an exact simulation of the second window's cached memory.
Whole MLA/KDA/MoE activation graphs, kernel-private workspaces, pending CUDA
frees, complete residual caches, CUDA graph pools, and private communication
allocations remain uncovered. External memory is **unknown**, not predicted
zero; replay `communication_gib=0` isolates tensor/allocator accounting.

The successful debug runs provide 24 per-rank first-window diagnostics. Their
large gaps mean this partial model cannot yet serve as a full-peak estimator.
The six OOMs constrain failure phases; they are not successful-step peaks and
cannot supply ordinary peak-error percentages. No accuracy percentage or
correct-fit count is claimed.

For validation success, first establish equivalent source and matching logger
reset windows, cover the omitted lifetimes, then evaluate the proposed 5% active
and 10% reserved tolerances on held-out recipe groups. Repeated configurations
must stay together when separating calibration and validation. Passing CPU
bookkeeping tests alone does not satisfy these criteria.
