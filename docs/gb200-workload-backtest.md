# GB200 workload backtest

**The partial estimator does not yet validate full training peaks.**
Replayed 9 saved recipes and made 24 per-rank successful-run comparisons. No memory observations were used to tune predictions.

## Successful debug runs

Values below are MiB. Model columns are maxima across modeled ranks; observed columns are maxima across corresponding first logged windows.

| Job | Model active | Logged active | Model reserved | Logged reserved |
|---|---:|---:|---:|---:|
| 3244745 | 57.15 | 278.29 | 88.00 | 462.00 |
| 3250214 | 59.72 | 278.36 | 108.00 | 466.00 |
| 3256150 | 59.72 | 278.36 | 108.00 | 422.00 |

These are discrepancies between the partial reference and logs, not a percentage accuracy claim. Exact source equivalence and startup/metric windows remain unresolved. Graph capture/replay peaks are excluded.

## Full-model OOM constraints

Values are GiB on the recorded failure rank. Phase peaks are computed reference peaks; snapshots describe the failure instant.

| Job | Rank | Model training peak | Model optimizer peak | Allocated at failure | Outside allocator at failure |
|---|---:|---:|---:|---:|---:|
| 3227951 | 224 | 119.92 | 60.16 | 154.26 | 24.79 |
| 3233459 | 224 | 114.25 | 60.16 | 156.87 | 22.96 |
| 3238922 | 224 | 114.25 | 60.16 | 156.87 | 22.96 |
| 3240283 | 211 | 127.38 | 70.13 | unknown | unknown |
| 3244815 | 160 | 135.10 | 72.62 | 156.22 | 24.16 |
| 3250222 | 0 | 126.28 | 63.40 | unknown | unknown |

A below-capacity subtotal cannot classify these runs as fitting. Private communication memory, complete kernel activations/workspaces and several source lifetimes remain uncovered.

## Interpretation

- Reference training engine uses two eager optimizer steps before CUDA graph capture; no graph pools/capture/replay are modeled.
- First logged windows can include construction, warmup, or earlier peak history absent from this replay; differences are diagnostics, not accuracy scores.
- Initialized replay starts from a fresh allocator and initialized optimizer state, not the exact second-iteration allocator history.
- Active prediction is rounded live allocator storage with immediate free completion; GPU pending frees are not modeled.
- OOM snapshots remain censored; failed requested allocations are never added to observed memory.
- OOM runs replay the recorded failure rank only; successful debug runs cover every owner/rank. No global worst-rank claim is made for OOM runs.
- External CUDA/NCCL/HybridEP memory remains unpredicted, not zero; communication_gib=0 in replay artifacts isolates tensor/allocator accounting.
- Reference source b4d5b404 is not verified equivalent to historical base53a45 plus local changes.

Next validation work should explain the successful debug-run gaps first: align construction/warmup windows, execute whole-block activation paths, model stream completion and graph pool lifetimes, and establish source equivalence. Keep repeated recipes together when later separating calibration and held-out data.
