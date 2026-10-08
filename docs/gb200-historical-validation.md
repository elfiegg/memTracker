# GB200 historical validation audit

**Result: not validated.** The frozen estimator was not tuned to these observations.

Examined 9 runs: 6 OOM and 3 PASS. 0 match the frozen prediction's known configuration. No peak-error percentage or fit-accuracy score is justified.

| Job | Observed result | Schedule / sequence / microbatch × count | Main configuration differences |
|---|---|---|---|
| 3227951 | OOM (full_model) | 1F1B / 4068 / 2 × 64 | schedule |
| 3233459 | OOM (full_model) | 1F1B / 2048 / 2 × 64 | schedule, sequence |
| 3238922 | OOM (full_model) | 1F1B / 2048 / 2 × 64 | schedule, sequence |
| 3240283 | OOM (full_model) | 1F1B / 2048 / 1 × 128 | schedule, sequence, microbatch_size, microbatches |
| 3244815 | OOM (full_model) | Interleaved1F1B / 2048 / 1 × 128 | sequence, microbatch_size, microbatches, vision_encoder, cuda_graphs |
| 3250222 | OOM (full_model) | Interleaved1F1B / 1024 / 1 × 16 | sequence, microbatch_size, microbatches, vision_encoder, cuda_graphs, accumulation_rounds |
| 3244745 | PASS (reduced_debug_model) | Interleaved1F1B / 128 / 1 × 8 | sequence, microbatch_size, microbatches, layers, hidden_dim, pp, fsdp, ep, vision_encoder, cuda_graphs |
| 3250214 | PASS (reduced_debug_model) | Interleaved1F1B / 128 / 1 × 16 | sequence, microbatch_size, microbatches, layers, hidden_dim, pp, fsdp, ep, vision_encoder, cuda_graphs, accumulation_rounds |
| 3256150 | PASS (reduced_debug_model) | Interleaved1F1B / 128 / 1 × 16 | sequence, microbatch_size, microbatches, layers, hidden_dim, pp, fsdp, ep, vision_encoder, cuda_graphs, accumulation_rounds |

The existing estimator was rerun without changing its inputs; the complete report reproduced exactly.
Reproducibility establishes consistent execution, not predictive accuracy.
Jobs sharing the same known configuration: [[3233459, 3238922], [3250214, 3256150]]. Runtime warmup may still differ; repeated recipes must not be treated as independent held-out cases.

## Findings

The three PASS cases are 17-layer, width-256 debug models, not full K3 passes. Four OOM cases report rounded failure-point memory; two report NCCL OOM without a numeric memory snapshot. None is a completed full-model peak target.

| OOM job | Process GPU usage outside PyTorch reserved backing (GiB) | Above previous 16 GiB allowance (GiB) |
|---|---:|---:|
| 3227951 | 24.79 | 8.79 |
| 3233459 | 22.96 | 6.96 |
| 3238922 | 22.96 | 6.96 |
| 3244815 | 24.16 | 8.16 |

These are approximate same-process residuals: process usage − (allocated + reserved-unused). They are not exclusively NCCL. Sibling processes affect device free memory separately. The failed allocation is never added to the observed live total.

| Debug PASS job | Max logged active (GiB) | Max logged reserved (GiB) | Last logged active / reserved (GiB) |
|---|---:|---:|---:|
| 3244745 | 0.271772 | 0.455078 | 0.059652 / 0.279297 |
| 3250214 | 0.271837 | 0.455078 | 0.059717 / 0.259766 |
| 3256150 | 0.271837 | 0.412109 | 0.059717 / 0.259766 |

Recorded optimizer communication warmup: 16 snapshots; distinct outside-allocator deltas [0.0, 432.0] MiB. These debug observations do not establish full-model communication memory.

`memory/max_active(GiB)` reads allocator active bytes in the pinned training source; it is not necessarily the same as live tensor storage. The validator preserves per-rank logger windows and does not merge graph startup/capture/replay measurements into an assumed steady-state peak.

## Required before claiming success

- Add recipe-driven shape/topology/vision/schedule adapters, including PP4 debug and ordinary 1F1B.
- Represent first-step initialization, accumulation rounds and CUDA graph warmup/capture/replay separately.
- Model active allocator bytes including pending frees and mirror logger peak resets.
- Reconstruct eager communication and optimizer initialization from the recorded runtime; replace universal allowances with scoped calibration.
- Pin runtime/source and score held-out configuration groups after these adapters; repeated recipes must not straddle calibration and validation splits.

The proposed 5% allocated / 10% reserved targets remain untested. Missing comparisons are null, not zero error or correct fit predictions. The old 186.084 GiB result is for a different scenario and cannot be validated by numerical comparison to these OOM snapshots.

Input SHA-256: `94049c448e51368346c888dacdf3b7c1d2b9710aeff28acaef33ef556b452b45`.

## Reproduce the audit

```sh
PYTHONPATH=src python -m memtracker_nccl.datapoint_validation /path/to/datapoints.json \
  --prediction experiments/k3_training_lifetimes_expandable.json \
  --output experiments/gb200_historical_validation.json \
  --markdown docs/gb200-historical-validation.md
```

Add `--replayed-prediction /path/to/fresh-replay.json` to verify exact reproduction. The committed `experiments/gb200_historical_observations.json` is a compact normalized input with original file hash, complete-config fingerprints, per-rank metrics and extracted warmup snapshots; it can replace the raw input for the audit. The original large configurations remain in the supplied file. No GPU jobs or remote transfers are needed.
