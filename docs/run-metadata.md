# Per-run source and runtime identity

Every estimate or observation can carry its target training source, TorchAO,
PyTorch, CUDA, NCCL and container identity. The target environment is separate
from the CPU environment executing FakeTensor. Metadata records assumptions and
compatibility; it does not install versions or choose a different memory model.

## Input fields

All fields are optional strings; `null` explicitly means unknown and blocks a
lower-priority fallback. Unknown field names are rejected to catch typos.

| Field | Meaning |
|---|---|
| `training_code_commit` | Exact training checkout, full 40-character Git commit |
| `training_code_base_commit` | Base revision, potentially with local changes |
| `training_code_patch_sha256` | SHA-256 of an accompanying local patch |
| `torchao_commit` | Full TorchAO Git commit |
| `pytorch_version` | Full version including dev date and CUDA suffix |
| `cuda_version` | CUDA version reported by PyTorch |
| `nccl_version` | NCCL library version |
| `nccl_build` | Full runtime build string, separate from PyTorch CUDA version |
| `container_identifier` | Container path, tag, or URI; treated as an opaque identifier |
| `container_digest` | Immutable content digest, `sha256:<64 lowercase hex characters>` |

For modified source, preserve the base commit and patch fingerprint; do not
substitute the base for an exact checkout. The fingerprint records identity but
does not establish equivalence with the estimator's extracted source. Paths and
tags can change contents; neither they nor digests automatically supply package
versions. Container-only inputs are accepted but leave version compatibility
incomplete. No remote container lookup or execution occurs.

## Tracker API

```python
from memtracker_nccl import ExtendedMemTracker

tracker = ExtendedMemTracker(
    run_metadata={
        "pytorch_version": "2.15.0.dev20260928+cu130",
        "cuda_version": "13.0",
        "nccl_version": "2.30.7",
        "nccl_build": "2.30.7+cuda13.3",
    },
    runtime_profile={
        "schema_version": 1,
        "name": "my-explicit-defaults",
        "values": {"cuda_version": "13.0"},
    },
)
report = tracker.report()
```

`run_metadata` overrides the profile. The generic tracker has **no automatic
target defaults**. `simulator_environment` describes the local PyTorch runtime;
the existing `torch_version` field remains its backwards-compatible alias.

## K3 estimate CLI

The existing `k3_training_lifetimes` CLI accepts `--run-metadata run.json`,
`--runtime-profile profile.json`, and individual field flags such as
`--training-code-commit`, `--training-code-base-commit`, `--torchao-commit`,
`--pytorch-version`, `--cuda-version`, `--nccl-version`, `--nccl-build`,
`--container-identifier`, `--container-digest`, and `--training-code-patch-sha256`.
The run file contains a plain object of the fields above. Profile files have the
`schema_version`, `name`, and `values` structure shown in the tracker example.

Priority: **CLI fields > run file > selected profile > unknown**. With no profile
selected, this K3 CLI explicitly falls back to `k3-b4d5-cu130`, matching its
existing source-derived implementation. These fallback fields are marked
assumed. A custom profile replaces this default; an empty `values` object
disables it. A run-file `null` also suppresses an individual fallback.

The Python `k3_training_lifetimes.run(..., run_metadata=...)` API accepts a
resolved object from `resolve_metadata`. The generic tracker accepts plain
field dictionaries as shown above.

Reports contain:

- `run_metadata`: resolved target values, field origins, assumed fields and
  overridden values.
- `modeled_metadata`: identity of the source-derived implementation actually used.
- `runtime_compatibility`: matching fields, differences and unknowns.

Compatibility can be `mismatch`, `incomplete`, `assumed_match`, or
`declared_match`. A different container path alone is not a mismatch; different
known content digests are. Even `declared_match` is not source-content verification
or a guarantee of coverage. Entering a newer version does not make the estimator
simulate its behavior. Unsupported inputs remain visibly incompatible or incomplete.

## Historical datasets

An experiment may embed a plain `run_metadata` object. Alternatively, pass
`--run-metadata-manifest manifest.json` to `datapoint_validation`:

```json
{
  "schema_version": 1,
  "shared": {"pytorch_version": "2.15.0.dev20260928+cu130"},
  "runs": {
    "3256150": {"nccl_version": "2.30.7"}
  }
}
```

Priority: **embedded run fields > per-job manifest fields > shared manifest
fields > explicitly selected `--runtime-profile` > unknown**. An audit does not
automatically apply the estimator's defaults to observed experiments. Unknown
job identifiers are errors. Explicit metadata can replace saved fallback
assumptions. Reprocessing normalized observations preserves their field origins.

The supplied [GB200 manifest](../experiments/gb200_run_metadata.json) records the
user's common identity for all nine runs. The original Downloads JSON is unchanged.
The full build string `2.30.7+cuda13.3` is retained alongside PyTorch CUDA `13.0`;
neither value is rewritten to match the other. The supplied TorchTitan base
`53a45ee31d260bcaacaa319dae29080c68675a49` remains distinct from the estimator's
exact extracted source `b4d5b404bd30dec67f86873203fbd4ba5f457531`.

```sh
PYTHONPATH=src python -m memtracker_nccl.datapoint_validation /path/to/datapoints.json \
  --prediction experiments/k3_training_lifetimes_expandable.json \
  --run-metadata-manifest experiments/gb200_run_metadata.json \
  --output experiments/gb200_historical_validation.json \
  --normalized-output experiments/gb200_historical_observations.json \
  --markdown docs/gb200-historical-validation.md
```

Success for this feature means provenance survives normalization and report
round-trips, explicit inputs beat defaults, missing values remain unknown, and
source/version differences cannot silently relabel the modeled implementation.
Tests cover these cases, tracker isolation, CLI inputs, and the nine-run manifest.
This feature does not resolve configuration adapters, CUDA graph lifetimes,
kernel workspaces, or metric-window alignment needed for accuracy validation.
