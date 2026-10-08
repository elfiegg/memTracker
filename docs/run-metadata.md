# Run parameters: one per component

Use the same six names in JSON, profiles, and the tracker API. CLI flags use
hyphens instead of underscores.

| JSON / API | CLI | Value |
|---|---|---|
| `training_code` | `--training-code` | Training-code base commit (full Git SHA) |
| `torchao` | `--torchao` | TorchAO commit (full Git SHA) |
| `pytorch` | `--pytorch` | Full PyTorch version |
| `cuda` | `--cuda` | CUDA version reported by PyTorch |
| `nccl` | `--nccl` | NCCL version or full build string |
| `container` | `--container` | Container path, tag, URI, or digest |

For example, the supplied GB200 runs use:

```json
{
  "training_code": "53a45ee31d260bcaacaa319dae29080c68675a49",
  "torchao": "6352062064145e9ca0d37b05a677c420ba5a7bb6",
  "pytorch": "2.15.0.dev20260928+cu130",
  "cuda": "13.0",
  "nccl": "2.30.7+cuda13.3",
  "container": "/lustre/fsw/coreai_dlfw_dev/elfieg/kimi-k3-distmuon-20260928/containers/kimi-distmuon-hybridep.sqsh"
}
```

NCCL needs only one input. The application extracts the library version from
its build string internally. A container digest, if used, goes in `container`
itself (for example `registry/image@sha256:...`); there is no second parameter.
A base commit is recorded as supplied, without claiming knowledge of local code
changes. These inputs record the target environment; they do not automatically
fetch source, inspect remote containers, or change the implementation modeled.

## Defaults and overrides

For the K3 CLI: **individual CLI options > `--run-metadata` JSON file >
`--runtime-profile` defaults > unknown**. Each field is optional; `null` in JSON
suppresses its fallback. Fallback values are marked assumed in reports.

A default profile has the same six component names inside `values`:

```json
{
  "schema_version": 1,
  "name": "gb200-defaults",
  "values": {"pytorch": "2.15.0.dev20260928+cu130", "cuda": "13.0", "nccl": "2.30.7+cuda13.3"}
}
```

The K3 CLI defaults to the existing `k3-b4d5-cu130` profile. A supplied profile
replaces it; an empty `values` object disables fallback. The generic tracker
and historical audit have no automatic target defaults.

## Tracker API

```python
from memtracker_nccl import ExtendedMemTracker

tracker = ExtendedMemTracker(
    run_metadata={"pytorch": "2.15.0.dev20260928+cu130", "nccl": "2.30.7+cuda13.3"},
    runtime_profile={"schema_version": 1, "name": "defaults", "values": {"cuda": "13.0"}},
)
```

Target inputs remain separate from `simulator_environment`, which describes
local execution. Reports retain detailed provenance internally, including
origins, fallback assumptions, extracted NCCL build information and any supplied
container digest. These are output details, not additional user parameters.
The Python K3 lifetime runner accepts the resolved result of `resolve_metadata`.

## Historical runs

The [GB200 manifest](../experiments/gb200_run_metadata.json) uses the same six
names, with shared values and optional per-job overrides:

```json
{
  "schema_version": 1,
  "shared": {"nccl": "2.30.7+cuda13.3"},
  "runs": {"3256150": {"container": "/path/to/container.sqsh"}}
}
```

```sh
PYTHONPATH=src python -m memtracker_nccl.datapoint_validation /path/to/datapoints.json \
  --prediction experiments/k3_training_lifetimes_expandable.json \
  --run-metadata-manifest experiments/gb200_run_metadata.json \
  --output experiments/gb200_historical_validation.json
```

Embedded run fields override per-job manifest fields, which override shared
fields and then an explicitly selected profile. Normalized reports remain
readable with their detailed provenance intact. The original Downloads JSON
is unchanged.

The former `*_commit`, `*_version`, `*_build`, `*_identifier`, and `*_digest`
input names are removed. Existing saved reports remain readable; input JSON
and profiles should use the six names above. Unknown input names are rejected.

Matching declarations do not verify source contents or memory accuracy.
Configuration adapters, kernel coverage, and measurement alignment remain
separate requirements for validating the estimator.
