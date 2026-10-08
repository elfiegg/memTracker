"""Explicit target-run identity, separate from the machine running the simulator.

Defaults are assumptions, not detected versions. Metadata never installs a
runtime, selects a different implementation, or establishes memory accuracy.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

from .report_io import read_report

FIELDS = (
    "training_code_commit", "training_code_base_commit", "training_code_patch_sha256",
    "torchao_commit", "pytorch_version", "cuda_version", "nccl_version", "nccl_build",
    "container_identifier", "container_digest",
)

# Identity of the existing source-derived model; not defaults for historical runs.
K3_MODEL_PROFILE = {
    "schema_version": 1,
    "name": "k3-b4d5-cu130",
    "values": {
        "training_code_commit": "b4d5b404bd30dec67f86873203fbd4ba5f457531",
        "torchao_commit": "6352062064145e9ca0d37b05a677c420ba5a7bb6",
        "pytorch_version": "2.15.0.dev20260928+cu130",
        "cuda_version": "13.0",
        "nccl_version": "2.30.7",
    },
}


def validate_values(values):
    if not isinstance(values, dict):
        raise ValueError("Run metadata must be an object")
    unknown = set(values) - set(FIELDS)
    if unknown:
        raise ValueError(f"Unknown run metadata fields: {sorted(unknown)}")
    for key, value in values.items():
        if value is None:
            continue  # Explicit unknown masks a fallback.
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"{key} must be a nonempty, trimmed string or null")
        if key.endswith("commit") and not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"{key} must be a full lowercase 40-character Git commit")
        if key == "training_code_patch_sha256" and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("training_code_patch_sha256 must be a lowercase SHA-256")
        if key == "container_digest" and not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            raise ValueError("container_digest must be sha256:<64 lowercase hex characters>")
    return values


def validate_profile(profile):
    if (not isinstance(profile, dict) or set(profile) != {"schema_version", "name", "values"}
            or profile["schema_version"] != 1 or not isinstance(profile["name"], str)
            or not profile["name"].strip()):
        raise ValueError("Default profile requires schema_version=1, name, and values")
    validate_values(profile["values"])
    return profile


def resolve_metadata(*, profile=None, layers=(), existing=None):
    """Resolve low-to-high priority (origin, values) layers, then existing data.

`existing` preserves provenance when processing a normalized report again.
Every chosen field retains its origin; null explicitly blocks lower defaults.
"""
    result = dict(schema_version=1, values=dict.fromkeys(FIELDS), origins=dict.fromkeys(FIELDS),
                  assumed_fields=[], overrides=[])
    assumed = set()

    def put(origin, values, is_assumed=False):
        validate_values(values)
        for key, value in values.items():
            previous = result["values"][key]
            if result["origins"][key] is not None and previous != value:
                result["overrides"].append(dict(field=key, previous_value=previous,
                    previous_origin=result["origins"][key], value=value, origin=origin))
            result["values"][key] = value
            result["origins"][key] = origin
            assumed.discard(key)
            if is_assumed and value is not None:
                assumed.add(key)

    if existing is not None:
        validate_resolved(existing)
        for key in existing["assumed_fields"]:
            put(existing["origins"][key], {key: existing["values"][key]}, True)
    if profile is not None:
        validate_profile(profile)
        put(f"default_profile:{profile['name']}", profile["values"], True)
    for origin, values in layers:
        put(origin, values)
    if existing is not None:
        for key in FIELDS:
            if existing["origins"][key] is not None and key not in existing["assumed_fields"]:
                put(existing["origins"][key], {key: existing["values"][key]})
        history = copy.deepcopy(existing["overrides"])
        for override in result["overrides"]:
            if override not in history:
                history.append(override)
        result["overrides"] = history
    result["assumed_fields"] = sorted(assumed)
    return result


def validate_resolved(metadata):
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported resolved metadata schema")
    validate_values(metadata["values"])
    if set(metadata["values"]) != set(FIELDS) or set(metadata["origins"]) != set(FIELDS):
        raise ValueError("Resolved metadata must include every field and origin")
    if not set(metadata["assumed_fields"]) <= set(FIELDS):
        raise ValueError("Unknown assumed metadata fields")
    if any(metadata["values"][key] is None for key in metadata["assumed_fields"]):
        raise ValueError("Assumed fields must have known values")
    for key in FIELDS:
        origin = metadata["origins"][key]
        if origin is not None and (not isinstance(origin, str) or not origin):
            raise ValueError("Invalid metadata origin")
        if metadata["values"][key] is not None and origin is None:
            raise ValueError("Known metadata value has no origin")
    return metadata


def apply_dataset_metadata(normalized, *, profile=None, manifest=None):
    """Attach a shared/per-job sidecar without mutating the source observations."""
    result = copy.deepcopy(normalized)
    manifest = manifest if manifest is not None else {"schema_version": 1, "shared": {}, "runs": {}}
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or set(manifest) - {"schema_version", "shared", "runs", "description"}):
        raise ValueError("Unsupported run metadata manifest")
    shared, runs = manifest.get("shared", {}), manifest.get("runs", {})
    validate_values(shared)
    if not isinstance(runs, dict):
        raise ValueError("Manifest runs must be keyed by job identifier")
    jobs = {str(e["job"]) for e in result["experiments"]}
    if set(runs) - jobs:
        raise ValueError(f"Metadata references unknown jobs: {sorted(set(runs)-jobs)}")
    for e in result["experiments"]:
        job = str(e["job"])
        e["run_metadata"] = resolve_metadata(profile=profile,
            layers=[("manifest:shared", shared), (f"manifest:runs:{job}", runs.get(job, {}))],
            existing=e.get("run_metadata"))
    return result


def compare_metadata(observed, modeled):
    """Compare claims; matching strings do not verify source content or coverage."""
    validate_resolved(observed)
    validate_resolved(modeled)
    matches, differences, unknown = [], [], []
    for key in FIELDS:
        a, b = observed["values"][key], modeled["values"][key]
        if a is None or b is None:
            unknown.append(key)
        elif a == b:
            matches.append(key)
        else:
            differences.append(dict(field=key, observed=a, modeled=b))
    required = ("training_code_commit", "pytorch_version", "cuda_version", "nccl_version")
    assumed = sorted(set(observed["assumed_fields"]) | set(modeled["assumed_fields"]))
    # A mutable path is informative but cannot establish or refute compatibility.
    substantive = [d for d in differences if d["field"] != "container_identifier"]
    status = ("mismatch" if substantive else "incomplete" if any(k in unknown for k in required)
              else "assumed_match" if any(k in assumed for k in required) else "declared_match")
    return dict(status=status, matching_fields=matches, differences=differences,
                unknown_fields=unknown, assumed_fields=assumed, identity_verified=False,
                notes=["Base commit is not an exact training-code commit; local changes may exist.",
                       "Container paths are mutable identifiers; no versions are inferred from a path.",
                       "Matching declarations do not prove source equivalence or memory-model coverage."])


def modeled_metadata(prediction):
    if "modeled_metadata" in prediction:
        return validate_resolved(prediction["modeled_metadata"])
    # Backward-compatible identity for our existing frozen K3 reports only.
    commit = prediction.get("source_commit")
    values = ({**K3_MODEL_PROFILE["values"]} if commit == K3_MODEL_PROFILE["values"]["training_code_commit"]
              else {"training_code_commit": commit})
    return resolve_metadata(layers=[("model_source_profile", values)])


def add_metadata_arguments(parser):
    parser.add_argument("--run-metadata", type=Path, help="JSON object of explicit target-run identity fields")
    parser.add_argument("--runtime-profile", type=Path, help="JSON named default profile; fallback fields are assumptions")
    for key in FIELDS:
        parser.add_argument("--" + key.replace("_", "-"), help=f"Explicit target {key}; overrides metadata file")


def metadata_from_args(args, *, default_profile=None):
    profile = read_report(args.runtime_profile) if args.runtime_profile else default_profile
    explicit = {key: getattr(args, key) for key in FIELDS if getattr(args, key) is not None}
    return resolve_metadata(profile=profile, layers=[
        ("run_metadata_file", read_report(args.run_metadata) if args.run_metadata else {}),
        ("cli", explicit),
    ])
