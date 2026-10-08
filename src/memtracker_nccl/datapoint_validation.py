"""Audit historical GB200 observations against a frozen K3 estimator report.

OOM snapshots are censored observations, not completed-iteration peak targets.
Configuration mismatches and missing metric/window coverage prohibit scoring.
This evaluator does not tune the estimator or execute code stored in configs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from .report_io import read_report, write_report

GiB = 2**30
ACTIVE = "memory/max_active(GiB)"
RESERVED = "memory/max_reserved(GiB)"


def _number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Expected a nonnegative finite observation or null, got {value!r}")
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _configuration(experiment):
    c = experiment["configuration"]
    p, t = c["parallelism"], c["training"]
    effective = experiment["effective_config"]
    # Refuse a flattened summary that disagrees with the actual saved recipe.
    if p != effective["parallelism"] or t != effective["training"]:
        raise ValueError("Configuration summary differs from saved training/parallelism configuration")
    model = effective["model"]
    checks = ((c["layers"], len(model["layers"])), (c["hidden_dim"], model["dim"]),
              (c["vocab_size"], model["vocab_size"]),
              (c["vision_encoder_enabled"], model["vision_encoder"] is not None))
    if any(a != b for a, b in checks):
        raise ValueError("Model summary differs from saved model configuration")
    return dict(layers=c["layers"], hidden_dim=c["hidden_dim"], vocab_size=c["vocab_size"],
                experts=c["experts"], top_k=c["top_k"], expert_input_dim=c["expert_input_dim"],
                expert_ffn_dim=c["expert_ffn_dim"], vision_encoder=c["vision_encoder_enabled"],
                sequence=c["sequence_length"], microbatch_size=c["microbatch_sequences"],
                microbatches=p["num_pp_microbatches"], accumulation_rounds=c["accumulation_rounds"],
                pp=p["pipeline_parallel_degree"], fsdp=p["data_parallel_shard_degree"],
                ep=p["expert_parallel_degree"], tp=p["tensor_parallel_degree"],
                cp=p["context_parallel_degree"], replication=p["data_parallel_replicate_degree"],
                schedule=p["pipeline_parallel_schedule"], cuda_graphs=not t["disable_cuda_graphs"],
                parameter_dtype=t["mixed_precision_param"], storage_dtype=t["dtype"],
                reduction_dtype=t["mixed_precision_reduce"], checkpointing=c["activation_checkpoint_type"],
                optimizer_types=c["optimizer_types"], reshard_after_forward=p["fsdp_reshard_after_forward"],
                forced_load_balance=c["forced_load_balance"])


def _warmup(experiment):
    rows = []
    pattern = re.compile(r"before=(\{[^{}]+\}) after=(\{[^{}]+\})")
    for entry in experiment.get("log_evidence", []):
        if "DistMuon communication warmup complete" not in entry["text"]:
            continue
        match = pattern.search(entry["text"])
        if not match:
            continue
        before, after = (ast.literal_eval(v) for v in match.groups())
        for snapshot in (before, after):
            for key in ("free", "total", "allocated", "reserved", "device_used_minus_reserved"):
                if _number(snapshot[key]) is None:
                    raise ValueError("Incomplete warmup byte snapshot")
            if snapshot["device_used_minus_reserved"] != snapshot["total"]-snapshot["free"]-snapshot["reserved"]:
                raise ValueError("Warmup residual does not reconcile with total/free/reserved")
        rows.append(dict(file=Path(entry["file"]).name, line=entry["line"], before=before, after=after,
                         external_delta_bytes=after["device_used_minus_reserved"]-before["device_used_minus_reserved"],
                         allocated_delta_bytes=after["allocated"]-before["allocated"],
                         reserved_delta_bytes=after["reserved"]-before["reserved"]))
    return rows


def normalize_dataset(dataset, *, source_sha256=None):
    """Retain observations/config summaries without copying multi-megabyte recipes."""
    if dataset["schema_version"] != 1:
        raise ValueError("Unsupported dataset schema")
    jobs = set(); rows = []
    for e in dataset["experiments"]:
        if e["job"] in jobs:
            raise ValueError("Duplicate job")
        jobs.add(e["job"])
        failure = {k: _number(v) for k, v in e["failure_point_memory_gib"].items()}
        metrics = e["per_rank_step_memory_metrics"]
        for rank, values in metrics.items():
            if len(values[ACTIVE]) != len(values[RESERVED]):
                raise ValueError(f"Unaligned memory metric windows on rank {rank}")
            for values_list in values.values():
                for value in values_list:
                    _number(value)
        rows.append(dict(job=e["job"], outcome=e["outcome"], scope=e["scope"], phase=e["phase"],
                         world_size=e["world_size"], representative_rank=e["representative_rank"],
                         configuration=_configuration(e), effective_config_sha256=_hash(e["effective_config"]),
                         failure_point_memory_gib=failure, per_rank_step_memory_metrics=metrics,
                         runtime_environment_evidence=e["runtime_environment_evidence"],
                         warmup_observations=_warmup(e),
                         log_references=[dict(file=Path(x["file"]).name, line=x["line"])
                                         for x in e["log_evidence"]]))
    return dict(schema_version=1, kind="normalized_gb200_validation_observations",
                source_sha256=source_sha256, source_generated_at=dataset.get("generated_at"),
                interpretation=dataset["interpretation"], experiments=rows)


def estimator_configuration(prediction):
    """Known capability of k3_training_lifetimes, not a general model adapter."""
    s = prediction["scenario"]
    return dict(layers=93, hidden_dim=7168, vocab_size=163840, experts=896, top_k=16,
                expert_input_dim=3584, expert_ffn_dim=3072, vision_encoder=True,
                **{k: s[k] for k in ("sequence", "microbatch_size", "microbatches", "pp", "fsdp", "ep", "tp", "cp", "schedule", "cuda_graphs")},
                accumulation_rounds=1, replication=1, parameter_dtype="bfloat16", storage_dtype="bfloat16",
                reduction_dtype="float32", checkpointing="FullAC", optimizer_types=["DistMuon", "AdamW"],
                reshard_after_forward="always", forced_load_balance=True)


def failure_accounting(memory):
    memory = {k: _number(v) for k, v in memory.items()}
    allocated = memory["torch_allocated_gib"]
    unused = memory["torch_reserved_unused_gib"]
    process = memory["process_gpu_gib"]
    reserved = None if allocated is None or unused is None else allocated+unused
    external = None if reserved is None or process is None else process-reserved
    if external is not None and external < -0.02:
        raise ValueError("Process usage is smaller than allocator backing beyond rounding tolerance")
    free, capacity = memory["device_free_gib"], memory["device_capacity_gib"]
    return dict(torch_reserved_gib=reserved, process_outside_allocator_gib=external,
                device_used_gib=None if free is None or capacity is None else capacity-free,
                metric_kind="rounded_failure_point_snapshot_not_completed_peak",
                completed_iteration_peak_gib=None,
                requested_allocation_included_in_allocated=False,
                attribution="Unattributed process GPU memory outside PyTorch reservation; not NCCL-only")


def step_windows(metrics):
    result = {}
    for metric in (ACTIVE, RESERVED):
        for values in metrics.values():
            for value in values[metric]:
                _number(value)
        observed = [(v, rank, i) for rank, values in metrics.items()
                    for i, v in enumerate(values[metric]) if v is not None]
        if not observed:
            result[metric] = None
            continue
        peak, rank, index = max(observed, key=lambda x: x[0])
        first = [v[metric][0] for v in metrics.values() if v[metric] and v[metric][0] is not None]
        last = [v[metric][-1] for v in metrics.values() if v[metric] and v[metric][-1] is not None]
        result[metric] = dict(max_logged_window_gib=peak, max_rank=rank, max_window_index=index,
                              first_logged_window_max_gib=max(first) if first else None,
                              last_logged_window_max_gib=max(last) if last else None,
                              observations=len(observed),
                              window_semantics="Logger windows; no inferred mapping to whole-iteration/graph-capture peaks")
    return result


def audit(normalized, prediction, *, replayed_prediction=None):
    expected = estimator_configuration(prediction)
    allowance = prediction["communication_gib"]
    comparisons = []; residuals = []
    for e in normalized["experiments"]:
        mismatches = [dict(field=key, measured=value, estimator=expected[key])
                      for key, value in e["configuration"].items() if value != expected[key]]
        failure = failure_accounting(e["failure_point_memory_gib"])
        external = failure["process_outside_allocator_gib"]
        if external is not None:
            residuals.append(dict(job=e["job"], process_outside_allocator_gib=external,
                                  minus_previous_allowance_gib=external-allowance))
        blockers = []
        if mismatches:
            blockers.append("configuration_mismatch")
        if e["outcome"] == "OOM":
            blockers.append("censored_failure_no_completed_peak")
        if e["outcome"] not in ("PASS", "OOM"):
            blockers.append("unmeasured_or_unsupported_outcome")
        if e["configuration"]["cuda_graphs"]:
            blockers.append("graph_warmup_capture_replay_windows_not_modeled")
        if not prediction.get("full_model_fit_verified", False):
            blockers.append("estimator_missing_full_training_components")
        # Dataset does not provide an exact runtime/source pin, matched metric
        # scope, or aligned operation/step window for our frozen prediction.
        blockers.extend(["source_runtime_identity_unverified", "metric_and_window_alignment_unverified"])
        comparisons.append(dict(job=e["job"], outcome=e["outcome"], scope=e["scope"], phase=e["phase"],
                                configuration=e["configuration"], mismatches=mismatches, blockers=blockers,
                                comparison_status="not_scored", peak_error_percent=None,
                                fit_classification_correct=None, observed_failure=failure,
                                logged_step_windows=step_windows(e["per_rank_step_memory_metrics"])))
    counts = Counter(e["outcome"] for e in normalized["experiments"])
    groups = {}
    for e in normalized["experiments"]:
        groups.setdefault(_hash(e["configuration"]), []).append(e["job"])
    max_rank = max(prediction["ranks"], key=lambda r: r["modeled_live_peak"]["bytes"])
    warmup = [w for e in normalized["experiments"] for w in e["warmup_observations"]]
    if replayed_prediction is not None and prediction != replayed_prediction:
        raise ValueError("Frozen estimator replay differs from the saved prediction")
    return dict(schema_version=1, status="not_validated", estimator_tuned=False,
                frozen_prediction_reproduced=None if replayed_prediction is None else True,
                source_dataset_sha256=normalized["source_sha256"], estimator_report_sha256=_hash(prediction),
                estimator_configuration=expected,
                frozen_estimator=dict(source_commit=prediction["source_commit"],
                    worst_live_peak_gib=max_rank["modeled_live_peak"]["bytes"]/GiB,
                    resident_communication_allowance_gib=allowance, pp_rank=max_rank["pp_rank"]),
                counts=dict(experiments=len(comparisons), outcomes=dict(counts),
                    known_configuration_matches=sum(not e["mismatches"] for e in comparisons),
                    scored_peak_comparisons=0, scored_fit_classifications=0),
                jobs_sharing_known_configuration=[jobs for jobs in groups.values() if len(jobs)>1],
                accuracy=dict(allocated_peak_error_percent=None, reserved_peak_error_percent=None,
                              false_fit_rate=None, success=False),
                external_memory_observations=residuals,
                communication_warmup=dict(samples=len(warmup),
                    distinct_external_delta_bytes=sorted({w["external_delta_bytes"] for w in warmup}),
                    all_allocator_allocated_deltas_zero=all(w["allocated_delta_bytes"] == 0 for w in warmup) if warmup else None,
                    scope="Observed debug-rank warmup deltas; not a 32-rank full-model NCCL calibration"),
                metric_semantics=["OOM torch_allocated values are failure-point snapshots, never completed-step peaks.",
                    "The failed requested allocation is not added to observed allocated memory.",
                    "Reserved = allocated + reserved-unused, not allocated + reserved.",
                    "Pinned TorchTitan DeviceMemoryMonitor reads active_bytes.all.peak for memory/max_active(GiB); do not assume requested live tensor bytes are identical.",
                    "Logger windows with CUDA graphs must be aligned before scoring; first/last logged windows are not automatically named warmup/steady state.",
                    "External process residuals include CUDA/NCCL/HybridEP/library memory; no exclusive attribution is inferred."],
                next_requirements=["Add recipe-driven shape/topology/vision/schedule adapters, including PP4 debug and ordinary 1F1B.",
                    "Represent first-step initialization, accumulation rounds and CUDA graph warmup/capture/replay separately.",
                    "Model active allocator bytes including pending frees and mirror logger peak resets.",
                    "Reconstruct eager communication and optimizer initialization from the recorded runtime; replace universal allowances with scoped calibration.",
                    "Pin runtime/source and score held-out configuration groups after these adapters; repeated recipes must not straddle calibration and validation splits."],
                experiments=comparisons)


def markdown(report):
    c = report["counts"]
    lines = ["# GB200 historical validation audit", "",
        "**Result: not validated.** The frozen estimator was not tuned to these observations.", "",
        f"Examined {c['experiments']} runs: {c['outcomes'].get('OOM',0)} OOM and {c['outcomes'].get('PASS',0)} PASS. "
        f"{c['known_configuration_matches']} match the frozen prediction's known configuration. "
        "No peak-error percentage or fit-accuracy score is justified.", "",
        "| Job | Observed result | Schedule / sequence / microbatch × count | Main configuration differences |",
        "|---|---|---|---|"]
    priority = ("schedule", "sequence", "microbatch_size", "microbatches", "layers", "hidden_dim", "pp", "fsdp", "ep", "vision_encoder", "cuda_graphs", "accumulation_rounds")
    for e in report["experiments"]:
        s=e['configuration']; diff={m['field'] for m in e['mismatches']}
        names=', '.join(k for k in priority if k in diff)
        lines.append(f"| {e['job']} | {e['outcome']} ({e['scope']}) | {s['schedule']} / {s['sequence']} / {s['microbatch_size']} × {s['microbatches']} | {names or 'No known-field differences; runtime/coverage still unverified'} |")
    lines += ["", "The existing estimator was rerun without changing its inputs; "
        + ("the complete report reproduced exactly." if report['frozen_prediction_reproduced'] else "a fresh replay was not supplied for comparison."),
        "Reproducibility establishes consistent execution, not predictive accuracy.",
        f"Jobs sharing the same known configuration: {report['jobs_sharing_known_configuration']}. "
        "Runtime warmup may still differ; repeated recipes must not be treated as independent held-out cases.",
        "", "## Findings", "", "The three PASS cases are 17-layer, width-256 debug models, not full K3 passes. "
        "Four OOM cases report rounded failure-point memory; two report NCCL OOM without a numeric memory snapshot. "
        "None is a completed full-model peak target.", "",
        "| OOM job | Process GPU usage outside PyTorch reserved backing (GiB) | Above previous 16 GiB allowance (GiB) |",
        "|---|---:|---:|"]
    for e in report['external_memory_observations']:
        lines.append(f"| {e['job']} | {e['process_outside_allocator_gib']:.2f} | {e['minus_previous_allowance_gib']:.2f} |")
    lines += ["", "These are approximate same-process residuals: process usage − (allocated + reserved-unused). "
        "They are not exclusively NCCL. Sibling processes affect device free memory separately. "
        "The failed allocation is never added to the observed live total.", "",
        "| Debug PASS job | Max logged active (GiB) | Max logged reserved (GiB) | Last logged active / reserved (GiB) |",
        "|---|---:|---:|---:|"]
    for e in report['experiments']:
        if e['outcome']!='PASS': continue
        a=e['logged_step_windows'][ACTIVE]; r=e['logged_step_windows'][RESERVED]
        if a is None or r is None: continue
        lines.append(f"| {e['job']} | {a['max_logged_window_gib']:.6f} | {r['max_logged_window_gib']:.6f} | {a['last_logged_window_max_gib']:.6f} / {r['last_logged_window_max_gib']:.6f} |")
    w=report['communication_warmup']
    lines += ["", f"Recorded optimizer communication warmup: {w['samples']} snapshots; distinct outside-allocator "
        f"deltas {[round(x/2**20,3) for x in w['distinct_external_delta_bytes']]} MiB. "
        "These debug observations do not establish full-model communication memory.", "",
        "`memory/max_active(GiB)` reads allocator active bytes in the pinned training source; it is not necessarily "
        "the same as live tensor storage. The validator preserves per-rank logger windows and does not merge "
        "graph startup/capture/replay measurements into an assumed steady-state peak.", "", "## Required before claiming success", ""]
    lines += [f"- {s}" for s in report['next_requirements']]
    lines += ["", "The proposed 5% allocated / 10% reserved targets remain untested. Missing comparisons are null, "
        "not zero error or correct fit predictions. The old 186.084 GiB result is for a different scenario and "
        "cannot be validated by numerical comparison to these OOM snapshots.", "",
        f"Input SHA-256: `{report['source_dataset_sha256']}`.", "",
        "## Reproduce the audit", "", "```sh",
        "PYTHONPATH=src python -m memtracker_nccl.datapoint_validation /path/to/datapoints.json \\",
        "  --prediction experiments/k3_training_lifetimes_expandable.json \\",
        "  --output experiments/gb200_historical_validation.json \\",
        "  --markdown docs/gb200-historical-validation.md", "```", "",
        "Add `--replayed-prediction /path/to/fresh-replay.json` to verify exact reproduction. "
        "The committed `experiments/gb200_historical_observations.json` is a compact normalized input "
        "with original file hash, complete-config fingerprints, per-rank metrics and extracted warmup snapshots; "
        "it can replace the raw input for the audit. The original large configurations remain in the supplied file. "
        "No GPU jobs or remote transfers are needed.", ""]
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('datapoints',type=Path)
    p.add_argument('--prediction',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--markdown',type=Path)
    p.add_argument('--normalized-output',type=Path)
    p.add_argument('--replayed-prediction',type=Path,help='Optional fresh replay; reject if it differs from frozen report')
    args=p.parse_args()
    raw=read_report(args.datapoints)
    normalized=(raw if raw.get('kind')=='normalized_gb200_validation_observations' else
                normalize_dataset(raw,source_sha256=hashlib.sha256(args.datapoints.read_bytes()).hexdigest()))
    result=audit(normalized,read_report(args.prediction),
                 replayed_prediction=read_report(args.replayed_prediction) if args.replayed_prediction else None)
    write_report(args.output,result)
    if args.normalized_output: write_report(args.normalized_output,normalized)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True,exist_ok=True)
        args.markdown.write_text(markdown(result))
    print(json.dumps(dict(status=result['status'],counts=result['counts'])))


if __name__=='__main__': main()
