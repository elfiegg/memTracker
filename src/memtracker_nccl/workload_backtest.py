"""Configuration-driven reference replays against historical GB200 observations.

Predictions use saved shapes, schedules and optimizer settings, never observed
memory as an input. Numerical discrepancies are diagnostics of a partial model,
not accuracy scores for a complete, source-equivalent training simulator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from .datapoint_validation import ACTIVE, RESERVED, failure_accounting, normalize_dataset
from .k3_fullac_probe import probe as residual_probe
from .k3_recipe_probe import recipe_inventory, run_recipe_probe
from .k3_training_lifetimes import run as replay
from .recipe_schedule import build_schedule
from .report_io import read_report, write_report
from .run_metadata import apply_dataset_metadata, compare_metadata

GiB = 2**30


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_recipes(raw, source_sha256):
    """Retain all modeled inputs; drop initializer/sharding repr and unrelated I/O."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in ("param_init", "sharding_config")}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    keys = ("model", "training", "parallelism", "optimizer", "activation_checkpoint", "compile", "loss")
    rows=[]
    for e in raw["experiments"]:
        retained=strip({k:e["effective_config"][k] for k in keys})
        rows.append(dict(job=e["job"], effective_config=retained,
                         effective_config_sha256=fingerprint(e["effective_config"]),
                         replay_config_sha256=fingerprint(retained)))
    return dict(schema_version=1, source_sha256=source_sha256,
        omitted=["param_init", "sharding_config", "unmodeled dataloader/logging configuration"],
        experiments=rows)


def validate_recipe_record(record, observation):
    if record["effective_config_sha256"] != observation["effective_config_sha256"]:
        raise ValueError(f"Original config fingerprint mismatch for {record['job']}")
    if fingerprint(record["effective_config"]) != record.get("replay_config_sha256"):
        raise ValueError(f"Retained replay config fingerprint mismatch for {record['job']}")
    c=record["effective_config"]
    if c["parallelism"]["pipeline_parallel_degree"] * c["parallelism"]["data_parallel_shard_degree"] != observation["world_size"]:
        raise ValueError("Saved topology does not match observed world size")
    if observation["configuration"]["checkpointing"] != "FullAC":
        raise ValueError("Reference replay currently requires FullAC")


def attach_run_metadata(prediction, metadata):
    """Reuse tensor math without reusing another run's identity comparison."""
    result=copy.deepcopy(prediction)
    result["run_metadata"]=copy.deepcopy(metadata)
    result["runtime_compatibility"]=compare_metadata(metadata,result["modeled_metadata"])
    return result


def _observed_window(metrics, rank, field, index):
    values = metrics.get(str(rank), {}).get(field, [])
    return values[index] if len(values) > index else None


def compare_run(observation, first, initialized):
    """Keep successful window diagnostics separate from censored OOM constraints."""
    first_rows = {r["global_rank"]: r for result in first for r in result["ranks"]}
    later_rows = {r["global_rank"]: r for result in initialized for r in result["ranks"]}
    diagnostics = []
    for rank, row in sorted(first_rows.items()):
        if observation["outcome"] != "PASS":
            continue
        measured = _observed_window(observation["per_rank_step_memory_metrics"], rank, ACTIVE, 0)
        reserved = _observed_window(observation["per_rank_step_memory_metrics"], rank, RESERVED, 0)
        modeled = row["modeled_active_peak"]["bytes"] / GiB
        modeled_reserved = row["modeled_reserved_peak"]["bytes"] / GiB
        second = _observed_window(observation["per_rank_step_memory_metrics"], rank, ACTIVE, 1)
        diagnostics.append(dict(global_rank=rank, pp_rank=row["pp_rank"], dp_owner=row["dp_owner"],
            modeled_first_step_active_gib=modeled, observed_first_logged_active_gib=measured,
            active_gap_gib=None if measured is None else modeled-measured,
            modeled_first_step_reserved_gib=modeled_reserved, observed_first_logged_reserved_gib=reserved,
            reserved_gap_gib=None if reserved is None else modeled_reserved-reserved,
            modeled_initialized_active_gib=later_rows[rank]["modeled_active_peak"]["bytes"]/GiB,
            observed_second_logged_active_gib=second,
            modeled_first_step_peak_event=row["modeled_active_peak"]["event"],
            accuracy_error_percent=None))
    failure = failure_accounting(observation["failure_point_memory_gib"])
    rank = observation["representative_rank"]
    failed_rank = first_rows.get(rank) if rank is not None else None
    constraint = None
    if observation["outcome"] == "OOM":
        constraint = dict(representative_rank=rank, recorded_phase=observation["phase"],
            failure_point=observation["failure_point_memory_gib"], failure_accounting=failure,
            modeled_rank_phase_peaks_gib=None if failed_rank is None else
                {k: v["bytes"]/GiB for k, v in failed_rank["phase_peaks"].items()},
            comparison="Censored failure; a phase peak is not the allocation-failure instant or a completed-step target",
            fit_classification_correct=None)
    return dict(job=observation["job"], outcome=observation["outcome"], configuration=observation["configuration"],
        run_metadata=observation.get("run_metadata"), modeled_global_ranks=sorted(first_rows),
        successful_window_diagnostics=diagnostics if observation["outcome"] == "PASS" else [],
        oom_constraint=constraint, external_memory_prediction_gib=None,
        observed_outside_allocator_gib=failure["process_outside_allocator_gib"],
        full_peak_accuracy_verified=False)


def run_backtest(recipes, observations, source, pytorch_schedules, evidence_dir):
    if recipes["source_sha256"] != observations["source_sha256"]:
        raise ValueError("Recipes and measurements come from different dataset files")
    obs = {e["job"]: e for e in observations["experiments"]}
    jobs = [e["job"] for e in recipes["experiments"]]
    if len(jobs) != len(set(jobs)) or set(jobs) != set(obs):
        raise ValueError("Recipe and observation job sets must match without duplicates")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    optimizers, residuals, prediction_cache = {}, {}, {}
    results = []
    artifacts = {}
    def save(name, value):
        path = evidence_dir/name
        write_report(path, value)
        artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for e in recipes["experiments"]:
        job, c = e["job"], e["effective_config"]
        observed = obs[job]
        validate_recipe_record(e, observed)
        print(f"job {job}: saved-config reference replay", flush=True)
        schedule = build_schedule(c, source, pytorch_schedules)
        save(f"schedule-{job}.json.gz", schedule)
        inv = recipe_inventory(c)
        # Owners are chosen from requested topology, never measured byte values.
        # Successful small cases cover every rank; OOM cases target the recorded failed rank.
        dp = c["parallelism"]["data_parallel_shard_degree"]
        if observed["outcome"] == "OOM" and observed["representative_rank"] is None:
            raise ValueError("OOM phase comparison requires a recorded representative rank")
        failed_rank = observed["representative_rank"]
        owners = tuple(range(dp)) if observed["outcome"] == "PASS" else (failed_rank % dp,)
        pp_ranks = None if observed["outcome"] == "PASS" else [failed_rank // dp]
        stages = [s for s in schedule["stages"] if pp_ranks is None or s["rank"] in pp_ranks]
        opt_config = copy.deepcopy(c["optimizer"])
        for opt in opt_config["optimizers"]:
            opt.pop("warmup_communication", None)  # External communicator backing is not predicted.
        opt_key = fingerprint([inv, opt_config, stages, dp, owners])[:16]
        if opt_key not in optimizers:
            print(f"  source optimizer planning: owners {owners}", flush=True)
            optimizers[opt_key] = run_recipe_probe(c, stages, source, owners=owners)
            save(f"optimizer-{opt_key}.json.gz", optimizers[opt_key])
        opt = optimizers[opt_key]
        tokens = c["training"]["num_tokens_per_microbatch_per_dp_rank"]
        sequence = c["training"]["max_context_length"]
        if tokens % sequence:
            raise ValueError("Nonintegral microbatch size")
        dim = c["model"]["dim"]
        count = (len(c["model"]["layers"])-1)//schedule["residual_block_size"]+1
        residual_key = (tokens, dim, count)
        if residual_key not in residuals:
            print(f"  FullAC helper probe: tokens={tokens}, dim={dim}", flush=True)
            residuals[residual_key] = residual_probe(source, tokens=tokens, dim=dim, residuals=range(1, count+1))
            save(f"residual-{tokens}-{dim}-{count}.json.gz", residuals[residual_key])
        all_first, all_initialized = [], []
        for initialized, destination in ((False, all_first), (True, all_initialized)):
            if initialized and observed["outcome"] != "PASS":
                continue
            key = fingerprint([opt_key, residual_key, schedule, c["training"], initialized])
            if key not in prediction_cache:
                prediction_cache[key] = [replay(schedule, opt["muon"], residuals[residual_key], opt["adam"],
                    sequence=sequence, microbatch_size=tokens//sequence, communication_gib=0,
                    recipe=c, aggregate=opt["aggregate"], initialized_optimizer=initialized, dp_owner=owner, pp_ranks=pp_ranks,
                    run_metadata=observed.get("run_metadata")) for owner in owners]
            # Different jobs may have different metadata, even when tensor math is shared.
            for prediction in prediction_cache[key]:
                destination.append(attach_run_metadata(prediction, observed["run_metadata"]))
            save(f"replay-{job}-{'initialized' if initialized else 'first'}.json.gz", destination)
        result = compare_run(observed, all_first, all_initialized)
        result.update(effective_config_sha256=e["effective_config_sha256"],
                      replay_config_sha256=e["replay_config_sha256"],
                      optimizer_evidence=f"optimizer-{opt_key}.json.gz", reference_limitations=opt["limitations"])
        results.append(result)
    return dict(schema_version=1, status="partial_model_backtested_not_validated",
        source_dataset_sha256=recipes["source_sha256"], estimator_tuned=False,
        experiments_replayed=len(results), successful_rank_window_comparisons=sum(len(r["successful_window_diagnostics"]) for r in results),
        full_peak_accuracy_verified=False, artifacts=artifacts,
        measurement_alignment=[
            "Reference training engine uses two eager optimizer steps before CUDA graph capture; no graph pools/capture/replay are modeled.",
            "First logged windows can include construction, warmup, or earlier peak history absent from this replay; differences are diagnostics, not accuracy scores.",
            "Initialized replay starts from a fresh allocator and initialized optimizer state, not the exact second-iteration allocator history.",
            "Active prediction is rounded live allocator storage with immediate free completion; GPU pending frees are not modeled.",
            "OOM snapshots remain censored; failed requested allocations are never added to observed memory.",
            "OOM runs replay the recorded failure rank only; successful debug runs cover every owner/rank. No global worst-rank claim is made for OOM runs.",
            "External CUDA/NCCL/HybridEP memory remains unpredicted, not zero; communication_gib=0 in replay artifacts isolates tensor/allocator accounting.",
            "Reference source b4d5b404 is not verified equivalent to historical base53a45 plus local changes."],
        experiments=results)


def markdown(report):
    lines = ["# GB200 workload backtest", "", "**The partial estimator does not yet validate full training peaks.**",
        f"Replayed {report['experiments_replayed']} saved recipes and made {report['successful_rank_window_comparisons']} per-rank successful-run comparisons. No memory observations were used to tune predictions.", "",
        "## Successful debug runs", "", "Values below are MiB. Model columns are maxima across modeled ranks; observed columns are maxima across corresponding first logged windows.", "",
        "| Job | Model active | Logged active | Model reserved | Logged reserved |",
        "|---|---:|---:|---:|---:|"]
    for e in report["experiments"]:
        rows=e["successful_window_diagnostics"]
        if not rows: continue
        keys=("modeled_first_step_active_gib","observed_first_logged_active_gib","modeled_first_step_reserved_gib","observed_first_logged_reserved_gib")
        values=[max(r[k] for r in rows if r[k] is not None)*1024 for k in keys]
        lines.append(f"| {e['job']} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines += ["", "These are discrepancies between the partial reference and logs, not a percentage accuracy claim. Exact source equivalence and startup/metric windows remain unresolved. Graph capture/replay peaks are excluded.", "",
        "## Full-model OOM constraints", "", "Values are GiB on the recorded failure rank. Phase peaks are computed reference peaks; snapshots describe the failure instant.", "",
        "| Job | Rank | Model training peak | Model optimizer peak | Allocated at failure | Outside allocator at failure |",
        "|---|---:|---:|---:|---:|---:|"]
    for e in report["experiments"]:
        failure=e["oom_constraint"]
        if failure is None: continue
        peaks=failure["modeled_rank_phase_peaks_gib"] or {}
        vals=[peaks.get("training"),peaks.get("optimizer"),failure["failure_point"]["torch_allocated_gib"],e["observed_outside_allocator_gib"]]
        lines.append(f"| {e['job']} | {failure['representative_rank']} | " + " | ".join("unknown" if v is None else f"{v:.2f}" for v in vals) + " |")
    lines += ["", "A below-capacity subtotal cannot classify these runs as fitting. Private communication memory, complete kernel activations/workspaces and several source lifetimes remain uncovered.", "",
        "## Interpretation", ""]
    lines += ["- "+s for s in report["measurement_alignment"]]
    lines += ["", "Next validation work should explain the successful debug-run gaps first: align construction/warmup windows, execute whole-block activation paths, model stream completion and graph pool lifetimes, and establish source equivalence. Keep repeated recipes together when later separating calibration and held-out data.", ""]
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("datapoints", type=Path)
    p.add_argument("--observations", type=Path, help="Normalized observations when input is the portable recipe fixture")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--pytorch-schedules", type=Path, required=True)
    p.add_argument("--run-metadata-manifest", type=Path)
    p.add_argument("--evidence-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--markdown", type=Path)
    args=p.parse_args()
    raw=read_report(args.datapoints)
    if args.observations:
        recipes, observations=raw, read_report(args.observations)
    else:
        digest=hashlib.sha256(args.datapoints.read_bytes()).hexdigest()
        recipes, observations=prepare_recipes(raw,digest), normalize_dataset(raw,source_sha256=digest)
    observations=apply_dataset_metadata(observations,
        manifest=read_report(args.run_metadata_manifest) if args.run_metadata_manifest else None)
    result=run_backtest(recipes,observations,args.source,args.pytorch_schedules,args.evidence_dir)
    write_report(args.output,result)
    if args.markdown: args.markdown.write_text(markdown(result))
    print(result["status"],flush=True)


if __name__ == "__main__": main()
