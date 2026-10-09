#!/usr/bin/env python3
"""One-device, eager CUDA calibration of pinned K3 helper storage lifetimes.

No process groups, training model, or distributed launch are created. Run in a
fresh process for each PYTORCH_ALLOC_CONF setting. FakeTensor and CUDA execute
the same source helpers; this does not certify whole-model fit.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

from memtracker_nccl.k3_fullac_probe import COMMIT, PINS, load_functions, trace_helper, summarize_trace
from memtracker_nccl.k3_muon_probe import kernel_probe

MUON_FILE = "torchtitan/distributed/flex_shard/dist_muon.py"
MUON_HASH = "26ce80ac34a1ca85a22c83a44e542ab3d2a683f969154d78c7cc34e8b6a03b8b"


def load_muon(source):
    raw = (source / MUON_FILE).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == MUON_HASH
    names = {"_MatrixBatchView", "_adjust_muon_learning_rate", "_prepare_muon_input",
             "_compute_muon_direction", "_apply_muon_update", "_zeropower_via_newtonschulz"}
    nodes = [n for n in ast.parse(raw).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(nodes) == len(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ns = dict(torch=torch, math=math, dataclass=dataclass)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(source / MUON_FILE), "exec"), ns)
    return SimpleNamespace(**{n: ns[n] for n in names})


def sample(phase, elapsed=None):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return dict(phase=phase, allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                driver_used_bytes=total-free, elapsed_seconds=elapsed)


def clean():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def attention(helper, policy, full_ac):
    clean()
    baseline = sample("baseline")
    projection = torch.nn.Linear(7168, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    norm = torch.nn.RMSNorm(7168, eps=1e-5, dtype=torch.bfloat16, device="cuda")
    partial = torch.randn(8136, 7168, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    stack = torch.randn(8136, 8, 7168, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    phases = [sample("inputs")]
    calls = []
    def function(x, residual):
        calls.append(len(calls))
        return helper(x, residual, projection, norm)
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = (checkpoint(function, partial, stack, use_reentrant=False,
                      context_fn=lambda: create_selective_checkpoint_contexts(policy),
                      preserve_rng_state=True, early_stop=True)
           if full_ac else function(partial, stack))
    torch.cuda.synchronize()
    phases.append(sample("forward", time.perf_counter()-start))
    start = time.perf_counter()
    out.sum().backward()
    torch.cuda.synchronize()
    phases.append(sample("backward", time.perf_counter()-start))
    return dict(component="attention_residual", full_ac=full_ac, helper_calls=len(calls),
                shape=[8136, 8, 7168], baseline=baseline, phases=phases)


def muon_cuda(muon, views):
    clean()
    baseline = sample("baseline")
    shape = (28, 2, 3072, 3584)
    grad = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    momentum = torch.zeros_like(grad)
    parameter = torch.ones_like(grad)
    prepared = torch.empty_like(grad)
    phases = [sample("inputs")]
    def run(phase, fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        phases.append(sample(phase, time.perf_counter()-start))
    run("prepare", lambda: muon._prepare_muon_input(grad, momentum, momentum=.95, nesterov=True, out=prepared))
    run("newton_schulz", lambda: muon._compute_muon_direction(prepared, matrix_views=views,
        lr_reference_shape=shape, adjust_lr_fn="match_rms_adamw",
        ns_coefficients=(3.4445,-4.7750,2.0315), ns_steps=5, eps=1e-7))
    run("final_update", lambda: muon._apply_muon_update(parameter, prepared, lr=8e-4,
        weight_decay=0, adjust_lr_fn="match_rms_adamw", compute_matrix_shape=shape))
    return dict(component="muon_expert_w13", shape=list(shape), ns_steps=5,
                weight_decay=0, baseline=baseline, phases=phases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1, "One process only"
    torch.cuda.set_device(0)
    helper, policy = load_functions(args.source)
    muon = load_muon(args.source)
    shape = torch.Size((28, 2, 3072, 3584))
    views = (muon._MatrixBatchView(shape, tuple(math.prod(shape[i+1:]) for i in range(len(shape))), 0),)
    props = torch.cuda.get_device_properties(0)
    result = dict(schema_version=1, source_commit=COMMIT,
        source_sha256={**PINS, MUON_FILE: MUON_HASH}, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, nccl=list(torch.cuda.nccl.version()),
            cudnn=torch.backends.cudnn.version(), device=props.name, device_total_bytes=props.total_memory,
            compute_capability=list(torch.cuda.get_device_capability()), host=socket.gethostname(),
            visible_device_count=torch.cuda.device_count(), device_used=0,
            allocator_config=os.environ.get("PYTORCH_ALLOC_CONF", ""), job_id=os.environ.get("SLURM_JOB_ID"),
            driver=subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()),
        scope="One CUDA device, isolated eager source helpers, no distributed process group or NCCL collectives",
        limitations=["Whole decoder checkpoint overlap, MLA/KDA/HybridEP kernels and full-model training remain uncovered.",
            "Driver used bytes are device-wide endpoint samples, not a private-library peak attribution.",
            "Fixed one-stream allocator replay cannot describe multi-stream reuse delays."],
        probes=[])
    def save():
        args.output.write_text(json.dumps(result, indent=2)+"\n")
    for ac in (False, True):
        fake = trace_helper(helper, policy, tokens=8136, dim=7168, residuals=8, full_ac=ac)
        actual = attention(helper, policy, ac)
        actual["fake"] = dict(after_forward_live_bytes=fake["after_forward_live_bytes"], allocator=fake["allocator"])
        result["probes"].append(actual)
        save()
        print(json.dumps(actual), flush=True)
    fake = kernel_probe(muon, shape, views, shape)
    # kernel_probe excludes input events; prepend the four equal persistent inputs.
    events = [dict(op="alloc", storage_id="input"+str(i), bytes=fake["input_bytes"]//4, phase="inputs") for i in range(4)]
    events += [dict(e, storage_id=str(e["storage_id"])) for e in fake["storage_events"]]
    actual = muon_cuda(muon, views)
    actual["fake"] = dict(input_bytes=fake["input_bytes"], transient_peak_bytes=fake["transient_peak_bytes"],
        allocator={mode: summarize_trace(events, expandable=mode=="expandable") for mode in ("fixed", "expandable")})
    result["probes"].append(actual)
    for row in result["probes"]:
        actual_peak = max(p["peak_allocated_bytes"] for p in row["phases"])
        actual_reserved = max(p["peak_reserved_bytes"] for p in row["phases"])
        mode = "expandable" if "expandable_segments:True" in result["runtime"]["allocator_config"] else "fixed"
        modeled = row["fake"]["allocator"][mode]
        row["comparison"] = dict(actual_peak_allocated_bytes=actual_peak,
            actual_peak_reserved_bytes=actual_reserved, modeled_peak_live_bytes=modeled["peak_live_bytes"],
            modeled_peak_reserved_bytes=modeled["peak_reserved_bytes"],
            allocated_minus_modeled_bytes=actual_peak-row["baseline"]["allocated_bytes"]-modeled["peak_live_bytes"],
            reserved_minus_modeled_bytes=actual_reserved-modeled["peak_reserved_bytes"])
    save()
    print(json.dumps(dict(output=str(args.output), probes=len(result["probes"]))), flush=True)


if __name__ == "__main__":
    main()
