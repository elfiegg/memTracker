"""Execute pinned scheduling/layout helpers with a saved workload's dimensions.

Only explicitly supplied, hash-checked local Python sources are executed. JSON
configuration repr strings are never executed. This is a reference schedule,
not evidence that a historical run used identical source contents.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from enum import Enum
import hashlib
import logging
from pathlib import Path
import runpy
from types import SimpleNamespace as NS
from typing import NamedTuple

SCHEDULE_HASH = "61c4e103e3bcb17c11c4dc46db138d2c3400b76e3a8b2d4797cf106e51e931da"
PIPELINE_HASH = "7fbe51d5e194c44cba0b4939d459bfc0ed432c5a20400d894c7d992d22210755"
LAYOUT_HASH = "5b156f384a5fee38802dfa0f11262d4f8a7e8f58334f6f340971708378b08229"


def _execute(node, path, namespace):
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def _single_actions(method, ns, pp, microbatches):
    """Trace the actual Schedule1F1B executor, not its visualization helper.

    The pinned visualization helper starts the last rank at F1. The executor
    correctly starts at F0, so source execution is essential here.
    """
    result = {}
    ns.update(_wait_batch_p2p=lambda *a, **k: None, _batch_p2p=lambda *a, **k: [], FSDPModule=type("FSDPModule", (), {}))
    for rank in range(pp):
        actions = []
        def append(op, mb=None):
            actions.append(f"{rank}{op}{mb if mb is not None else ''}")
            return []
        stage = NS(stage_index=rank, submod=ns["FSDPModule"](),
            get_fwd_recv_ops=lambda mb: append("RECV_F", mb) if rank else [],
            get_bwd_recv_ops=lambda mb: append("RECV_B", mb) if rank != pp-1 else [],
            get_fwd_send_ops=lambda mb: append("SEND_F", mb) if rank != pp-1 else [],
            get_bwd_send_ops=lambda mb: append("SEND_B", mb) if rank else [],
            forward_one_chunk=lambda mb, *a, **k: append("F", mb),
            backward_one_chunk=lambda mb, *a, **k: append("B", mb),
            perform_reduce_grad=lambda *a: append("REDUCE_GRAD"))
        context = NS(_n_microbatches=microbatches, _num_stages=pp, _stage=stage,
            _finalize_gradients=True, scale_grads=True,
            _check_inputs=lambda *a: ([()] * microbatches, [{}] * microbatches),
            _initialize_stage=lambda *a: None, _maybe_compute_loss=lambda *a: None,
            _maybe_get_loss=lambda *a: None, _update_losses=lambda *a: None)
        method(context, return_outputs=False)
        result[str(rank)] = actions
    return result


def build_schedule(config, source: Path, pytorch_schedules: Path):
    pipeline = source / "torchtitan/distributed/pipeline_parallel.py"
    layout_path = source / "torchtitan/models/kimi_k3/pipeline_parallel/layout.py"
    pins = {pytorch_schedules: SCHEDULE_HASH, pipeline: PIPELINE_HASH, layout_path: LAYOUT_HASH}
    for path, expected in pins.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unreviewed scheduling source: {path}")
    p = config["parallelism"]
    for key in ("pipeline_parallel_module_fqns_per_model_part", "pipeline_parallel_layers_per_stage", "pipeline_parallel_schedule_csv"):
        if p.get(key):
            raise ValueError(f"Explicit layout override is not implemented: {key}")
    pp, microbatches = p["pipeline_parallel_degree"], p["num_pp_microbatches"]
    name = p["pipeline_parallel_schedule"]
    if name not in ("1F1B", "Interleaved1F1B") or microbatches < pp:
        raise ValueError("Require 1F1B/Interleaved1F1B and at least PP microbatches")
    local_stages = 2 if name == "Interleaved1F1B" else 1
    stages = pp * local_stages
    block_sizes = {layer["attn_res_block_size"] for layer in config["model"]["layers"]}
    if len(block_sizes) != 1:
        raise ValueError("Mixed residual block sizes are unsupported")
    block_size = next(iter(block_sizes))
    ns = dict(Enum=Enum, NamedTuple=NamedTuple, defaultdict=defaultdict, Counter=Counter,
              logger=logging.getLogger(__name__))
    tree = ast.parse(pytorch_schedules.read_text())
    names = {"_requires_reduce_grad", "_ComputationType", "_Action", "_get_warmup_ops",
             "_get_1f1b_rank_ops", "_add_unshard_reshard", "_add_reduce_grad", "_add_send_recv", "_resolve_unshard_lookahead"}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            _execute(node, pytorch_schedules, ns)
            if node.name == "_ComputationType":
                ns.update({v.name: v for v in ns[node.name]})
                ns.update(F=ns["FORWARD"], B=ns["FULL_BACKWARD"], I=ns["BACKWARD_INPUT"], W=ns["BACKWARD_WEIGHT"])
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Schedule" + name)
    method_name = "_calculate_single_rank_operations" if local_stages == 2 else "_step_microbatches"
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    _execute(fn, pytorch_schedules, ns)
    active = p.get("pp_max_unsharded_active_stages") or local_stages
    lookahead = p.get("pp_num_unshard_lookahead_factor", "auto")
    lookahead = tuple(lookahead) if isinstance(lookahead, list) else lookahead
    distances = ns["_resolve_unshard_lookahead"](lookahead, pp, active)
    if local_stages == 2:
        rounds = max(1, microbatches // pp)
        if microbatches % rounds:
            raise ValueError("Microbatch count is not divisible by interleaved rounds")
        context = NS(n_local_stages=2, pp_group_size=pp, microbatches_per_round=microbatches//rounds, _n_microbatches=microbatches)
        compute = {r: ns[method_name](context, r) for r in range(pp)}
        actions = ns["_add_send_recv"]({r: ns["_add_reduce_grad"](
            ns["_add_unshard_reshard"](a, max_active_stages=active, unshard_lookahead=distances[r]), microbatches)
            for r, a in compute.items()}, lambda s: s % pp, stages)
        actions = {str(r): [str(a) for a in aa] for r, aa in actions.items()}
    else:
        actions = _single_actions(ns[method_name], ns, pp, microbatches)
    fn = next(n for n in ast.parse(pipeline.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_generate_llm_fqn_per_model_part")
    _execute(fn, pipeline, ns)
    split = ns[fn.name](stages, len(config["model"]["layers"]),
        p["pipeline_parallel_first_stage_less_layers"], p["pipeline_parallel_last_stage_less_layers"])
    layout_ns = runpy.run_path(str(layout_path))
    layout = layout_ns["infer_block_layout_tables"](stage_to_rank={s:s%pp for s in range(stages)},
        n_layers=len(config["model"]["layers"]), layers_per_block=block_size,
        layer_to_stage=layout_ns["layer_to_stage_from_split"](split), cache=True)
    stage_rows = []
    for st, fqns in enumerate(split):
        layers = [int(f.split(".")[1]) for f in fqns if f.startswith("layers.")]
        if not layers:
            raise ValueError("Empty decoder stage is not supported")
        stage_rows.append(dict(stage=st, rank=st%pp, first_layer=min(layers), last_layer=max(layers),
            incoming_delta_blocks=layout.delta_to_send(st-1) if st else [], outgoing_delta_blocks=layout.delta_to_send(st),
            cached_blocks_at_entry=sorted(layout.cache_at_entry(st)), committed_blocks=layout.commits_at(st)))
    return dict(schema_version=1, torch_runtime="2.15.0.dev20260928+cu130",
        training_source="b4d5b404bd30dec67f86873203fbd4ba5f457531",
        source_sha256={path.name: digest for path, digest in pins.items()}, schedule=name,
        pp=pp, virtual_stages=stages, stages_per_rank=local_stages, microbatches=microbatches,
        max_active_stages=active, unshard_lookahead=list(distances), residual_block_size=block_size,
        stages=stage_rows, actions=actions)
