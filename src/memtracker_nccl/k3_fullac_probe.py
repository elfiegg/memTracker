"""Execute K3 attention-residual storage lifetimes with the pinned FullAC policy.

This isolates an actual source function, not a substitute attention kernel. CUDA
kernel-internal allocations remain unobservable to FakeTensor.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.mem_tracker import _UpdateType
from torch.utils.checkpoint import checkpoint, CheckpointPolicy, create_selective_checkpoint_contexts

from .allocator_model import AllocatorConfig, CachingAllocatorModel
from .nccl_model import _integer
from .tracker import ExtendedMemTracker

COMMIT = "b4d5b404bd30dec67f86873203fbd4ba5f457531"
PINS = {
    "torchtitan/models/kimi_k3/model.py": "ee48193981eebb9fe65930211890277dcfcf8a646af078e106fb7bace6b6d4c8",
    "torchtitan/distributed/activation_checkpoint.py": "829a96827cb8daa41588fe7d4ac8189cb375ea0040723785af1a977da780cc46",
}


def load_functions(source: Path):
    namespace = {"torch": torch, "CheckpointPolicy": CheckpointPolicy}
    for relative, expected in PINS.items():
        path = source / relative
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Source differs from pinned {COMMIT}: {relative}")
        name = "_apply_attention_residual" if relative.endswith("model.py") else "_full_ac_policy"
        node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == name)
        node.returns = None
        for arg in [*node.args.args, *node.args.kwonlyargs, node.args.vararg, node.args.kwarg]:
            if arg is not None:
                arg.annotation = None
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["_apply_attention_residual"], namespace["_full_ac_policy"]


class StorageTrace(ExtendedMemTracker):
    """Unique storage events, including deletes, without retaining tensor objects."""

    def __init__(self):
        self.storage_events = []
        self.phase = "inputs"
        self._ids = {}
        self._next_id = 0
        super().__init__()

    def _update_snap(self, u_type, winfo, *args, **kwargs):
        if u_type in (_UpdateType.SIZE, _UpdateType.DEL) and winfo in self._ids:
            self.storage_events.append(dict(op="free", storage_id=self._ids.pop(winfo), phase=self.phase))
        if u_type in (_UpdateType.ADD, _UpdateType.SIZE):
            self._next_id += 1
            self._ids[winfo] = str(self._next_id)
            self.storage_events.append(dict(op="alloc", storage_id=str(self._next_id),
                                           bytes=winfo.size*winfo.element_size, phase=self.phase))
        super()._update_snap(u_type, winfo, *args, **kwargs)


def summarize_trace(events, *, expandable=False):
    allocator = CachingAllocatorModel(AllocatorConfig(expandable_segments=expandable))
    live = {}; phases = {}; peak = 0; reserved_peak = 0
    for event in events:
        sid = event["storage_id"]
        if event["op"] == "alloc":
            live[sid] = event["bytes"]
            allocator.allocate(sid, event["bytes"])
        else:
            del live[sid]
            allocator.free(sid)
        current = sum(live.values())
        reserved = sum(s["reserved_bytes"] for s in allocator.snapshot().values())
        row = phases.setdefault(event["phase"], dict(peak_live_bytes=0, peak_reserved_bytes=0))
        row["peak_live_bytes"] = max(row["peak_live_bytes"], current)
        row["peak_reserved_bytes"] = max(row["peak_reserved_bytes"], reserved)
        row["end_live_bytes"] = current
        peak = max(peak, current); reserved_peak = max(reserved_peak, reserved)
    return dict(peak_live_bytes=peak, peak_reserved_bytes=reserved_peak, phases=phases,
                allocator_mode="expandable" if expandable else "fixed")


def trace_helper(helper, policy, *, tokens, dim, residuals, full_ac):
    tracker = StorageTrace()
    calls = []
    with FakeTensorMode(allow_fallback_kernels=False):
        projection = torch.nn.Linear(dim, 1, bias=False, dtype=torch.bfloat16)
        norm = torch.nn.RMSNorm(dim, eps=1e-5, dtype=torch.bfloat16)
        partial = torch.empty(tokens, dim, dtype=torch.bfloat16, requires_grad=True)
        stack = torch.empty(tokens, residuals, dim, dtype=torch.bfloat16, requires_grad=True)
        tracker.track_external(projection, norm, partial, stack)

        def function(x, residual):
            calls.append(tracker.phase)
            return helper(x, residual, projection, norm)

        @contextmanager
        def mark_recompute(context):
            previous = tracker.phase
            tracker.phase = "recompute"
            try:
                with context:
                    yield
            finally:
                tracker.phase = previous

        def context_fn():
            forward, recompute = create_selective_checkpoint_contexts(policy)
            return forward, mark_recompute(recompute)

        with tracker:
            tracker.phase = "forward"
            out = (checkpoint(function, partial, stack, use_reentrant=False, context_fn=context_fn,
                              preserve_rng_state=True, early_stop=True)
                   if full_ac else function(partial, stack))
            gc.collect()
            retained = tracker.report()["current"]["cpu"]["tensor_bytes"]
            tracker.phase = "backward"
            out.sum().backward()
            tracker.phase = "cleanup"
            projection.zero_grad(set_to_none=True); norm.zero_grad(set_to_none=True)
            partial.grad = None; stack.grad = None
            del out
            gc.collect()
        result = dict(full_ac=full_ac, function_call_phases=calls,
                      after_forward_live_bytes=retained, storage_events=tracker.storage_events.copy())
        # Captured before deleting tracked inputs; input storage is replay context.
        result["allocator"] = {mode: summarize_trace(result["storage_events"], expandable=mode == "expandable")
                               for mode in ("fixed", "expandable")}
        return result


def probe(source: Path, *, tokens=8136, dim=7168, residuals=(1, 2, 3, 4, 5, 6, 7, 8)):
    for name, value in (("tokens", tokens), ("dim", dim)):
        _integer(name, value, 1)
    for count in residuals:
        _integer("residuals", count, 0)
    helper, policy = load_functions(source)
    rows = []
    for count in residuals:
        for ac in (False, True):
            rows.append(dict(residual_entries=count, **trace_helper(helper, policy, tokens=tokens,
                                                                  dim=dim, residuals=count, full_ac=ac)))
    return dict(schema_version=1, source_commit=COMMIT, source_sha256=PINS,
                torch_runtime=str(torch.__version__), tokens=tokens, dim=dim, probes=rows,
                coverage="Exact attention-residual helper forward/autograd backward; pinned FullAC policy via non-reentrant checkpoint",
                limitations=["Helper is isolated; the training checkpoint wraps the entire decoder block.",
                             "For whole-block recompute, helper saved tensors can overlap other block operations.",
                             "No MLA/KDA/HybridEP custom kernel internal workspaces or external CUDA allocations.",
                             "Allocator replay assumes a single stream and does not impose GPU capacity."],
                full_model_fit_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--tokens", type=int, default=8136)
    parser.add_argument("--dim", type=int, default=7168)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.source, tokens=args.tokens, dim=args.dim)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["probes"]:
        print(row["residual_entries"], row["full_ac"], row["allocator"]["expandable"]["peak_live_bytes"]/2**30,
              row["after_forward_live_bytes"]/2**30)


if __name__ == "__main__":
    main()
