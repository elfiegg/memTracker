"""Probe an exact K3 source function without importing CUDA-only dependencies."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from .k3_experiment import COMMIT
from .tracker import ExtendedMemTracker


def probe(source: Path, tokens: int = 8192, dim: int = 7168, residuals: int = 8) -> dict:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != COMMIT:
        raise ValueError(f"Expected exact TorchTitan revision {COMMIT}, got {revision}")
    path = source / "torchtitan/models/kimi_k3/model.py"
    original = subprocess.check_output(["git", "show", f"{COMMIT}:torchtitan/models/kimi_k3/model.py"], cwd=source)
    if path.read_bytes() != original:
        raise ValueError("Model file differs from pinned commit")
    tree = ast.parse(original)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_attention_residual")
    # Keep the exact function body, discard annotations to avoid importing K3.
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None
    program = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(program, str(path), "exec"), namespace)
    with FakeTensorMode(allow_fallback_kernels=False):
        projection = torch.nn.Linear(dim, 1, bias=False, dtype=torch.bfloat16)
        norm = torch.nn.RMSNorm(dim, eps=1e-5, dtype=torch.bfloat16)
        partial = torch.empty(tokens, dim, dtype=torch.bfloat16, requires_grad=True)
        stack = torch.empty(tokens, residuals, dim, dtype=torch.bfloat16, requires_grad=True)
        tracker = ExtendedMemTracker()
        tracker.track_external(projection, norm, partial, stack)
        with tracker:
            out = namespace["_apply_attention_residual"](partial, stack, projection, norm)
            out.sum().backward()
        result = tracker.report()
    return {"source_commit": revision, "source_file_sha256": hashlib.sha256(original).hexdigest(),
            "executed": "Exact _apply_attention_residual function body, fake forward and backward",
            "substitutions": "torch.nn.Linear and RMSNorm only supply weight/eps attributes; function never calls their forward",
            "tokens": tokens, "dim": dim, "residual_entries": residuals,
            "output_shape": list(out.shape), "accounting": result,
            "coverage": "Only attention residual helper, NOT KDA/MLA, MoE, or full training graph"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=7168)
    parser.add_argument("--residuals", type=int, default=8)
    args = parser.parse_args()
    report = probe(args.source, args.tokens, args.dim, args.residuals)
    data = json.dumps(report, indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data)
    print(data)


if __name__ == "__main__":
    main()
