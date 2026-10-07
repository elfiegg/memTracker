"""Source-derived K3 state/gather envelope; NOT a full K3 training simulation."""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from .tracker import ExtendedMemTracker

COMMIT = "948d65c868c5fa8f0290bcf9e54b69f004721f54"
SOURCE = f"https://github.com/pytorch/torchtitan/blob/{COMMIT}/"


@dataclass(frozen=True)
class K3Shape:
    dim: int = 7168
    vocab: int = 163840
    layers: int = 93
    heads: int = 96
    q_rank: int = 1536
    kv_rank: int = 512
    nope: int = 128
    rope: int = 64
    head_dim: int = 128
    dense_hidden: int = 33792
    latent: int = 3584
    expert_hidden: int = 3072
    experts: int = 896
    top_k: int = 16
    shared_experts: int = 2
    conv_kernel: int = 4
    vision_dim: int = 1024
    vision_qkv: int = 1536
    vision_hidden: int = 4096
    vision_layers: int = 27
    vision_pos: int = 64


def inventory(shape: K3Shape = K3Shape()) -> list[dict]:
    """Parameter elements from shapes in flavors.py, not an imported K3 model.

    No biases in these linears/convolutions. RMSNorm contributes one vector.
    All 896 experts are counted, regardless of top-k active routing.
    """
    s = shape
    d, h, k = s.dim, s.heads, s.head_dim
    projection = h * k
    result = []
    for layer in range(s.layers):
        mla = layer % 4 == 3 or layer == s.layers - 1
        if mla:
            attention = (d * s.q_rank + s.q_rank
                         + s.q_rank * h * (s.nope + s.rope)
                         + d * (s.kv_rank + s.rope) + s.kv_rank
                         + s.kv_rank * h * (s.nope + k)
                         + 2 * d * projection)
        else:
            attention = (5 * d * projection + 3 * projection * s.conv_kernel
                         + d * k + k * projection + d * h
                         + k + h + projection)
        # attention_norm, ffn_norm; two norms+projections for attention residuals,
        # except layer zero has only the FFN residual pair.
        norms_and_residuals = (4 if layer == 0 else 6) * d
        if layer == 0:
            ffn, expert = 3 * d * s.dense_hidden, 0
        else:
            expert = 3 * s.experts * s.latent * s.expert_hidden
            ffn = (d * s.experts + 2 * d * s.latent + s.latent
                   + 3 * d * s.shared_experts * s.expert_hidden)
        result.append({"name": f"layers.{layer}", "layer": layer,
                       "attention": "MLA" if mla else "KDA",
                       "dense_parameters": attention + norms_and_residuals + ffn,
                       "expert_parameters": expert,
                       # quantile histogram, tokens/expert, expert bias; buffers replicated
                       "replicated_buffer_bytes": 0 if layer == 0 else s.experts * (1000 * 4 + 8)})
    v, q, f = s.vision_dim, s.vision_qkv, s.vision_hidden
    vision = (3 * 14 * 14 * v + s.vision_pos**2 * v
              + s.vision_layers * (4 * v * q + 2 * v * f + 2 * v)
              + v + (4 * v)**2 + 4 * v * d + d)
    result.extend([
        {"name": "tok_embeddings", "layer": None, "dense_parameters": s.vocab*d, "expert_parameters": 0, "replicated_buffer_bytes": 0},
        {"name": "vision_encoder", "layer": None, "dense_parameters": vision, "expert_parameters": 0, "replicated_buffer_bytes": 0},
        {"name": "output", "layer": None, "dense_parameters": s.vocab*d + 3*d, "expert_parameters": 0, "replicated_buffer_bytes": 0},
    ])
    return result


def simulate(*, pp: int = 8, dp: int = 64, ep: int = 8,
             prefetch: int = 2, nccl_mib_per_communicator: int = 128,
             communicators: int = 3, activation_mib: int = 0,
             nccl_profile: str = "allowance", channels: int = 16) -> dict:
    """Run real fake tensor allocations for an explicitly supplied partial schedule.

    TP=CP=DP_REPLICATE=1. EP overlays DP, EDP=DP/EP. FP32 parameters,
    gradients, Adam m/v; BF16 gather outputs. All state co-resident is an
    envelope assumption. Activations default to OMITTED, not predicted zero.
    """
    from .nccl_model import NcclMemoryModel, calibrated_profile, shared_net_profile
    from .fake_collectives import FakeCollectiveRunner
    if pp < 1 or pp > 93 or dp < 1 or ep < 1 or dp % ep or 896 % ep:
        raise ValueError("Require 1<=PP<=93, DP divisible by EP, and EP dividing 896")
    if prefetch < 1 or min(nccl_mib_per_communicator, communicators, activation_mib) < 0:
        raise ValueError("Prefetch must be positive; byte allowances/counts nonnegative")
    items = inventory()
    stages = [[] for _ in range(pp)]
    for row in items:
        if row["layer"] is not None:
            stage = min(row["layer"] * pp // 93, pp - 1)
        else:
            stage = pp - 1 if row["name"] == "output" else 0
        stages[stage].append(row)
    if communicators < 2:
        raise ValueError("Need at least two modeled communicators: dense DP and expert EDP")
    if nccl_profile == "allowance":
        profile = calibrated_profile(nccl_mib_per_communicator * 2**20,
                provenance="Uncalibrated scenario allowance, NOT measured NCCL memory")
    elif nccl_profile == "shared-net-component":
        profile = shared_net_profile(channels=channels)
    else:
        raise ValueError("Unknown NCCL profile")
    results = []
    for stage, rows in enumerate(stages):
        # Equal element division with padding; no per-parameter FSDP pad model.
        local_params = sum((r["dense_parameters"] + dp-1)//dp
                           + (r["expert_parameters"] + dp-1)//dp for r in rows)
        buffer_bytes = sum(r["replicated_buffer_bytes"] for r in rows)
        with FakeTensorMode(allow_fallback_kernels=False):
            tracker = ExtendedMemTracker()
            model = NcclMemoryModel(tracker)
            runner = FakeCollectiveRunner(model)
            with tracker:
                state = [torch.empty(local_params, dtype=torch.float32) for _ in range(4)]
                buffers = torch.empty(buffer_bytes, dtype=torch.uint8)
                activation = torch.empty(activation_mib * 2**20, dtype=torch.uint8)
                for i in range(communicators):
                    model.register_communicator(f"comm-{i}", profile)
                    token = model.begin_collective(f"comm-{i}", operation="assumed_prior_use")
                    model.complete_collective(token)
                # Simulated BF16 layer gathers. Keep N concurrent layer outputs.
                # This is a size/lifetime envelope, not automatic FSDP tracing.
                live = []
                for row in rows:
                    if len(live) >= prefetch:
                        del live[0]
                    dense_shard = torch.empty((row["dense_parameters"]+dp-1)//dp, dtype=torch.bfloat16)
                    dense = runner.all_gather(dense_shard, "comm-0", dp)
                    dense.wait()
                    expert_shard = torch.empty((row["expert_parameters"]+dp-1)//dp, dtype=torch.bfloat16)
                    expert = runner.all_gather(expert_shard, "comm-1", dp//ep)
                    expert.wait()
                    live.append((dense.output, expert.output))
                    del dense, expert, dense_shard, expert_shard
                report = tracker.report()
                for i in range(communicators):
                    model.destroy_communicator(f"comm-{i}")
                del live, state, buffers, activation
                gc.collect()
            peak = report["peak"]["cpu"]
            results.append({"stage": stage, "modules": [r["name"] for r in rows],
                            "state_bytes": local_params * 16,
                            "replicated_router_buffer_bytes": buffer_bytes,
                            "partial_envelope_bytes": peak["combined_bytes"],
                            "partial_envelope_gib": peak["combined_bytes"] / 2**30,
                            "accounting": report})
    return {
        "schema_version": 1,
        "result_kind": "partial source-derived state and gathered-parameter envelope; NOT full K3 peak",
        "source_commit": COMMIT,
        "sources": [SOURCE+p for p in ["torchtitan/models/kimi_k3/flavors.py", "torchtitan/models/common/attention/kda.py", "torchtitan/models/common/moe.py", "torchtitan/distributed/parallelism_context.py"]],
        "shape": asdict(K3Shape()), "parameter_inventory": items,
        "nccl_profile": profile.to_dict(),
        "total_parameters_source_derived": sum(r["dense_parameters"]+r["expert_parameters"] for r in items),
        "scenario": {"world_size": pp*dp, "pp": pp, "dp_shard": dp, "ep": ep,
                     "edp_shard": dp//ep, "tp": 1, "cp": 1, "dp_replicate": 1,
                     "partition": "contiguous nearly equal layer counts; NOT TorchTitan PP schedule",
                     "state_bytes_per_parameter": 16, "compute_dtype": "bfloat16",
                     "optimizer": "AdamW, FP32 parameter + gradient + two moments; all co-resident",
                     "prefetch_layers": prefetch, "modeled_communicators_per_rank": communicators,
                     "nccl_mib_per_communicator": nccl_mib_per_communicator,
                     "nccl_profile": nccl_profile, "channels": channels,
                     "activation_allowance_mib": activation_mib,
                     "batch_size": None, "sequence_length": None, "checkpointing": "not modeled"},
        "coverage": {"actual_k3_forward_backward_executed": False, "actual_fake_tensor_and_memtracker_executed": True,
                     "full_training_peak_bytes": None, "oom_prediction": None},
        "excluded": ["Activation graph, KDA/MLA saved tensors and backward workspaces", "EP dispatch/combine payloads and routing imbalance", "Pipeline sends, receives, residual cache and schedule", "Gradient reduction/full gradients and optimizer temporaries", "CUDA context, reserved allocator memory, fragmentation and library workspaces", "Exact FSDP wrapping/padding and vision rotary/cache buffers", "NCCL topology/transport discovery; allowance is uncalibrated"],
        "worst_stage_partial_envelope_gib": max(r["partial_envelope_gib"] for r in results),
        "stages": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [("pp",8),("dp",64),("ep",8),("prefetch",2),
                          ("nccl-mib-per-communicator",128),("communicators",3),("activation-mib",0)]:
        parser.add_argument("--"+name, type=int, default=default)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nccl-profile", choices=["allowance", "shared-net-component"], default="allowance")
    parser.add_argument("--channels", type=int, default=16)
    args = vars(parser.parse_args())
    output = args.pop("output")
    result = simulate(**args)
    serialized = json.dumps(result, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized+"\n")
    print(serialized)


if __name__ == "__main__":
    main()
