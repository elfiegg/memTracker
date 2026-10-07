"""Same fake payload, different explicit NCCL assumptions. No GPUs required."""
import argparse
import json
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from memtracker_nccl import ExtendedMemTracker
from memtracker_nccl.fake_collectives import FakeCollectiveRunner
from memtracker_nccl.nccl_model import NcclMemoryModel, calibrated_profile, p2p_profile


def run(profile):
    with FakeTensorMode(allow_fallback_kernels=False):
        tracker = ExtendedMemTracker()
        nccl = NcclMemoryModel(tracker)
        nccl.register_communicator("dp", profile, device="cpu")
        collectives = FakeCollectiveRunner(nccl)
        with tracker:
            # 64 MiB per rank; eight-rank gather produces a 512 MiB output.
            shard = torch.empty(32 * 2**20, dtype=torch.bfloat16)
            first = collectives.all_gather(shard, "dp", 8)
            full = first.wait()
            # Delete both holders so the first output is truly no longer live.
            del first, full
            second = collectives.all_gather(shard, "dp", 8)
            second.wait()
            del second, shard
        nccl.destroy_communicator("dp")
        return {"profile": profile.to_dict(), "tracker": tracker.report()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profiles = {
        "tensor_only": calibrated_profile(0, "zero-overhead comparison, not a measurement"),
        "p2p_4_channels": p2p_profile(channels=4),
        "p2p_16_channels": p2p_profile(channels=16),
    }
    result = {"description": "CPU-only fake collectives with assumed NCCL P2P transport configurations; not GPU measurements",
              "scenarios": {name: run(profile) for name, profile in profiles.items()}}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, scenario in result["scenarios"].items():
        peak = scenario["tracker"]["peak"]["cpu"]
        print(f"{name}: tensor={peak['tensor_bytes']/2**20:.1f} MiB, "
              f"modeled NCCL={peak['external_bytes']/2**20:.1f} MiB, "
              f"combined={peak['combined_bytes']/2**20:.1f} MiB")


if __name__ == "__main__":
    main()
