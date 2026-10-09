#!/usr/bin/env python3
"""Run under torchrun, in fresh processes per allocator config, on reserved GPUs.

Example: PYTORCH_ALLOC_CONF=backend:native,expandable_segments:True \
 torchrun --standalone --nproc-per-node=2 scripts/calibrate_cuda.py --output work/gpu
No MemTracker/private hooks are required for the hardware measurement.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import socket
import subprocess


def residual(used: int, reserved: int) -> int:
    """Unattributed device-wide residual, never automatically labeled NCCL."""
    return used - reserved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPUs required; run CPU simulations separately')
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if world < 2:
        raise RuntimeError('NCCL calibration requires torchrun with at least two ranks')
    torch.cuda.set_device(local)
    args.output.mkdir(parents=True, exist_ok=True)
    phases = []

    def sample(name):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        reserved = torch.cuda.memory_reserved()
        row = dict(phase=name, allocated_bytes=torch.cuda.memory_allocated(),
                   reserved_bytes=reserved, driver_used_bytes=total-free,
                   driver_residual_bytes=residual(total-free, reserved),
                   inactive_split_bytes=stats.get('inactive_split_bytes.all.current'),
                   active_bytes=stats.get('active_bytes.all.current'))
        try:
            used = torch.cuda.device_memory_used(local)
            row.update(device_used_bytes=used, unattributed_device_bytes=residual(used, reserved))
        except Exception as exc:
            row['device_used_unavailable'] = f'{type(exc).__name__}: {exc}'
        phases.append(row)
        (args.output/f'rank-{rank}-{name}-snapshot.json').write_text(json.dumps(torch.cuda.memory_snapshot()))
        print(json.dumps(dict(rank=rank, **row)), flush=True)

    torch.cuda.init()
    sample('context')
    blocks = [torch.empty(16*2**20, dtype=torch.uint8, device='cuda') for _ in range(3)]
    sample('allocator_3x16')
    del blocks
    gc.collect()
    blocks = [torch.empty(14*2**20, dtype=torch.uint8, device='cuda') for _ in range(3)]
    sample('allocator_3x14')
    extra = torch.empty(5*2**20, dtype=torch.uint8, device='cuda')
    sample('allocator_fragmentation')
    del blocks, extra
    gc.collect()
    torch.cuda.empty_cache()
    sample('allocator_released')
    payload = torch.ones(8*2**20, device='cuda', dtype=torch.float32)
    sample('payload_before_nccl')
    dist.init_process_group('nccl')
    sample('communicator_created')
    dist.all_reduce(payload[:1024])
    sample('first_small_collective')
    dist.all_reduce(payload[:1024])
    sample('repeat_small_collective')
    dist.all_reduce(payload)
    sample('first_large_collective')
    dist.all_reduce(payload)
    sample('repeat_large_collective')
    group = dist.new_group(list(range(world)), backend='nccl')
    dist.all_reduce(payload, group=group)
    sample('second_communicator')
    dist.destroy_process_group(group)
    del group
    dist.destroy_process_group()
    gc.collect()
    sample('communicators_destroyed')
    del payload
    gc.collect()
    torch.cuda.empty_cache()
    sample('all_released')
    props = torch.cuda.get_device_properties(local)
    try:
        driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
    except Exception as exc:
        driver = str(exc)
    result = dict(schema_version=1, kind='GPU measurement, isolated synthetic schedule',
                  rank=rank, world_size=world, local_rank=local, hostname=socket.gethostname(),
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  nccl_version=torch.cuda.nccl.version(), device_name=props.name,
                  device_total_memory=props.total_memory, driver=driver,
                  environment={k: v for k, v in os.environ.items() if k.startswith(('NCCL_', 'PYTORCH_ALLOC'))},
                  phases=phases,
                  limitations=['Device-wide residual includes CUDA context, libraries, NCCL, driver accounting and any other process.',
                               'Checkpoint deltas are NCCL-associated, not exact ownership attribution.',
                               'Synchronized checkpoints do not measure asynchronous transient peaks.',
                               'This topology and version do not calibrate other clusters or communicator sizes.'])
    (args.output/f'rank-{rank}.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
