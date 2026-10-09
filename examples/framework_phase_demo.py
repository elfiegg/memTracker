"""Small model-independent FakeTensor/CUDA phase comparison harness.

Run fake and CUDA modes in separate fresh processes with identical arguments.
This uses one device; no distributed job or communicator is created.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.checkpoint import checkpoint

from memtracker_nccl.allocator_model import AllocatorConfig
from memtracker_nccl.cuda_capture import PhaseRecorder, TorchCudaProbe
from memtracker_nccl.phase_memory import analyze_capture
from memtracker_nccl.report_io import write_report
from memtracker_nccl.tensor_inventory import tensor_inventory
from memtracker_nccl.tracker import ExtendedMemTracker


def run(*, mode, family, width, batch, steps, full_ac, expandable):
    config = dict(family=family, width=width, batch=batch, steps=steps, full_ac=full_ac,
                  expandable_segments=expandable, optimizer='AdamW', foreach=False, dtype='float32')
    identity = dict(effective_config=config,
        effective_config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())
    device = 'cuda' if mode == 'cuda' else 'cpu'
    context = FakeTensorMode(allow_fallback_kernels=False) if mode == 'fake' else nullcontext()
    with context:
        recorder = (ExtendedMemTracker(allocator_config=AllocatorConfig(expandable_segments=expandable))
                    if mode == 'fake' else PhaseRecorder(probe=TorchCudaProbe(isolated_peaks=True), metadata=identity))
        with recorder:
            with recorder.phase('model_initialization'):
                if family == 'mlp':
                    model = torch.nn.Sequential(torch.nn.Linear(width,width*2,device=device), torch.nn.GELU(), torch.nn.Linear(width*2,width,device=device))
                    inputs = torch.randn(batch,width,device=device)
                else:
                    model = torch.nn.Sequential(torch.nn.Conv2d(2,width,3,padding=1,device=device), torch.nn.GELU(), torch.nn.Conv2d(width,2,1,device=device))
                    inputs = torch.randn(batch,2,8,8,device=device)
                optimizer = torch.optim.AdamW(model.parameters(),foreach=False)
            if mode == 'cuda':
                recorder.owner_provider = lambda: tensor_inventory([model], [optimizer])
            for step in range(1, steps+1):
                if mode == 'fake':
                    recorder.reset_mod_stats()
                with recorder.phase('prepare_step', step=step):
                    optimizer.zero_grad(set_to_none=True)
                with recorder.phase('forward_backward', step=step):
                    output = checkpoint(model, inputs, use_reentrant=False) if full_ac else model(inputs)
                    loss = output.square().mean()
                    loss.backward()
                    del output, loss
                with recorder.phase('optimizer', step=step):
                    optimizer.step()
        report = recorder.report()
        if mode == 'fake':
            report['metadata'] = identity
        else:
            try:
                report['analysis'] = analyze_capture(report)
            except Exception as error:
                report['analysis_error'] = f'{type(error).__name__}: {error}'
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('fake','cuda'), default='fake')
    parser.add_argument('--family', choices=('mlp','conv'), default='mlp')
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--full-ac', action='store_true')
    parser.add_argument('--expandable', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if min(args.width,args.batch,args.steps) < 1:
        parser.error('width, batch and steps must be positive')
    if args.mode == 'cuda':
        import os
        # Must be configured before CUDA initialization; the fake model uses the
        # same requested allocator mode. Other allocator knobs remain unsupported.
        os.environ['PYTORCH_ALLOC_CONF'] = f'backend:native,expandable_segments:{args.expandable}'
        os.environ.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    report=run(mode=args.mode,family=args.family,width=args.width,batch=args.batch,
               steps=args.steps,full_ac=args.full_ac,expandable=args.expandable)
    write_report(args.output,report)
    if 'analysis_error' in report:
        raise RuntimeError(f"Capture saved, attribution failed: {report['analysis_error']}")


if __name__ == '__main__': main()
