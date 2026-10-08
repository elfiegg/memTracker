#!/usr/bin/env python3
"""Single-GPU allocation witness for a K3 stage lower bound; NOT model execution."""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', type=int, choices=range(8), default=6)
    args = parser.parse_args()
    import torch
    from torch.distributed.pipelining.stage import PipelineStage, _PipelineStageBase
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    from memtracker_nccl.pipeline_buffers import allocate_receive_buffers, buffer_bytes

    if not torch.cuda.is_available():
        raise RuntimeError('Run on an allocated CUDA GPU')
    torch.cuda.set_device(0)
    scenario = json.loads(args.scenario.read_text())
    index = args.stage
    row = scenario['stages'][index]
    cfg = scenario['scenario']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the installed allocation and gradient-lifetime methods as evidence.
    funcs = [PipelineStage._setup_forward_recv_info, _PipelineStageBase._setup_backward_recv_info,
             PipelineStage._create_grad_recv_info, _PipelineStageBase.backward_maybe_with_nosync,
             FSDPParam.to_accumulated_grad_if_needed, FSDPParamGroup.post_backward]
    source = '\n\n'.join(inspect.getsource(fn) for fn in funcs)
    args.output.with_suffix('.source.txt').write_text(source)
    result = dict(kind='real CUDA receive-helper and synthetic state allocations; not a training run',
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(), stage=index,
        capacity_bytes=torch.cuda.get_device_properties(0).total_memory,
        allocator_config=__import__('os').environ.get('PYTORCH_ALLOC_CONF'),
        installed_method_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        scenario_sha256=hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        phases=[], status='running')
    def sample(phase):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        result['phases'].append(dict(phase=phase, allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(), free_bytes=free, total_bytes=total))
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    sample('baseline')
    held=[]
    phase='pipeline_receive_buffers'
    try:
        stage = allocate_receive_buffers(stage_index=index, tokens=cfg['sequence']*cfg['microbatch_size'],
            incoming_blocks=row['incoming_blocks'], outgoing_blocks=row['outgoing_blocks'],
            microbatches=cfg['microbatches'], device='cuda')
        held.append(stage)
        result['actual_pipeline_receive_bytes'] = buffer_bytes(stage)
        observed = buffer_bytes(stage)['total_bytes']
        result['observed_receive_policy'] = 'lazy' if observed == 0 else 'eager'
        if observed not in (0, row['receive_buffers_all_microbatches_bytes']):
            raise RuntimeError('Unrecognized installed receive allocation behavior')
        sample(phase)
        if observed == 0:
            # Replay allocation/ownership transfer through installed helpers.
            # No distributed communication or K3 compute is performed here.
            for chunk in range(row['outstanding_forward_microbatches']):
                for info in stage.args_recv_info.get(chunk, ()):
                    if getattr(info, 'is_root_arg', False) or info.tensor_meta is None:
                        continue
                    info.allocate_buffer('cuda')
                    held.append(info.take_buffer())
                    assert info.buffer is None
            for info in stage.grad_recv_info.get(0, ()):
                if info.tensor_meta is None:
                    continue
                info.allocate_buffer('cuda')
                held.append(info.take_buffer())
                assert info.buffer is None
            result['replayed_live_input_bytes'] = sum(
                t.numel()*t.element_size() for t in held if isinstance(t, torch.Tensor))
            expected_lazy = cfg['sequence'] * cfg['microbatch_size'] * 7168 * 2 * (
                (row['outstanding_forward_microbatches']*(1+row['incoming_blocks']) if index else 0)
                + (1+row['outgoing_blocks'] if index < 7 else 0))
            assert result['replayed_live_input_bytes'] == expected_lazy
            sample('lazy_warmup_inputs_and_one_backward_receive')
        # Only expert state: omit dense state, activations, NCCL/HybridEP and all
        # scratch. This is an allocation-only subtotal, NOT a demonstrated
        # training lifetime or a full-model fit witness.
        for phase, key in [('expert_bf16_parameters','expert_parameter_bytes'),
                           ('expert_bf16_momentum','expert_momentum_min_bytes'),
                           ('expert_fp32_accumulated_gradients','expert_fp32_accumulated_gradient_bytes')]:
            # One allocation per layer avoids introducing a giant cross-layer block.
            # Layer zero is dense; only routed-expert layers contribute.
            count=row['last_layer']-max(1, row['first_layer'])+1
            per_layer=row[key]//count
            assert per_layer*count == row[key]
            for _ in range(count):
                held.append(torch.empty(per_layer, dtype=torch.uint8, device='cuda'))
            sample(phase)
        result['status']='synthetic_subtotal_allocations_succeeded'
        result['full_model_fit_verified']=False
    except torch.OutOfMemoryError as exc:
        result.update(status='out_of_memory', failure_phase=phase, error=str(exc))
        sample('oom')
    except Exception as exc:
        result.update(status='probe_error', failure_phase=phase, error=repr(exc))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        held.clear()
        if 'stage' in locals(): del stage
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
