"""CPU replay of K3 Interleaved1F1B storage boundaries; partial peak, not full fit.

A source-extracted schedule supplies real compute/receive/reduction order. This
replay models unique checkpoint input storages and wire payloads, not GPU kernels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .k3_experiment import inventory
from .nccl_model import _integer

GiB = 2**30
_ACTION = re.compile(r'^(\d+)(REDUCE_GRAD|UNSHARD|RESHARD|RECV_F|RECV_B|SEND_F|SEND_B|F|B)(\d*)$')


def parse_action(value):
    match = _ACTION.fullmatch(value)
    if not match:
        raise ValueError(f'Unsupported schedule action: {value}')
    stage, op, mb = match.groups()
    return int(stage), op, int(mb) if mb else None


def stage_shapes(stage, *, tokens, last_stage, dim=7168, block_size=12):
    """Unique saved FullAC inputs + PP outputs; shared residual stacks counted once.

    Each layer has a distinct hidden input. A residual stack changes storage only
    at a block boundary (torch.cat); otherwise all layers share the same storage.
    The fresh stage-entry assembled stack is distinct from incoming delta backing.
    """
    first, last = stage['first_layer'], stage['last_layer']
    h = tokens*dim*2
    prefixes = {(layer+block_size-1)//block_size for layer in range(first, last+1)}
    checkpoint_inputs = (last-first+1 + sum(prefixes))*h
    # _pack_outgoing_delta uses torch.stack: a distinct retained payload allocation.
    outgoing = (1+len(stage['outgoing_delta_blocks']))*h if stage['stage'] != last_stage else 0
    return dict(checkpoint_input_bytes=checkpoint_inputs,
                retained_boundary_bytes=checkpoint_inputs+outgoing,
                forward_receive_bytes=(1+len(stage['incoming_delta_blocks']))*h if stage['stage'] else 0,
                backward_receive_bytes=outgoing,
                distinct_checkpoint_stack_sizes=sorted(prefixes))


def run(schedule, *, sequence=4068, microbatch_size=2, capacity_bytes=197940150272):
    for name, value in dict(sequence=sequence, microbatch_size=microbatch_size, capacity_bytes=capacity_bytes).items():
        _integer(name, value, 1)
    if (schedule['pp'], schedule['virtual_stages'], schedule['stages_per_rank']) != (8,16,2):
        raise ValueError('This model currently covers PP8 with two virtual stages per rank')
    items=inventory(); by_stage={}; stage_reports=[]
    for s in schedule['stages']:
        local=[row for row in items if row['layer'] is not None and s['first_layer'] <= row['layer'] <= s['last_layer']]
        dense=sum(row['dense_parameters'] for row in local)
        experts=sum(row['expert_parameters'] for row in local)//32
        # Embeddings/vision on first stage; output/norm on last. Text-only: no vision grads.
        extras=[row for row in items if row['layer'] is None and
                ((s['stage']==0 and row['name'] in ('tok_embeddings','vision_encoder')) or
                 (s['stage']==15 and row['name']=='output'))]
        extra_params=sum(row['dense_parameters'] for row in extras)
        extra_grads=sum(row['dense_parameters'] for row in extras if row['name']!='vision_encoder')
        shapes=stage_shapes(s,tokens=sequence*microbatch_size,last_stage=15)
        entry=dict(s, **shapes,
            persistent_parameter_bytes=2*((dense+extra_params+31)//32+experts),
            persistent_expert_momentum_bytes=2*experts,
            fp32_accumulated_gradient_bytes=4*(dense+extra_grads+experts))
        by_stage[s['stage']]=entry;stage_reports.append(entry)
    ranks=[]
    for rank in range(8):
        selected=[s for s in stage_reports if s['rank']==rank]
        base=sum(s['persistent_parameter_bytes']+s['persistent_expert_momentum_bytes'] for s in selected)
        active={}; pending_f={};pending_b={};grad_stages=set();finished=set();forwarded=set()
        max_live=0;max_by_stage=Counter();max_pending_f=0;max_pending_b=0
        peak=base;peak_event=None;boundary_peak=0;gradient_peak=0
        before_first_backward=None
        for pos, action in enumerate(schedule['actions'][str(rank)]):
            stage, op, mb=parse_action(action);key=(stage,mb);s=by_stage[stage]
            if s['rank']!=rank:raise ValueError('Schedule stage is on wrong rank')
            if op=='RECV_F':
                if key in pending_f or key in forwarded:raise ValueError('Duplicate forward receive')
                pending_f[key]=s['forward_receive_bytes']
            elif op=='RECV_B':
                if key in pending_b or key in finished:raise ValueError('Duplicate backward receive')
                pending_b[key]=s['backward_receive_bytes']
            elif op=='F':
                if key in forwarded:raise ValueError('Duplicate forward')
                if stage:pending_f.pop(key)
                active[key]=s['retained_boundary_bytes'];forwarded.add(key)
            elif op=='B':
                if before_first_backward is None:before_first_backward=len(active)
                if stage!=15:pending_b.pop(key)
                active.pop(key);finished.add(key);grad_stages.add(stage)
            elif op=='REDUCE_GRAD':
                # Drop full accumulated gradients here; reduced local gradients
                # and reduction temporaries are intentionally omitted.
                grad_stages.discard(stage)
            count=Counter(k[0] for k in active)
            for st, n in count.items():max_by_stage[st]=max(max_by_stage[st],n)
            max_live=max(max_live,len(active));max_pending_f=max(max_pending_f,len(pending_f));max_pending_b=max(max_pending_b,len(pending_b))
            retained=sum(active.values()); pending=sum(pending_f.values())+sum(pending_b.values())
            grads=sum(by_stage[st]['fp32_accumulated_gradient_bytes'] for st in grad_stages)
            total=base+retained+pending+grads
            boundary_peak=max(boundary_peak,retained);gradient_peak=max(gradient_peak,grads)
            if total>peak:
                peak=total;peak_event=dict(position=pos,action=action,live_stage_microbatches=len(active),
                    live_by_stage=dict(count),persistent_state_bytes=base,accumulated_gradient_bytes=grads,
                    retained_boundary_bytes=retained,pending_receive_bytes=pending,subtotal_bytes=total)
        expected={(s['stage'],mb) for s in selected for mb in range(schedule['microbatches'])}
        if forwarded!=expected or finished!=expected or active or pending_f or pending_b or grad_stages:
            raise ValueError('Schedule did not drain all modeled lifetimes')
        ranks.append(dict(rank=rank,virtual_stages=[s['stage'] for s in selected],
            layers=[[s['first_layer'],s['last_layer']] for s in selected],
            peak_live_pairs=max_live,peak_live_per_virtual_stage=dict(max_by_stage),
            live_pairs_before_first_backward=before_first_backward,
            max_pending_forward_receives=max_pending_f,max_pending_backward_receives=max_pending_b,
            persistent_parameter_and_expert_momentum_bytes=base,
            peak_accumulated_gradient_bytes=gradient_peak,peak_retained_boundary_bytes=boundary_peak,
            partial_peak_bytes=peak,partial_peak_event=peak_event,
            communication_budget=[dict(eager_communication_gib=c,
                partial_peak_plus_communication_bytes=peak+c*GiB,
                remaining_for_omitted_memory_bytes=capacity_bytes-peak-c*GiB,
                verdict='exceeds_capacity_under_assumptions' if peak+c*GiB>capacity_bytes else 'fit_unproven')
                for c in (0,8,16,32)]))
    return dict(schema_version=1,kind='source-schedule replay with partial storage lifetimes; not full model execution',
        scenario=dict(sequence=sequence,microbatch_size=microbatch_size,microbatches=schedule['microbatches'],
            pp=8,fsdp=32,ep=32,tp=1,cp=1,virtual_stages=16,schedule='Interleaved1F1B',
            checkpointing='FullAC',parameter_dtype='BF16',reduction_dtype='FP32',optimizer='DistMuon + AdamW',
            cuda_graphs=False,communication_initialization='eager',receive_policy='on demand at scheduled RECV',
            defer_pp_recv=schedule['defer_pp_recv']),capacity_bytes=capacity_bytes,
        source_provenance={k:schedule[k] for k in ('torch_runtime','training_source','source_sha256')},
        assumptions=['Later optimizer iteration: expert momentum already exists.',
            'All text-path trainable parameters get full FP32 accumulated gradient allocations after their virtual stage backward.',
            'FullAC keeps layer inputs; unchanged residual stacks alias one storage within a stage.',
            'Only completed-action boundaries are sampled; intra-kernel peaks are omitted.',
            'Communication allowances are assumed aggregate backing per GPU, resident for the entire timeline.',
            'Per-stage activation maxima are not added independently; the common local action sequence determines co-residency.'],
        excluded=['Additional gathered dense parameters and dense optimizer state',
            'Backward/recompute kernels and their intermediates; FullAC preserved effects',
            'Rank-local residual cache backing and backward gradient deposits beyond counted aliases',
            'Last-stage loss/output retained intermediates; input/mask/position storage',
            'CUDA/library memory and communication temporaries',
            'Allocator cache, rounding, pending stream frees and fragmentation'],
        full_model_fit_verified=False,stages=stage_reports,ranks=ranks,
        largest_partial_peak_rank=max(ranks,key=lambda r:r['partial_peak_bytes'])['rank'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule',type=Path,required=True)
    parser.add_argument('--sequence',type=int,default=4068)
    parser.add_argument('--microbatch-size',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(json.loads(args.schedule.read_text()),sequence=args.sequence,microbatch_size=args.microbatch_size)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in result['ranks']:
        print(f"rank {r['rank']}: {r['peak_live_pairs']} live pairs, boundary peak {r['peak_retained_boundary_bytes']/GiB:.3f} GiB, "
              f"partial simultaneous peak {r['partial_peak_bytes']/GiB:.3f} GiB, "
              f"remaining with 16 GiB comm {r['communication_budget'][2]['remaining_for_omitted_memory_bytes']/GiB:.3f} GiB")


if __name__=='__main__':main()
