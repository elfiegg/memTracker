"""Conservative fit screen for the requested PP8/FSDP32/EP32 BF16 K3 recipe.

Executes real PipelineStage allocation helpers on FakeTensors. Model/gradient
storage is source-derived arithmetic, not a full-model forward/backward trace.
An over-capacity lower bound can reject fit; a below-capacity bound cannot prove it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .k3_experiment import inventory

GiB = 2**30
TRAINING_SOURCE = 'b4d5b404bd30dec67f86873203fbd4ba5f457531'


def run(*, sequence: int = 4068, microbatch_size: int = 2, microbatches: int = 64,
        capacity_bytes: int = 197940150272, receive_policy: str = "lazy") -> dict:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from .allocator_model import AllocatorConfig
    from .pipeline_buffers import allocate_receive_buffers, buffer_bytes
    from .tracker import ExtendedMemTracker

    for value in (sequence, microbatch_size, microbatches, capacity_bytes):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('All sizes and counts must be positive integers')
    if receive_policy not in ("eager", "lazy"):
        raise ValueError("receive_policy must be eager or lazy")
    # Native split balances 93 layers + one embedding weight + one output weight.
    counts = [11, 12, 12, 12, 12, 12, 12, 10]
    rows, stages, first = inventory(), [], 0
    tokens = sequence * microbatch_size
    for stage_index, count in enumerate(counts):
        selected = [r for r in rows if r['layer'] is not None and first <= r['layer'] < first+count]
        dense = sum(r['dense_parameters'] for r in selected)
        expert = sum(r['expert_parameters'] for r in selected)
        incoming = (first+11)//12 if stage_index else 0
        outgoing = (first+count+11)//12 if stage_index < 7 else 0
        # BF16 payloads: hidden [T,D] plus full accumulated block stack on each
        # hop. With one stage/rank, the receiver has no earlier local cache.
        slots = (1+incoming if stage_index else 0) + (1+outgoing if stage_index < 7 else 0)
        expected = tokens * 7168 * 2 * microbatches * slots
        with FakeTensorMode(allow_fallback_kernels=False):
            tracker = ExtendedMemTracker(allocator_config=AllocatorConfig(expandable_segments=True))
            with tracker:
                stage = allocate_receive_buffers(stage_index=stage_index, tokens=tokens,
                    incoming_blocks=incoming, outgoing_blocks=outgoing, microbatches=microbatches)
                observed = buffer_bytes(stage)
                backing = tracker.report()['allocator']['current']['cpu']['reserved_bytes']
                setup_policy = 'eager' if observed['total_bytes'] else 'lazy'
                if observed['total_bytes'] not in (0, expected):
                    raise AssertionError('Unknown PipelineStage receive setup behavior')
                del stage
        # EP32 overlays FSDP32: EDP=1. All 28 experts/rank are active under round-robin.
        # During no-sync pipeline accumulation, local expert gradients are FP32.
        # The momentum lower bound is just one BF16 element per expert parameter.
        # Target 2.15 allocates on receive and transfers ownership to compute.
        # At the first backward, warmup forward inputs are live plus one backward
        # receive. K3 assembles a same-sized stack leaf retained in fwd_cache.
        # This is a limited live-data floor, not all retained activations or peak.
        outstanding = min(microbatches, 8-stage_index)
        lazy_live = tokens * 7168 * 2 * ((outstanding*(1+incoming) if stage_index else 0)
                                          + (1+outgoing if stage_index < 7 else 0))
        receive_floor = expected if receive_policy == 'eager' else lazy_live
        local_expert = expert // 32
        expert_weights = 2 * local_expert
        expert_accum = 4 * local_expert
        expert_momentum_min = 2 * local_expert
        expert_state = expert_weights + expert_accum + expert_momentum_min
        # Eager buffers persist throughout backward. Lazy warmup input storage
        # and fully accumulated gradients need not attain their maxima together.
        # Use max, not sum, for the strict lower bound in that case.
        floor = (receive_floor + expert_state if receive_policy == 'eager'
                 else max(receive_floor, expert_state))
        # Secondary first-iteration floor includes dense decoder parameters and
        # unreduced dense FP32 gradients, but omits all optimizer state.
        decoder_weights = 2 * ((dense+31)//32 + local_expert)
        decoder_accum = 4 * (dense + local_expert)
        decoder_state = decoder_weights + decoder_accum
        first_step_floor = (receive_floor + decoder_state if receive_policy == 'eager'
                            else max(receive_floor, decoder_state))
        steady_state = decoder_state + expert_momentum_min
        steady_floor = (receive_floor + steady_state if receive_policy == 'eager'
                        else max(receive_floor, steady_state))
        stages.append(dict(stage=stage_index, first_layer=first, last_layer=first+count-1,
            incoming_blocks=incoming, outgoing_blocks=outgoing,
            receive_buffers_all_microbatches_bytes=expected,
            native_setup_observation=observed, native_setup_policy=setup_policy,
            modeled_native_setup_reserved_bytes=backing,
            outstanding_forward_microbatches=outstanding,
            receive_and_input_floor_bytes=receive_floor,
            expert_only_floor_excludes_checkpointed_layer_activations=True,
            expert_state_bytes=expert_state,
            illustrative_expert_state_plus_inputs_bytes=expert_state+receive_floor,
            local_expert_parameters=local_expert, expert_parameter_bytes=expert_weights,
            expert_fp32_accumulated_gradient_bytes=expert_accum,
            expert_momentum_min_bytes=expert_momentum_min,
            expert_only_steady_floor_bytes=floor,
            decoder_first_backward_floor_bytes=first_step_floor,
            decoder_steady_state_floor_bytes=steady_floor,
            decoder_sharded_parameter_bytes=decoder_weights,
            decoder_fp32_accumulated_gradient_bytes=decoder_accum,
            steady_state_excess_bytes=max(0, steady_floor-capacity_bytes),
            capacity_bytes=capacity_bytes, expert_floor_excess_bytes=max(0, floor-capacity_bytes),
            first_backward_excess_bytes=max(0, first_step_floor-capacity_bytes)))
        first += count
    worst = max(stages, key=lambda r: r['decoder_steady_state_floor_bytes'])
    return dict(schema_version=1, kind='source-derived lower-bound screen plus actual fake receive allocations',
        verdict='does_not_fit_under_stated_implementation' if worst['steady_state_excess_bytes'] else 'fit_unproven',
        hardware='GB200 capacity from measured cluster device; override for another device',
        capacity_bytes=capacity_bytes, largest_lower_bound_stage=worst['stage'],
        helper_torch_version=str(torch.__version__), training_source=TRAINING_SOURCE,
        target_runtime_evidence='experiments/k3_lazy_receive_gpu_probe.json' if receive_policy == 'lazy' else None, receive_policy=receive_policy,
        scenario=dict(model='full Kimi-K3, 93 layers, 896 experts, top-k16', gpus=256,
            pp=8, dp_shard=32, ep=32, edp=1, tp=1, cp=1, dp_replicate=1, schedule='1F1B',
            sequence=sequence, microbatch_size=microbatch_size, microbatches=microbatches,
            global_batch=microbatch_size*microbatches*32,
            tokens_per_optimizer_step=sequence*microbatch_size*microbatches*32,
            parameter_dtype='bfloat16', compute_dtype='bfloat16', reduction_dtype='float32',
            optimizer='DistMuon + AdamW', learning_rate=8e-4, weight_decay=0,
            checkpointing='FullAC', routing='forced round-robin', moe_backend='HybridEP',
            cuda_graphs=False, qat=False, fsdp_reshard_after_forward='always'),
        assumptions=[
            'Native PP8 split uses default first/last weights 1 and attention-residual block size12.',
            f'Declared target receive policy: {receive_policy}; local helper observation is reported separately.',
            'Lazy floor includes outstanding K3 stage input leaves plus one backward receive; other activations and async keepalives omitted.',
            'Pipeline FSDP disables gradient synchronization within the step; FP32 accumulated gradients remain live.',
            'All counted decoder parameters participate in backward; dense gradients have their full allocated shapes after a no-sync backward.',
            'Expert momentum persists after the first optimizer step, with at least BF16 storage.',
            'Lazy input and accumulated-gradient peaks are not assumed simultaneous; their maximum gives the lower bound.',
            'Expert-only floor intentionally excludes all dense state and all transient compute memory.'],
        excluded=['Dense parameter/gradient/AdamW state from expert-only floor',
            'Checkpointed activations and recompute workspaces', 'NCCL, HybridEP backing, CUDA context/libraries',
            'Allocator cache/rounding from capacity comparison', 'Gathered parameters and optimizer scratch',
            'Input/target buffers and vision execution'],
        coverage=dict(full_model_forward_backward_executed=False, full_model_peak_predicted=False,
            actual_pipeline_buffer_helpers_executed_with_faketensors=True), stages=stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=int, default=4068)
    parser.add_argument('--microbatch-size', type=int, default=2)
    parser.add_argument('--microbatches', type=int, default=64)
    parser.add_argument('--capacity-bytes', type=int, default=197940150272)
    parser.add_argument('--receive-policy', choices=['eager', 'lazy'], default='lazy')
    parser.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args()); output = args.pop('output')
    result = run(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['verdict'])
    for row in result['stages']:
        print(f"stage {row['stage']}: PP buffers {row['receive_and_input_floor_bytes']/GiB:.3f} GiB; "
              f"expert-only steady floor {row['expert_only_steady_floor_bytes']/GiB:.3f} GiB; "
              f"decoder steady state floor {row['decoder_steady_state_floor_bytes']/GiB:.3f} GiB")


if __name__ == '__main__':
    main()
