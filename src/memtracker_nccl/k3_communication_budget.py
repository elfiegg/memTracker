"""CPU-only K3 capacity sensitivity with communication fully resident up front.

Pipeline receive allocation and external communication initialization are separate
policies. Communication sizes here are explicit assumptions, not extrapolations
from two-rank measurements. A lower-bound screen can reject fit, not prove it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .nccl_model import _integer

GiB = 2**30


def assess(screen: dict, *, persistent_communication_bytes: int,
           provenance: str, additional_peak_bytes: int = 0) -> dict:
    """Charge disjoint communication backing for the entire training lifetime.

    persistent_communication_bytes is the aggregate locally owned NCCL/HybridEP
    backing PER GPU, counted once. It excludes PyTorch payload tensors and peer
    mappings. additional_peak_bytes is a co-resident allowance for everything
    omitted by the state screen, including allocator overhead; zero omits it.
    """
    _integer('persistent_communication_bytes', persistent_communication_bytes)
    _integer('additional_peak_bytes', additional_peak_bytes)
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError('Explicit provenance is required for communication assumptions')
    stages = []
    capacity = screen['capacity_bytes']
    for stage in screen['stages']:
        minimum = stage['decoder_steady_state_floor_bytes']
        subtotal = minimum + persistent_communication_bytes
        total = subtotal + additional_peak_bytes
        stages.append(dict(stage=stage['stage'], state_and_receive_lower_bound_bytes=minimum,
            state_receive_and_communication_bytes=subtotal,
            subtotal_with_additional_allowance_bytes=total,
            remaining_budget_for_omitted_memory_bytes=capacity-subtotal,
            remaining_after_additional_allowance_bytes=capacity-total,
            exceeds_capacity=total > capacity))
    limiting = min(stages, key=lambda s: s['remaining_after_additional_allowance_bytes'])
    return dict(receive_policy=screen['receive_policy'], communication_initialization='eager',
        persistent_communication_bytes_per_gpu=persistent_communication_bytes,
        communication_provenance=provenance, additional_peak_allowance_bytes=additional_peak_bytes,
        verdict=('exceeds_capacity_under_assumptions' if limiting['exceeds_capacity']
                 else 'not_ruled_out_by_partial_budget'),
        full_model_fit_verified=False, capacity_bytes=capacity,
        limiting_stage_for_this_bound=limiting['stage'],
        remaining_budget_for_omitted_memory_bytes=limiting['remaining_budget_for_omitted_memory_bytes'],
        remaining_after_additional_allowance_bytes=limiting['remaining_after_additional_allowance_bytes'],
        stages=stages)


def run(*, communication_gib=(0, 8, 16, 32, 64, 80)) -> dict:
    from .k3_fit_experiment import run as fit_screen
    for value in communication_gib:
        _integer('communication_gib', value)
    results = []
    for policy in ('lazy', 'eager'):
        screen = fit_screen(receive_policy=policy)
        for amount in communication_gib:
            results.append(assess(screen, persistent_communication_bytes=amount*GiB,
                provenance='Synthetic sensitivity input for total NCCL+HybridEP external backing; not a measured footprint'))
    return dict(schema_version=1, kind='CPU lower-bound sensitivity; not full training simulation',
        communication_initialization='eager',
        assumptions=[
            'All declared communication backing is resident before model allocation and persists through every optimizer step.',
            'Total locally owned communication backing is counted once per GPU; no world-size, expert-count, or microbatch-count multiplier.',
            'Pipeline receive policy is independent of communication initialization policy.',
            'Zero communication bytes is the algebraic baseline, not a real zero-overhead prediction.',
            'Remaining budget must cover every omitted component, not just activations.',
            'No discount is taken for lazy communicator initialization.'],
        excluded=['Complete activation/recomputation lifetimes', 'Additional dense gathers and dense optimizer state',
                  'Optimizer and kernel scratch', 'CUDA context/library overhead',
                  'Allocator rounding, cached/pending backing and fragmentation',
                  'Communication temporaries beyond declared persistent backing'],
        scenarios=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--communication-gib', type=int, nargs='+', default=[0, 8, 16, 32, 64, 80])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(communication_gib=args.communication_gib)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for row in result['scenarios']:
        print(f"receives={row['receive_policy']}, eager comm={row['persistent_communication_bytes_per_gpu']/GiB:g} GiB: "
              f"{row['remaining_budget_for_omitted_memory_bytes']/GiB:.3f} GiB remaining; {row['verdict']}")


if __name__ == '__main__':
    main()
