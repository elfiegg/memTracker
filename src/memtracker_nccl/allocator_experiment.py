"""CPU reproducible scenarios for backing, lazy NCCL, and explicit other memory."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from .allocator_model import AllocatorConfig, CachingAllocatorModel, MiB


def fragmentation(*, expandable: bool) -> dict:
    model = CachingAllocatorModel(AllocatorConfig(expandable_segments=expandable))
    checkpoints = []
    def checkpoint(name):
        checkpoints.append({'phase': name, **model.snapshot()['cpu']})
    for i in range(3):
        model.allocate(f'old-{i}', 16*MiB)
    checkpoint('allocator_3x16')
    for i in range(3):
        model.free(f'old-{i}')
    for i in range(3):
        model.allocate(f'new-{i}', 14*MiB)
    checkpoint('allocator_3x14')
    model.allocate('extra', 5*MiB)
    checkpoint('allocator_fragmentation')
    for i in range(3):
        model.free(f'new-{i}')
    model.free('extra')
    checkpoint('cached_after_free')
    model.empty_cache()
    checkpoint('allocator_released')
    return {'model': model.describe(), 'checkpoints': checkpoints}


def combined_timeline(*, expandable: bool, late_nccl: bool) -> dict:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from .tracker import ExtendedMemTracker
    from .nccl_model import NcclMemoryModel, calibrated_profile
    from .overhead_model import MemoryComponent, OtherMemoryModel

    with FakeTensorMode(allow_fallback_kernels=False):
        tracker = ExtendedMemTracker(record_tensor_events=True,
                     allocator_config=AllocatorConfig(expandable_segments=expandable))
        other = OtherMemoryModel(tracker)
        nccl = NcclMemoryModel(tracker)
        profile = calibrated_profile(8*MiB, provenance='Synthetic 8 MiB allowance; not measured NCCL')
        nccl.register_communicator('dp', profile)
        with tracker:
            other.start(MemoryComponent('context', 4*MiB, 'CUDA context', 'assumed', 'Synthetic scenario'))
            if not late_nccl:
                nccl.complete_collective(nccl.begin_collective('dp'))
            large = torch.empty(25*MiB, dtype=torch.uint8)
            with other.scope(MemoryComponent('workspace', 3*MiB, 'External library workspace',
                                            'assumed', 'Synthetic externally owned workspace')):
                pass
            del large
            gc.collect()
            tracker.empty_allocator_cache()
            small = torch.empty(1*MiB, dtype=torch.uint8)
            first = nccl.begin_collective('dp', payload_bytes=MiB, temporary_bytes=2*MiB)
            second = nccl.begin_collective('dp', payload_bytes=MiB, temporary_bytes=2*MiB)
            tracker.record_tensor_stream(small, 'two-collectives')
            del small
            gc.collect()
            nccl.complete_collective(first)
            nccl.complete_collective(second)
            tracker.complete_stream('two-collectives')
            nccl.destroy_communicator('dp')
            other.stop('context')
            tracker.empty_allocator_cache()
        report = tracker.report()
        # Drop randomized allocation namespaces; retain exact numerical timeline.
        report['events'] = [{k: event[k] for k in ('sequence', 'event', 'devices', 'resident')}
                            for event in report['events']]
        report['inputs'] = {'expandable_segments': expandable, 'late_nccl': late_nccl,
                            'nccl': profile.to_dict(), 'other': other.describe()}
        return report


def run() -> dict:
    return {'schema_version': 1, 'kind': 'synthetic CPU scenarios, not GPU predictions',
            'fragmentation': {mode: fragmentation(expandable=(mode == 'expandable'))
                              for mode in ('fixed', 'expandable')},
            'combined': {f'{mode}_{timing}': combined_timeline(expandable=(mode == 'expandable'),
                                                            late_nccl=(timing == 'late'))
                         for mode in ('fixed', 'expandable') for timing in ('early', 'late')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for name, report in result['combined'].items():
        print(f"{name}: {report['resident']['peak']['cpu']['combined_bytes']/MiB:g} MiB modeled backing+external")


if __name__ == '__main__':
    main()
