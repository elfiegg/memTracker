"""Compare GPU checkpoints with allocator predictions; retain unassigned residuals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .allocator_experiment import fragmentation

CHECKPOINTS = ('allocator_3x16', 'allocator_3x14', 'allocator_fragmentation', 'allocator_released')


def analyze(root: Path, *, topology: str | None = None) -> dict:
    runs = []
    for mode in ('fixed', 'expandable'):
        predicted = {r['phase']: r for r in fragmentation(expandable=mode == 'expandable')['checkpoints']}
        files = sorted((root/mode).glob('rank-[0-9]*.json'))
        # Snapshot filenames also begin rank-N, so require an integer stem suffix.
        files = [p for p in files if p.stem.removeprefix('rank-').isdigit()]
        if not files:
            raise ValueError(f'No rank reports in {root/mode}')
        reports = [json.loads(path.read_text()) for path in files]
        world = reports[0]['world_size']
        if any(r['world_size'] != world for r in reports) or sorted(r['rank'] for r in reports) != list(range(world)):
            raise ValueError(f'Incomplete or inconsistent rank set for {mode}')
        for path in files:
            measured = json.loads(path.read_text())
            phases = {row['phase']: row for row in measured['phases']}
            comparisons = []
            for name in CHECKPOINTS:
                prediction, actual = predicted[name], phases[name]
                comparisons.append({'phase': name,
                    'predicted_allocated_bytes': prediction['allocated_bytes'],
                    'measured_allocated_bytes': actual['allocated_bytes'],
                    'allocated_error_bytes': prediction['allocated_bytes'] - actual['allocated_bytes'],
                    'predicted_reserved_bytes': prediction['reserved_bytes'],
                    'measured_reserved_bytes': actual['reserved_bytes'],
                    'reserved_error_bytes': prediction['reserved_bytes'] - actual['reserved_bytes']})
            baseline = phases['payload_before_nccl']['driver_residual_bytes']
            runs.append({
                'allocator_mode': mode, 'rank': measured['rank'], 'world_size': measured['world_size'],
                'raw_report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'runtime': {k: measured[k] for k in ('torch_version', 'cuda_version', 'nccl_version',
                                                    'device_name', 'device_total_memory', 'driver')},
                'allocator_config': measured['environment'].get('PYTORCH_ALLOC_CONF'),
                'runtime_environment': {k: v for k, v in measured['environment'].items() if k in {
                    'NCCL_MNNVL_CLIQUE_ID', 'NCCL_MNNVL_CROSS_NVLD', 'NCCL_MNNVL_CROSS_CLIQUE',
                    'NCCL_NET_PLUGIN', 'NCCL_VERSION', 'NCCL_SOCKET_IFNAME', 'NCCL_DEBUG_SUBSYS',
                    'NCCL_DEBUG', 'NCCL_IB_SL', 'NCCL_ALGO', 'NCCL_PROTO', 'NCCL_NVLS_ENABLE',
                    'NCCL_NVLS_CHUNKSIZE', 'NCCL_NVLS_NCHANNELS', 'NCCL_MIN_CTAS', 'NCCL_MAX_CTAS',
                    'NCCL_BUFFSIZE', 'NCCL_RUNTIME_CONNECT'}},
                'allocator_comparisons': comparisons,
                'phases': measured['phases'],
                'external_calibration': {
                    'category': 'Unattributed CUDA/NCCL/driver overhead', 'evidence': 'measured',
                    'pre_nccl_driver_residual_bytes': baseline,
                    'checkpoint_deltas_from_pre_nccl': {
                        name: phases[name]['driver_residual_bytes'] - baseline
                        for name in ('communicator_created', 'first_small_collective',
                                     'repeat_small_collective', 'first_large_collective',
                                     'repeat_large_collective', 'second_communicator',
                                     'communicators_destroyed', 'all_released')},
                    'ownership': 'Unattributed; do not add an NCCL formula/profile on top of this residual.',
                },
                'limitations': measured['limitations'],
            })
    return {'schema_version': 1, 'kind': 'measured GPU calibration and independent allocator replay',
            'topology': topology,
            'scope': 'One run per allocator mode; validates only this allocation schedule. External residuals are observations, not predictions.',
            'runs': runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument("--topology", help="Explicit description from the launch/topology manifest")
    args = parser.parse_args()
    result = analyze(args.input, topology=args.topology)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    errors = [abs(row['reserved_error_bytes']) for run in result['runs'] for row in run['allocator_comparisons']]
    print(f'{len(result["runs"])} rank reports; maximum reserved-backing error = {max(errors)} bytes')


if __name__ == '__main__':
    main()
