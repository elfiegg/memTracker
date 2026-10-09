import json
from pathlib import Path
import tempfile
import unittest

from memtracker_nccl.calibration_analysis import analyze


class CalibrationTests(unittest.TestCase):
    def test_saved_hardware_measurements_match_allocator_and_expose_perturbation(self):
        result = json.loads((Path(__file__).parents[1]/'experiments/gb200_calibration.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for run in result['runs']:
                path = root/run['allocator_mode']/f"rank-{run['rank']}.json"
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps({**run['runtime'], 'rank': run['rank'],
                    'world_size': run['world_size'], 'phases': run['phases'],
                    'environment': {**run.get('runtime_environment', {}), 'PYTORCH_ALLOC_CONF': run['allocator_config']},
                    'limitations': run['limitations']}))
            replay = analyze(root)
            self.assertEqual(len(replay['runs']), 4)
            self.assertTrue(all(r['reserved_error_bytes'] == 0 for run in replay['runs'] for r in run['allocator_comparisons']))
            target = root/'fixed/rank-0.json'
            altered = json.loads(target.read_text())
            next(r for r in altered['phases'] if r['phase'] == 'allocator_fragmentation')['reserved_bytes'] += 2**20
            target.write_text(json.dumps(altered))
            replay = analyze(root)
            self.assertEqual(replay['runs'][0]['allocator_comparisons'][2]['reserved_error_bytes'], -2**20)
            target.unlink()
            with self.assertRaisesRegex(ValueError, 'rank set'):
                analyze(root)
