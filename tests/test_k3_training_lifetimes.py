import copy
from pathlib import Path
import tempfile
import unittest

from memtracker_nccl.k3_training_lifetimes import Ledger, run
from memtracker_nccl.report_io import read_report, write_report

ROOT = Path(__file__).resolve().parents[1]


class LifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedule = read_report(ROOT/'experiments/k3_interleaved_schedule.json')
        cls.muon = read_report(ROOT/'experiments/k3_muon_source_probe.json.gz')
        cls.adam = read_report(ROOT/'experiments/k3_adamw_source_probe.json.gz')
        cls.residual = read_report(ROOT/'experiments/k3_fullac_probe.json')

    def test_missing_coverage_and_wrong_shapes_rejected(self):
        bad = dict(self.residual, tokens=8192)
        with self.assertRaises(ValueError):
            run(self.schedule, self.muon, bad, self.adam)
        with self.assertRaises(ValueError):
            run(self.schedule, self.muon, self.residual, dict(self.adam, vision_encoder_present=False))
        with self.assertRaises(ValueError):
            run(self.schedule, self.muon, self.residual, self.adam, communication_gib=float('nan'))

    def test_all_training_lifetimes_drain_before_optimizer(self):
        # Synthetic one-MB lifecycle exercises each transition cheaply. This is
        # not presented as a valid hardware Interleaved1F1B schedule.
        schedule = copy.deepcopy(self.schedule)
        schedule['microbatches'] = 1
        for rank in range(8):
            stages = [rank, rank+8]
            actions = [f'{s}UNSHARD' for s in stages]
            for s in stages:
                if s: actions.append(f'{s}RECV_F0')
                actions.append(f'{s}F0')
            for s in reversed(stages):
                if s != 15: actions.append(f'{s}RECV_B0')
                actions.append(f'{s}B0')
            for s in stages:
                actions.extend([f'{s}REDUCE_GRAD', f'{s}RESHARD'])
            schedule['actions'][str(rank)] = actions
        result = run(schedule, self.muon, self.residual, self.adam, communication_gib=0)
        self.assertFalse(result['full_model_fit_verified'])
        for rank in result['ranks']:
            cats = rank['after_schedule_live_by_category']
            for name in ('fsdp_gather_staging', 'fsdp_unsharded_weights', 'checkpoint_inputs',
                         'pipeline_receive', 'pipeline_output', 'fp32_accumulated_gradients', 'attention_residual'):
                self.assertEqual(cats.get(name, 0), 0, name)
            self.assertGreater(cats['muon_reserved_buffers'], 0)
            self.assertEqual(cats['bf16_optimizer_gradients'], cats['sharded_parameters'])
            self.assertGreater(rank['phase_peaks']['optimizer']['categories']['muon_kernel_temporary'], 0)
            self.assertEqual(rank['live_plus_communication_bytes'], rank['modeled_live_peak']['bytes'])

    def test_trace_phase_filtering_and_no_sum_of_independent_peaks(self):
        ledger = Ledger(True)
        ledger.alloc('base', 100, 'state')
        events = [dict(op='alloc',storage_id='input',bytes=100,phase='inputs'),
                  dict(op='alloc',storage_id='tmp',bytes=40,phase='forward'),
                  dict(op='free',storage_id='tmp',phase='forward'),
                  dict(op='alloc',storage_id='grad',bytes=60,phase='backward')]
        ledger.trace(events, 'a', phases={'forward'})
        self.assertEqual(ledger.peak['bytes'], 140)
        ledger.trace(events, 'b')
        self.assertEqual(ledger.peak['bytes'], 160)
        self.assertEqual(list(ledger.live), ['base'])

    def test_compressed_reports_round_trip_and_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'a.json.gz'; b=Path(directory)/'b.json.gz'
            data={'bytes': 2**40, 'trace': ['alloc','free']}
            write_report(a,data); write_report(b,data)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(read_report(a), data)


if __name__=='__main__': unittest.main()
