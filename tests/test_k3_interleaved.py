import copy
import json
from pathlib import Path
import unittest

from memtracker_nccl.k3_interleaved import run, stage_shapes

FIXTURE=Path(__file__).resolve().parents[1]/'experiments/k3_interleaved_schedule.json'


class InterleavedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedule=json.loads(FIXTURE.read_text())
        cls.result=run(cls.schedule)

    def test_real_schedule_counts_and_partition(self):
        self.assertEqual([r['peak_live_pairs'] for r in self.result['ranks']], [23,21,19,17,15,13,11,9])
        layers=[i for s in self.schedule['stages'] for i in range(s['first_layer'],s['last_layer']+1)]
        self.assertEqual(layers,list(range(93)))
        self.assertEqual([s['last_layer']-s['first_layer']+1 for s in self.schedule['stages']], [5]+[6]*14+[4])
        for r in self.result['ranks']:
            self.assertEqual(r['virtual_stages'],[r['rank'],r['rank']+8])
        # Per-stage maxima cannot be added: 16+8 exceeds the simultaneous 23.
        self.assertEqual(sum(self.result['ranks'][0]['peak_live_per_virtual_stage'].values()),24)
        self.assertEqual(self.result['ranks'][0]['peak_live_pairs'],23)

    def test_shared_checkpoint_stacks_count_unique_storage(self):
        h=7168*2
        first=stage_shapes(self.schedule['stages'][0],tokens=1,last_stage=15)
        self.assertEqual(first['checkpoint_input_bytes'],6*h)  # five hiddens + one shared stack
        crossed=stage_shapes(self.schedule['stages'][2],tokens=1,last_stage=15)
        self.assertEqual(crossed['distinct_checkpoint_stack_sizes'],[1,2])
        self.assertEqual(crossed['checkpoint_input_bytes'],9*h)  # six hiddens + old/new stacks
        self.assertEqual(crossed['backward_receive_bytes'],3*h)

    def test_sequence_scaling_changes_activations_not_state(self):
        half=run(self.schedule,sequence=2034)
        for a,b in zip(self.result['stages'],half['stages']):
            self.assertEqual(a['retained_boundary_bytes'],2*b['retained_boundary_bytes'])
            self.assertEqual(a['persistent_parameter_bytes'],b['persistent_parameter_bytes'])
            self.assertEqual(a['fp32_accumulated_gradient_bytes'],b['fp32_accumulated_gradient_bytes'])

    def test_communication_is_resident_and_not_multiplied_by_stage_count(self):
        for r in self.result['ranks']:
            row=r['communication_budget'][2]
            self.assertEqual(row['partial_peak_plus_communication_bytes'],r['partial_peak_bytes']+16*2**30)
            self.assertEqual(row['verdict'],'fit_unproven')
        self.assertFalse(self.result['full_model_fit_verified'])

    def test_incomplete_or_duplicate_schedule_rejected(self):
        for duplicate in (False,True):
            schedule=copy.deepcopy(self.schedule)
            if duplicate:schedule['actions']['0'].append('0F0')
            else:schedule['actions']['0'].remove('0B0')
            with self.assertRaises((ValueError,KeyError)):run(schedule)

    def test_bad_sizes_rejected_before_replay(self):
        for size in (-1,0,True,1.5):
            with self.assertRaises((ValueError,TypeError)):run(self.schedule,sequence=size)
