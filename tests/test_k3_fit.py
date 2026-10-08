import unittest
from memtracker_nccl.k3_fit_experiment import run


class K3FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run(receive_policy='eager')

    def test_native_partition_and_expert_only_bound(self):
        result = self.result
        self.assertEqual([(s['first_layer'],s['last_layer']) for s in result['stages']],
            [(0,10),(11,22),(23,34),(35,46),(47,58),(59,70),(71,82),(83,92)])
        row=result['stages'][6]
        expert_elements=12*28*3*3584*3072
        self.assertEqual(row['local_expert_parameters'], expert_elements)
        self.assertEqual(row['receive_and_input_floor_bytes'], 64*8136*7168*2*(7+8))
        self.assertEqual(row['expert_only_steady_floor_bytes'], row['receive_and_input_floor_bytes']+8*expert_elements)
        self.assertGreater(row['expert_only_steady_floor_bytes'], result['capacity_bytes'])
        self.assertEqual(result['verdict'],'does_not_fit_under_stated_implementation')
        self.assertFalse(result['coverage']['full_model_forward_backward_executed'])

    def test_mb_count_changes_buffers_even_when_live_pipeline_depth_is_fixed(self):
        smaller=run(microbatches=32,receive_policy='eager')
        self.assertEqual(smaller['stages'][6]['receive_and_input_floor_bytes']*2,
                         self.result['stages'][6]['receive_and_input_floor_bytes'])
        self.assertEqual(smaller['verdict'],'fit_unproven')

    def test_half_mb_double_count_preserves_receive_footprint(self):
        altered=run(microbatch_size=1,microbatches=128,receive_policy='eager')
        self.assertEqual(altered['stages'][6]['receive_and_input_floor_bytes'],self.result['stages'][6]['receive_and_input_floor_bytes'])

    def test_lazy_runtime_cannot_inherit_eager_no_fit_verdict(self):
        result=run(receive_policy='lazy')
        self.assertEqual(result['verdict'],'fit_unproven')
        self.assertEqual(result['stages'][6]['expert_only_steady_floor_bytes'], 12*28*3*3584*3072*8)
        row=result['stages'][6]
        self.assertEqual(row['receive_and_input_floor_bytes'], 8136*7168*2*(2*7+8))
        self.assertLess(row['receive_and_input_floor_bytes'], self.result['stages'][6]['receive_and_input_floor_bytes'])

    def test_verdict_uses_combined_decoder_steady_bound(self):
        result=run(receive_policy='lazy', capacity_bytes=100*2**30)
        row=result['stages'][6]
        self.assertLess(row['expert_only_steady_floor_bytes'], result['capacity_bytes'])
        self.assertGreater(row['decoder_steady_state_floor_bytes'], result['capacity_bytes'])
        self.assertEqual(result['verdict'], 'does_not_fit_under_stated_implementation')

    def test_invalid_inputs(self):
        for value in (0,-1,True,1.5):
            with self.assertRaises(ValueError): run(sequence=value)
