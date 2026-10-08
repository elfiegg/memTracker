import unittest
from memtracker_nccl.k3_communication_budget import assess, GiB


class CommunicationBudgetTests(unittest.TestCase):
    def screen(self, policy='lazy'):
        return dict(receive_policy=policy, capacity_bytes=100*GiB, stages=[
            dict(stage=0, decoder_steady_state_floor_bytes=60*GiB),
            dict(stage=1, decoder_steady_state_floor_bytes=80*GiB)])

    def test_persistent_communication_is_added_once_to_each_stage(self):
        result=assess(self.screen(), persistent_communication_bytes=8*GiB, provenance='test')
        self.assertEqual(result['remaining_budget_for_omitted_memory_bytes'], 12*GiB)
        self.assertEqual(result['limiting_stage_for_this_bound'], 1)
        self.assertEqual(result['verdict'], 'not_ruled_out_by_partial_budget')
        self.assertFalse(result['full_model_fit_verified'])
        self.assertEqual(result['stages'][0]['state_receive_and_communication_bytes'], 68*GiB)

    def test_exact_threshold_and_one_byte_over(self):
        at_limit=assess(self.screen(), persistent_communication_bytes=20*GiB, provenance='test')
        self.assertEqual(at_limit['remaining_budget_for_omitted_memory_bytes'], 0)
        self.assertFalse(at_limit['full_model_fit_verified'])
        over=assess(self.screen(), persistent_communication_bytes=20*GiB+1, provenance='test')
        self.assertEqual(over['verdict'], 'exceeds_capacity_under_assumptions')
        self.assertEqual(over['remaining_budget_for_omitted_memory_bytes'], -1)

    def test_additional_allowance_and_receive_policy_are_independent(self):
        for policy in ('eager', 'lazy'):
            result=assess(self.screen(policy), persistent_communication_bytes=8*GiB,
                          additional_peak_bytes=13*GiB, provenance='test')
            self.assertEqual(result['receive_policy'], policy)
            self.assertEqual(result['communication_initialization'], 'eager')
            self.assertEqual(result['remaining_after_additional_allowance_bytes'], -GiB)
            self.assertEqual(result['verdict'], 'exceeds_capacity_under_assumptions')

    def test_bad_allowances_and_missing_provenance_are_rejected(self):
        for value in (-1, True, 0.5):
            with self.assertRaises((ValueError, TypeError)):
                assess(self.screen(), persistent_communication_bytes=value, provenance='test')
        with self.assertRaises(ValueError):
            assess(self.screen(), persistent_communication_bytes=0, provenance='')
