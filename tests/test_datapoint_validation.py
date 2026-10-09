import copy
from pathlib import Path
import unittest

from memtracker_nccl.datapoint_validation import (
    ACTIVE, RESERVED, audit, estimator_configuration, failure_accounting, normalize_dataset, step_windows, _warmup,
)
from memtracker_nccl.report_io import read_report

ROOT=Path(__file__).resolve().parents[1]


class HistoricalValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=read_report(ROOT/'experiments/gb200_historical_observations.json')
        cls.prediction=read_report(ROOT/'experiments/k3_training_lifetimes_expandable.json')

    def test_dataset_cannot_be_reported_as_nine_correct_predictions(self):
        result=audit(self.data,self.prediction)
        self.assertEqual(result['counts']['outcomes'], {'OOM':6,'PASS':3})
        self.assertEqual(result['counts']['known_configuration_matches'],0)
        self.assertEqual(result['counts']['scored_peak_comparisons'],0)
        self.assertEqual(result['counts']['scored_fit_classifications'],0)
        self.assertIsNone(result['accuracy']['false_fit_rate'])
        self.assertFalse(result['accuracy']['success'])
        self.assertFalse(result['estimator_tuned'])
        self.assertTrue(all(r['peak_error_percent'] is None for r in result['experiments']))

    def test_failed_allocation_is_not_added_and_reserved_not_double_counted(self):
        e=self.data['experiments'][0]
        before=copy.deepcopy(e['failure_point_memory_gib'])
        result=failure_accounting(before)
        self.assertAlmostEqual(result['torch_reserved_gib'],156.0)
        self.assertAlmostEqual(result['process_outside_allocator_gib'],24.79)
        self.assertIsNone(result['completed_iteration_peak_gib'])
        self.assertFalse(result['requested_allocation_included_in_allocated'])
        self.assertEqual(before['torch_allocated_gib'],154.26)
        self.assertEqual(before['requested_gib'],1.96)

    def test_nccl_oom_unknowns_stay_null(self):
        e=next(e for e in self.data['experiments'] if e['job']==3240283)
        r=failure_accounting(e['failure_point_memory_gib'])
        for k in ('torch_reserved_gib','process_outside_allocator_gib','device_used_gib','completed_iteration_peak_gib'):
            self.assertIsNone(r[k])

    def test_preserve_rank_and_window_not_sum_of_independent_peaks(self):
        rows={'0':{ACTIVE:[1,5,2],RESERVED:[8,8,8]},'1':{ACTIVE:[3,2,4],RESERVED:[7,9,8]}}
        r=step_windows(rows)
        self.assertEqual(r[ACTIVE]['max_logged_window_gib'],5)
        self.assertEqual(r[ACTIVE]['max_rank'],'0')
        self.assertEqual(r[ACTIVE]['max_window_index'],1)
        self.assertEqual(r[ACTIVE]['first_logged_window_max_gib'],3)
        self.assertEqual(r[ACTIVE]['last_logged_window_max_gib'],4)
        self.assertEqual(r[RESERVED]['max_logged_window_gib'],9)
        self.assertEqual(r[ACTIVE]['observations'],6)

    def test_graph_and_accumulation_mismatches_are_explicit(self):
        r=audit(self.data,self.prediction)
        e=next(e for e in r['experiments'] if e['job']==3250222)
        diff={x['field']:x for x in e['mismatches']}
        self.assertEqual(diff['accumulation_rounds']['measured'],16)
        self.assertEqual(diff['accumulation_rounds']['estimator'],1)
        self.assertEqual(diff['vision_encoder']['measured'],False)
        self.assertIn('graph_warmup_capture_replay_windows_not_modeled',e['blockers'])
        self.assertIn('censored_failure_no_completed_peak',e['blockers'])

    def test_warmup_external_change_with_no_tensor_change(self):
        warm=[w for e in self.data['experiments'] for w in e['warmup_observations']]
        self.assertEqual(len(warm),16)
        self.assertEqual({w['external_delta_bytes'] for w in warm},{0,432*2**20})
        self.assertTrue(all(w['allocated_delta_bytes']==0 for w in warm))
        self.assertTrue(all(w['reserved_delta_bytes']==0 for w in warm))

    def test_bad_warmup_accounting_is_rejected(self):
        entry={'file':'rank-0.log','line':1,'text':"DistMuon communication warmup complete: before={'total':100,'free':50,'reserved':10,'allocated':5,'device_used_minus_reserved':99} after={'total':100,'free':40,'reserved':10,'allocated':5,'device_used_minus_reserved':50}"}
        with self.assertRaises(ValueError): _warmup({'log_evidence':[entry]})

    def test_prediction_replay_must_match(self):
        r=audit(self.data,self.prediction,replayed_prediction=copy.deepcopy(self.prediction))
        self.assertTrue(r['frozen_prediction_reproduced'])
        changed=copy.deepcopy(self.prediction)
        changed['communication_gib']=24
        with self.assertRaises(ValueError): audit(self.data,self.prediction,replayed_prediction=changed)

    def test_duplicate_recipes_and_known_field_match_do_not_inflate_success(self):
        result=audit(self.data,self.prediction)
        self.assertIn([3233459,3238922], result['jobs_sharing_known_configuration'])
        matched=copy.deepcopy(self.data)
        matched['experiments'][0]['configuration']=estimator_configuration(self.prediction)
        r=audit(matched,self.prediction)
        self.assertEqual(r['counts']['known_configuration_matches'],1)
        self.assertEqual(r['counts']['scored_peak_comparisons'],0)
        self.assertFalse(r['accuracy']['success'])

    def test_no_mutation_of_evidence_or_predictions(self):
        before=copy.deepcopy((self.data,self.prediction))
        audit(self.data,self.prediction)
        self.assertEqual((self.data,self.prediction),before)

    def test_supplied_runtime_identity_does_not_imply_memory_accuracy(self):
        result = audit(self.data, self.prediction)
        for row in result['experiments']:
            metadata = row['run_metadata']['values']
            self.assertEqual(metadata['nccl_build'], '2.30.7+cuda13.3')
            self.assertIsNone(metadata['training_code_commit'])
            self.assertEqual(row['runtime_compatibility']['status'], 'incomplete')
            self.assertIn('pytorch_version', row['runtime_compatibility']['matching_fields'])
            self.assertIn('source_runtime_identity_incomplete', row['blockers'])
            self.assertIsNone(row['peak_error_percent'])

    def test_raw_summary_inconsistency_rejected(self):
        e=copy.deepcopy(self.data['experiments'][0])
        e['configuration']={'parallelism':{'a':1},'training':{}}
        e['effective_config']={'parallelism':{'a':2},'training':{}}
        e['log_evidence']=[]
        with self.assertRaises(ValueError):
            normalize_dataset({'schema_version':1,'interpretation':[],'experiments':[e]})


if __name__=='__main__': unittest.main()
