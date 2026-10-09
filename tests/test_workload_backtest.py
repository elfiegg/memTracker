import copy
from pathlib import Path
import unittest

from memtracker_nccl.datapoint_validation import ACTIVE, RESERVED
from memtracker_nccl.k3_interleaved import parse_action
from memtracker_nccl.k3_training_lifetimes import Ledger, run as replay
from memtracker_nccl.recipe_schedule import build_schedule
from memtracker_nccl.report_io import read_report
from memtracker_nccl.workload_backtest import compare_run, prepare_recipes, attach_run_metadata, validate_recipe_record
from memtracker_nccl.run_metadata import modeled_metadata, resolve_metadata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'k3-fit/source'
SCHEDULES = ROOT.parent/'k3-fit/gpu-pipeline-schedules.py'
GiB = 2**30


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipes=read_report(ROOT/'experiments/gb200_workload_recipes.json.gz')['experiments']
        cls.observations=read_report(ROOT/'experiments/gb200_historical_observations.json')['experiments']

    def test_rounded_active_bytes_are_distinct_from_requested_and_reserved(self):
        ledger=Ledger(True)
        ledger.alloc('a',1,'parameter')
        ledger.alloc('b',513,'temporary')
        self.assertEqual(ledger.active_bytes,1536)
        self.assertEqual(ledger.peak['bytes'],514)
        self.assertEqual(ledger.active_peak['bytes'],1536)
        ledger.free('b'); ledger.free('a')
        self.assertEqual(ledger.active_bytes,0)
        self.assertGreater(ledger.reserved_peak['bytes'],1536)

    @unittest.skipUnless(SCHEDULES.exists() and SOURCE.exists(), 'Trusted reference sources unavailable')
    def test_source_schedules_cover_every_microbatch_exactly_once(self):
        for index in (0,4,5,6,7):
            c=self.recipes[index]['effective_config']
            schedule=build_schedule(c,SOURCE,SCHEDULES)
            expected={(s,mb) for s in range(schedule['virtual_stages']) for mb in range(schedule['microbatches'])}
            seen={op:[] for op in ('F','B')}
            for rank, actions in schedule['actions'].items():
                active=set()
                for action in actions:
                    st,op,mb=parse_action(action)
                    self.assertEqual(st%schedule['pp'],int(rank))
                    if op in seen: seen[op].append((st,mb))
                    if op=='F': active.add((st,mb))
                    if op=='B': active.remove((st,mb))
                self.assertFalse(active)
            for op,values in seen.items():
                self.assertEqual(set(values),expected,op)
                self.assertEqual(len(values),len(expected),op)
            self.assertEqual(sum(s['last_layer']-s['first_layer']+1 for s in schedule['stages']),len(c['model']['layers']))
            if schedule['schedule']=='1F1B':
                self.assertIn('7F0',schedule['actions']['7'])

    def test_pass_diagnostics_compare_same_rank_and_do_not_claim_accuracy(self):
        e=copy.deepcopy(self.observations[6])
        e['per_rank_step_memory_metrics']={'0':{ACTIVE:[.25,.1],RESERVED:[.5,.4]}}
        rank=dict(global_rank=0,pp_rank=0,dp_owner=0,modeled_active_peak={'bytes':GiB//8,'event':'B0'},
                  modeled_reserved_peak={'bytes':GiB//4})
        result=compare_run(e,[{'ranks':[rank]}],[{'ranks':[rank]}])
        d=result['successful_window_diagnostics'][0]
        self.assertEqual(d['active_gap_gib'],-.125)
        self.assertEqual(d['reserved_gap_gib'],-.25)
        self.assertIsNone(d['accuracy_error_percent'])
        self.assertIsNone(result['external_memory_prediction_gib'])
        self.assertFalse(result['full_peak_accuracy_verified'])

    def test_oom_is_censored_and_has_no_success_window_error(self):
        e=self.observations[0]
        rank=dict(global_rank=224,pp_rank=7,dp_owner=0,phase_peaks={'training':{'bytes':100*GiB}})
        result=compare_run(e,[{'ranks':[rank]}],[])
        self.assertFalse(result['successful_window_diagnostics'])
        f=result['oom_constraint']
        self.assertEqual(f['failure_point']['torch_allocated_gib'],154.26)
        self.assertIsNone(f['fit_classification_correct'])
        self.assertFalse(f['failure_accounting']['requested_allocation_included_in_allocated'])

    def test_recipe_extraction_does_not_embed_memory_targets(self):
        effective=self.recipes[6]['effective_config']
        raw={'experiments':[{'job':1,'effective_config':effective,'observed_memory':999999}]}
        extracted=prepare_recipes(raw,'hash')
        self.assertNotIn('observed_memory',str(extracted))
        changed=copy.deepcopy(raw); changed['experiments'][0]['observed_memory']=1
        self.assertEqual(prepare_recipes(changed,'hash'),extracted)

    def test_modified_retained_configuration_is_rejected(self):
        record=copy.deepcopy(self.recipes[6])
        validate_recipe_record(record,self.observations[6])
        record['effective_config']['model']['dim']*=2
        with self.assertRaisesRegex(ValueError,'Retained replay config fingerprint'):
            validate_recipe_record(record,self.observations[6])

    def test_cache_metadata_is_recompared_without_mutating_cached_math(self):
        model=modeled_metadata({'source_commit':'b4d5b404bd30dec67f86873203fbd4ba5f457531'})
        prediction={'modeled_metadata':model,'runtime_compatibility':{'status':'old'},'peak':123}
        target=resolve_metadata(layers=[('run',{'pytorch':'different'})])
        result=attach_run_metadata(prediction,target)
        self.assertEqual(result['runtime_compatibility']['status'],'mismatch')
        self.assertEqual(prediction['runtime_compatibility']['status'],'old')
        self.assertEqual(result['peak'],123)

    def test_first_step_states_are_lazy_and_accumulation_reuses_local_gradients(self):
        report_path=ROOT/'experiments/gb200_workload_backtest.json'
        if not report_path.exists(): self.skipTest('Generated replay evidence unavailable')
        report=read_report(report_path)
        row=next(e for e in report['experiments'] if e['job']==3244745)
        evidence=ROOT/'experiments/gb200_replay_evidence'
        opt=read_report(evidence/row['optimizer_evidence'])
        schedule=read_report(evidence/'schedule-3244745.json.gz')
        residual=read_report(evidence/'residual-128-256-5.json.gz')
        c=copy.deepcopy(self.recipes[6]['effective_config'])
        c['training']['num_tokens_per_train_step']*=2
        result=replay(schedule,opt['muon'],residual,opt['adam'],recipe=c,aggregate=opt['aggregate'],
            sequence=128,microbatch_size=1,communication_gib=0,initialized_optimizer=False,dp_owner=0,pp_ranks=[0])
        self.assertEqual(result['scenario']['accumulation_rounds'],2)
        rank=result['ranks'][0]
        self.assertEqual(rank['phase_peaks']['training']['categories'].get('muon_momentum',0),0)
        self.assertEqual(rank['phase_peaks']['training']['categories'].get('adamw_moments',0),0)
        self.assertGreater(rank['phase_peaks']['optimizer']['categories']['muon_momentum'],0)
        final=rank['after_schedule_live_by_category']
        self.assertEqual(final['bf16_optimizer_gradients'],final['sharded_parameters'])
        self.assertEqual(final.get('fp32_accumulated_gradients',0),0)


if __name__=='__main__': unittest.main()
