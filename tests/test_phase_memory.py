import copy
import unittest
from pathlib import Path
import runpy
from memtracker_nccl.report_io import read_report

from memtracker_nccl import phase_memory as pm


def fixture():
    return dict(schema_version=1, device=0, history_complete=True, baseline_segments=[],
        events=[
            dict(action='segment_alloc', addr=1000, size=4096, stream=0, pool_id=[0, 0]),
            dict(action='alloc', addr=1000, size=1024, stream=0, frames=[dict(filename='model.py', name='initialize', line=1)]),
            dict(action='alloc', addr=2024, size=512, stream=1),
            dict(action='free_requested', addr=2024, size=512, stream=1),
            dict(action='free_completed', addr=2024, size=512, stream=1),
            dict(action='free_requested', addr=1000, size=1024, stream=0),
            dict(action='free_completed', addr=1000, size=1024, stream=0),
            dict(action='segment_free', addr=1000, size=4096, stream=0),
        ], spans=[dict(id=0, name='initialize', start=0, end=2),
                  dict(id=1, name='train', start=2, end=8),
                  dict(id=2, name='backward', start=2, end=4)], checkpoints=[])


class PhaseMemoryTests(unittest.TestCase):
    def test_api_exists(self):
        self.assertTrue(callable(getattr(pm, 'analyze_capture', None)))

    def test_transient_peak_carries_initialization_and_pending_frees(self):
        result = pm.analyze_capture(fixture())
        train = result['phases'][1]
        self.assertEqual(train['peaks']['active_bytes']['bytes'], 1536)
        self.assertEqual(train['peaks']['pending_free_bytes']['bytes'], 1024)
        self.assertEqual(train['peaks']['reserved_bytes']['bytes'], 4096)
        peak = train['peaks']['active_bytes']
        self.assertEqual(peak['active_by_origin_phase'], {'initialize': 1024, 'backward': 512})
        self.assertEqual(result['final']['reserved_bytes'], 0)
        self.assertEqual(result['phases'][2]['end']['pending_free_bytes'], 512)
        self.assertEqual(result['phases'][0]['end']['active_bytes'], 1024)

    def test_peak_includes_pending_free_even_after_tensor_release(self):
        c = fixture()
        c['events'].insert(4, dict(action='alloc', addr=2536, size=512, stream=0))
        c['spans'] = [dict(id=0, name='phase', start=0, end=5)]
        c['events'] = c['events'][:5]
        r = pm.analyze_capture(c)['phases'][0]['peaks']['active_bytes']
        self.assertEqual(r['bytes'], 2048)
        self.assertEqual(r['allocated_bytes'], 1536)
        self.assertEqual(r['pending_free_bytes'], 512)

    def test_expandable_mapping_and_graph_pools_are_not_double_counted(self):
        c = fixture()
        c['events'][0].update(action='segment_map', pool_id=[3, 9])
        c['events'][-1]['action'] = 'segment_unmap'
        r = pm.analyze_capture(c)
        self.assertEqual(r['phases'][0]['peaks']['reserved_bytes']['bytes'], 4096)
        self.assertEqual(r['final']['reserved_bytes'], 0)
        self.assertEqual(r['phases'][0]['peaks']['reserved_bytes']['reserved_by_pool'], {'3:9': 4096})

    def test_oom_requests_are_not_live_allocations(self):
        c = fixture()
        c['events'] = c['events'][:2]+[dict(action='oom', size=999999, device_free=10)]
        c['spans'] = [dict(id=0, name='optimizer', start=0, end=3, status='error')]
        r = pm.analyze_capture(c)
        self.assertEqual(r['final']['allocated_bytes'], 1024)
        self.assertEqual(r['oom_events'][0]['requested_bytes'], 999999)
        self.assertFalse(r['completed_successfully'])

    def test_bad_or_truncated_history_cannot_produce_valid_peaks(self):
        for mutation in ('truncated', 'missing_free', 'unknown_action'):
            c = fixture()
            if mutation == 'truncated': c['history_complete'] = False
            elif mutation == 'missing_free': c['events'][2]['addr'] = 99
            else: c['events'][2]['action'] = 'new_unsupported_action'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                pm.analyze_capture(c)

    def test_same_time_external_measurement_not_added_to_phase_peak(self):
        c = fixture()
        c['checkpoints'] = [dict(trace_index=2, allocated_bytes=1024, active_bytes=1024,
            reserved_bytes=4096, device_used_bytes=10000, process_used_bytes=None)]
        r = pm.analyze_capture(c)
        self.assertEqual(r['checkpoints'][0]['device_outside_allocator_bytes'], 5904)
        self.assertIsNone(r['checkpoints'][0]['process_outside_allocator_bytes'])
        self.assertNotIn('combined_peak_bytes', r)

    def test_baseline_live_storage_is_counted_but_not_assigned_to_new_phase(self):
        c = fixture()
        c['baseline_segments'] = [dict(device=0, address=8000, total_size=2048,
            segment_pool_id=[0,0], blocks=[dict(address=8000, size=512, state='active_allocated'),
            dict(address=8512, size=1536, state='inactive')])]
        r = pm.analyze_capture(c)
        self.assertEqual(r['final']['active_bytes'], 512)
        self.assertEqual(r['final']['reserved_bytes'], 2048)

    def test_owner_inventory_partitions_live_bytes_without_claiming_untracked_activations(self):
        c = fixture()
        c['checkpoints'] = [dict(trace_index=3, allocated_bytes=1536, active_bytes=1536, reserved_bytes=4096,
            tensor_owners={'storages':[dict(device='cuda:0',address=1000,storage_bytes=1000,category='parameters')]})]
        owner = pm.analyze_capture(c)['checkpoints'][0]['ownership']
        self.assertEqual(owner['storage_bytes_by_category'], {'parameters':1000})
        self.assertEqual(owner['allocator_padding_bytes'],24)
        self.assertEqual(owner['unattributed_allocated_bytes'],512)

    def test_phase_comparison_uses_active_not_tensor_requested_bytes_and_keeps_oom_censored(self):
        measured = pm.analyze_capture(fixture())
        estimated = {'phases':[dict(name='train',devices={'cpu':dict(
            tensor_peak_bytes=999, active_peak_bytes=1024, allocated_peak_bytes=1000, reserved_peak_bytes=2048)})]}
        result = pm.compare_phase_estimate(measured, estimated)
        row = next(r for r in result['comparisons'] if r['name']=='train')
        self.assertEqual(row['gaps_bytes']['active_bytes'], 512)
        self.assertEqual(row['gaps_bytes']['reserved_bytes'], 2048)
        self.assertFalse(result['accuracy_verified'])
        measured['phases'][1]['status'] = 'error'
        row = next(r for r in pm.compare_phase_estimate(measured, estimated)['comparisons'] if r['name']=='train')
        self.assertIsNone(row['gaps_bytes'])

    def test_counter_mismatch_blocks_scoring(self):
        c = fixture()
        c['checkpoints'] = [dict(trace_index=2,active_bytes=10000)]
        measured = pm.analyze_capture(c)
        self.assertFalse(measured['counters_reconciled'])
        result = pm.compare_phase_estimate(measured, {'phases':[]})
        self.assertEqual(result['status'], 'counter_mismatch')

    def test_requested_trace_sizes_use_snapshot_block_sizes_when_available(self):
        c = fixture()
        c['trace_size_semantics'] = 'requested'
        c['events'] = [dict(action='segment_alloc',addr=100,size=2048),dict(action='alloc',addr=100,size=513)]
        c['spans'] = [dict(id=0,name='init',start=0,end=2)]
        c['checkpoints'] = [dict(trace_index=2,allocated_bytes=1024,active_bytes=1024,reserved_bytes=2048,
            block_sizes=[dict(address=100,size=1024)])]
        r = pm.analyze_capture(c)
        self.assertEqual(r['phases'][0]['peaks']['active_bytes']['bytes'],1024)
        self.assertEqual(r['phases'][0]['peaks']['requested_active_bytes']['bytes'],513)
        self.assertTrue(r['counters_reconciled'])

    def test_transient_requested_size_is_not_claimed_as_exact_block_peak(self):
        c = fixture()
        c['trace_size_semantics'] = 'requested'
        c['events'] = [dict(action='segment_alloc',addr=100,size=2048),dict(action='alloc',addr=100,size=513),
            dict(action='free_requested',addr=100,size=513),dict(action='free_completed',addr=100,size=513)]
        c['spans'] = [dict(id=0,name='forward',start=0,end=4)]
        r = pm.analyze_capture(c)
        self.assertIsNone(r['phases'][0]['peaks']['active_bytes']['bytes'])
        self.assertEqual(r['phases'][0]['peaks']['active_bytes']['lower_bound_bytes'],513)
        self.assertEqual(r['phases'][0]['peaks']['requested_active_bytes']['bytes'],513)
        self.assertEqual(r['phases'][0]['peaks']['reserved_bytes']['bytes'],2048)

    def test_external_delta_can_change_without_any_allocator_event(self):
        c = fixture()
        c['events'] = []
        c['spans'] = [dict(id=0,name='communication',start=0,end=0,start_checkpoint=0,end_checkpoint=1)]
        c['checkpoints'] = [dict(trace_index=0,reserved_bytes=0,device_used_bytes=100),
                            dict(trace_index=0,reserved_bytes=0,device_used_bytes=532)]
        r = pm.analyze_capture(c)
        self.assertEqual(r['phases'][0]['external_boundary_delta_bytes']['device'],432)

    def test_isolated_peak_counters_supply_physical_peak_without_false_stack_attribution(self):
        c = fixture()
        c['trace_size_semantics'] = 'requested'
        c['isolated_peak_counters'] = True
        c['events'][1]['size'] = 513
        c['spans'] = [dict(id=0,name='forward',start=0,end=8,start_checkpoint=0,end_checkpoint=1)]
        c['checkpoints'] = [dict(trace_index=0), dict(trace_index=8,interval_peaks={
            'allocated_bytes':2048,'active_bytes':2560,'reserved_bytes':4096})]
        r = pm.analyze_capture(c)
        peak = r['phases'][0]['peaks']['active_bytes']
        self.assertEqual(peak['bytes'],2560)
        self.assertIsNone(peak['active_by_stack'])
        self.assertEqual(peak['source'],'isolated_allocator_peak_counters')

    def test_size_hints_follow_allocation_generations_not_reused_addresses(self):
        c = fixture()
        c['trace_size_semantics'] = 'requested'
        c['events'] = [dict(action='segment_alloc',addr=100,size=4096),dict(action='alloc',addr=100,size=100),
            dict(action='free_requested',addr=100,size=100),dict(action='free_completed',addr=100,size=100),
            dict(action='alloc',addr=100,size=513)]
        c['spans'] = [dict(id=0,name='phase',start=0,end=5)]
        c['checkpoints'] = [dict(trace_index=2,block_sizes=[dict(address=100,size=512)]),
            dict(trace_index=5,block_sizes=[dict(address=100,size=1024)])]
        r = pm.analyze_capture(c)
        self.assertEqual(r['final']['allocated_bytes'],1024)
        self.assertEqual(r['phases'][0]['peaks']['active_bytes']['bytes'],1024)

    def test_real_gb200_snapshot_accounting(self):
        root = Path(__file__).resolve().parents[1]
        validate = runpy.run_path(str(root/'scripts/validate_framework_snapshots.py'))['validate']
        r = validate(read_report(root/'experiments/framework_snapshot_fixtures.json.gz'))
        self.assertEqual(r['snapshots'],56)
        self.assertTrue(r['all_counters_reconciled'])
        self.assertEqual(r['maximum_absolute_difference_bytes'],0)


if __name__ == '__main__':
    unittest.main()
