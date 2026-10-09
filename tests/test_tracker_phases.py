import gc
import unittest
from pathlib import Path
import runpy
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from memtracker_nccl.allocator_model import AllocatorConfig
from memtracker_nccl.tracker import ExtendedMemTracker


class TrackerPhaseTests(unittest.TestCase):
    def test_phase_api_exists(self):
        self.assertTrue(callable(getattr(ExtendedMemTracker, 'phase', None)))

    def test_phase_peaks_do_not_reset_global_or_drop_carried_state(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            t = ExtendedMemTracker(allocator_config=AllocatorConfig())
            with t:
                with t.phase('initialize'):
                    x = torch.empty(1024, dtype=torch.uint8)
                with t.phase('forward'):
                    y = torch.empty(2048, dtype=torch.uint8)
                    del y
                    gc.collect()
                with t.phase('optimizer'):
                    z = torch.empty(512, dtype=torch.uint8)
                rows = t.report()['phases']
                self.assertEqual(rows[0]['devices']['cpu']['tensor_peak_bytes'],1024)
                self.assertEqual(rows[1]['devices']['cpu']['tensor_peak_bytes'],3072)
                self.assertEqual(rows[2]['devices']['cpu']['tensor_peak_bytes'],1536)
                self.assertEqual(t.report()['tensor_peak_bytes']['cpu'],3072)
                self.assertEqual(rows[1]['devices']['cpu']['active_peak_bytes'],3072)

    def test_two_model_families_run_without_model_specific_inventory(self):
        for family in ('linear', 'convolution'):
            with self.subTest(family=family), FakeTensorMode(allow_fallback_kernels=False):
                t = ExtendedMemTracker(allocator_config=AllocatorConfig())
                with t:
                    with t.phase('model_initialization'):
                        model = torch.nn.Linear(8, 4) if family == 'linear' else torch.nn.Conv2d(2, 3, 3)
                        opt = torch.optim.AdamW(model.parameters(), foreach=False)
                        x = torch.empty(2, 8) if family == 'linear' else torch.empty(2, 2, 8, 8)
                    with t.phase('forward_backward'):
                        loss = model(x).square().mean()
                        loss.backward()
                    with t.phase('optimizer'):
                        opt.step()
                r = t.report()
                self.assertEqual([p['name'] for p in r['phases']], ['model_initialization','forward_backward','optimizer'])
                self.assertGreater(r['phases'][-1]['devices']['cpu']['tensor_peak_bytes'], 0)

    def test_demo_repeated_full_checkpoint_steps_preserve_allocator_history(self):
        demo = runpy.run_path(str(Path(__file__).resolve().parents[1]/'examples/framework_phase_demo.py'))
        for family in ('mlp','conv'):
            r = demo['run'](mode='fake',family=family,width=8,batch=2,steps=3,full_ac=True,expandable=True)
            self.assertEqual(len(r['phases']),10)
            self.assertTrue(all(p['status']=='ok' for p in r['phases']))
            self.assertEqual([p['metadata']['step'] for p in r['phases'] if p['name']=='optimizer'],[1,2,3])


if __name__ == '__main__': unittest.main()
