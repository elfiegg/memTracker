import gc
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from memtracker_nccl.allocator_model import AllocatorConfig
from memtracker_nccl.tracker import ExtendedMemTracker

MiB = 2**20


class AllocatorTrackerTests(unittest.TestCase):
    def test_storage_alias_and_reserved_external_peak(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            tracker = ExtendedMemTracker(allocator_config=AllocatorConfig())
            with tracker:
                x = torch.empty(100, dtype=torch.uint8)
                view = x.view(10, 10)
                self.assertEqual(tracker.report()['allocator']['current']['cpu']['allocated_bytes'], 512)
                del x, view
                gc.collect()
                tracker.external_alloc('nccl', 80)
                report = tracker.report()
                self.assertEqual(report['current']['cpu']['combined_bytes'], 80)
                self.assertEqual(report['allocator']['current']['cpu']['requested_bytes'], 0)
                self.assertEqual(report['resident']['peak']['cpu']['combined_bytes'], 2*MiB+80)
                tracker.empty_allocator_cache()
                self.assertEqual(tracker.report()['resident']['current']['cpu']['combined_bytes'], 80)

    def test_explicit_pending_stream_and_pool(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            tracker = ExtendedMemTracker(allocator_config=AllocatorConfig())
            with tracker:
                with tracker.allocation_scope(pool='communication', stream='comm'):
                    x = torch.empty(100, dtype=torch.uint8)
                tracker.record_tensor_stream(x, 'collective-1')
                del x
                gc.collect()
                tracker.empty_allocator_cache()
                stats = tracker.report()['allocator']['current']['cpu']
                self.assertEqual(stats['pending_free_bytes'], 512)
                self.assertEqual(stats['reserved_bytes'], 2*MiB)
                tracker.complete_stream('collective-1')
                tracker.empty_allocator_cache()
                self.assertEqual(tracker.report()['allocator']['current']['cpu']['reserved_bytes'], 0)

    def test_track_external_and_resize(self):
        # Eager CPU storage resize drives the same private lifetime hook.
        tracker = ExtendedMemTracker(allocator_config=AllocatorConfig())
        x = torch.empty(100, dtype=torch.uint8)
        tracker.track_external(x)
        with tracker:
            x.resize_(3*MiB)
        stats = tracker.report()['allocator']['current']['cpu']
        self.assertEqual(stats['requested_bytes'], 3*MiB)
        self.assertEqual(stats['allocated_bytes'], 3*MiB)
        del x
        gc.collect()
        self.assertEqual(tracker.report()['allocator']['current']['cpu']['allocated_bytes'], 0)

    def test_peak_samples_external_lifetime_not_sum_of_maxima(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            tracker = ExtendedMemTracker(allocator_config=AllocatorConfig())
            with tracker:
                x = torch.empty(25*MiB, dtype=torch.uint8)
                del x
                gc.collect()
                tracker.empty_allocator_cache()
                tracker.external_alloc('other', 10*MiB, category='CUDA context')
            self.assertEqual(tracker.report()['resident']['peak']['cpu']['combined_bytes'], 26*MiB)

    def test_training_reclassification_and_cpu_aliases_match_live_storage(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            model = torch.nn.Linear(8, 8)
            optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
            x = torch.empty(4, 8)
            tracker = ExtendedMemTracker(allocator_config=AllocatorConfig())
            tracker.track_external(model, optimizer, x)
            with tracker:
                for _ in range(2):
                    tracker.reset_mod_stats()
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(x).square().mean()
                    loss.backward()
                    optimizer.step()
                    del loss
                zero = torch.empty(0)
                aliased = torch.empty(13, device="cpu:7")
                report = tracker.report()
                self.assertEqual(set(report["allocator"]["current"]), {"cpu"})
                self.assertEqual(report["allocator"]["current"]["cpu"]["requested_bytes"],
                                 report["current"]["cpu"]["tensor_bytes"])
                del zero, aliased

    def test_no_allocator_report_without_opt_in(self):
        self.assertNotIn('allocator', ExtendedMemTracker().report())
        with self.assertRaises(RuntimeError):
            ExtendedMemTracker().empty_allocator_cache()
