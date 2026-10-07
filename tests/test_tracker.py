import gc
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from memtracker_nccl.tracker import ExtendedMemTracker


class TrackerTests(unittest.TestCase):
    def test_peak_is_simultaneous_not_sum_of_peaks(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            tracker = ExtendedMemTracker()
            with tracker:
                x = torch.empty(100, dtype=torch.uint8)
                del x
                gc.collect()
                tracker.external_alloc("nccl", 80)
                y = torch.empty(10, dtype=torch.uint8)
                report = tracker.report()
                self.assertEqual(report["peak"]["cpu"]["combined_bytes"], 100)
                self.assertEqual(report["current"]["cpu"]["combined_bytes"], 90)
                self.assertEqual(report["tensor_peak_bytes"]["cpu"], 100)
                self.assertEqual(report["external_peak_bytes"]["cpu"], 80)
                tracker.external_free("nccl")
                del y

    def test_tensor_peak_while_external_allocation_live(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            tracker = ExtendedMemTracker()
            tracker.external_alloc("nccl", 80)
            with tracker:
                x = torch.empty(100, dtype=torch.uint8)
                view = x.view(10, 10)
                self.assertEqual(tracker.report()["peak"]["cpu"]["combined_bytes"], 180)
                del x
                self.assertEqual(tracker.report()["current"]["cpu"]["tensor_bytes"], 100)
                del view
                gc.collect()
                self.assertEqual(tracker.report()["current"]["cpu"]["tensor_bytes"], 0)
            tracker.external_free("nccl")

    def test_cpu_device_aliases_share_tensor_accounting(self):
        for device in ("cpu:0", "cpu:7", torch.device("cpu:0")):
            with self.subTest(device=device), FakeTensorMode(allow_fallback_kernels=False):
                tracker = ExtendedMemTracker()
                with tracker:
                    x = torch.empty(100, dtype=torch.uint8)
                    aliased = torch.empty(25, dtype=torch.uint8, device=device)
                    tracker.external_alloc("nccl", 50, device=device)
                    report = tracker.report()
                    self.assertEqual(set(report["current"]), {"cpu"})
                    self.assertEqual(report["current"]["cpu"]["combined_bytes"], 175)
                    self.assertEqual(report["peak"]["cpu"]["combined_bytes"], 175)
                    self.assertEqual(report["live_external_allocations"]["nccl"]["device"], "cpu")
                    self.assertEqual(report["events"][-1]["allocation"]["device"], "cpu")
                    tracker.external_free("nccl")
                    report = tracker.report()
                    self.assertEqual(report["current"]["cpu"]["combined_bytes"], 125)
                    self.assertEqual(report["events"][-1]["allocation"]["device"], "cpu")
                    del x, aliased
                    gc.collect()
                    self.assertEqual(tracker.report()["current"], {})
                    self.assertEqual(tracker.report()["peak"]["cpu"]["combined_bytes"], 175)

    def test_external_validation_and_device_separation(self):
        tracker = ExtendedMemTracker()
        tracker.external_alloc("a", 10, device="cpu")
        tracker.external_alloc("b", 20, device="cuda:0")
        self.assertEqual(tracker.report()["peak"]["cpu"]["combined_bytes"], 10)
        self.assertEqual(tracker.report()["peak"]["cuda:0"]["combined_bytes"], 20)
        with self.assertRaises(ValueError):
            tracker.external_alloc("a", 2)
        for invalid in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                tracker.external_alloc("bad", invalid)
        tracker.external_free("a")
        with self.assertRaises(KeyError):
            tracker.external_free("a")

    def test_training_backward_and_optimizer_states(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            model = torch.nn.Linear(8, 8)
            optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
            x = torch.empty(4, 8)
            tracker = ExtendedMemTracker()
            tracker.track_external(model, optimizer, x)
            tracker.external_alloc("persistent", 128)
            with tracker:
                for _ in range(2):
                    tracker.reset_mod_stats()
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(x).square().mean()
                    loss.backward()
                    optimizer.step()
                    del loss
            report = tracker.report()
            self.assertGreater(report["peak"]["cpu"]["tensor_bytes"], 8 * 8 * 4)
            self.assertEqual(report["peak"]["cpu"]["external_bytes"], 128)
            tracker.external_free("persistent")


class NestedTrackerTests(unittest.TestCase):
    def test_nested_tracker_scope_keeps_external_and_tensor_lifetimes(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            tracker = ExtendedMemTracker()
            tracker.external_alloc("persistent", 50)
            with tracker:
                x = torch.empty(100, dtype=torch.uint8)
                with tracker:
                    y = torch.empty(200, dtype=torch.uint8)
                del y
                gc.collect()
                self.assertEqual(tracker.report()["current"]["cpu"]["combined_bytes"], 150)
            self.assertEqual(tracker.report()["peak"]["cpu"]["combined_bytes"], 350)
            tracker.reset_mod_stats()
            # reset_mod_stats only clears module attribution, not session peaks.
            self.assertEqual(tracker.report()["peak"]["cpu"]["combined_bytes"], 350)
            del x
            gc.collect()
            self.assertEqual(tracker.report()["current"]["cpu"]["combined_bytes"], 50)
            tracker.external_free("persistent")
            self.assertEqual(tracker.report()["current"], {})


if __name__ == "__main__":
    unittest.main()
