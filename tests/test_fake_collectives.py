import gc
import unittest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from memtracker_nccl import ExtendedMemTracker
from memtracker_nccl.fake_collectives import FakeCollectiveRunner


class TestModel:
    def __init__(self, tracker):
        self.tracker = tracker
        self.sequence = 0
        self.active = set()

    def begin_collective(self, communicator, *, operation, payload_bytes, temporary_bytes):
        self.sequence += 1
        token = f"{communicator}:{self.sequence}"
        self.tracker.external_alloc(token, temporary_bytes)
        self.active.add(token)
        return token

    def complete_collective(self, token):
        self.tracker.external_free(token)
        self.active.remove(token)


class FakeCollectiveTests(unittest.TestCase):
    def test_actual_torch_fake_collective_and_overlap(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            tracker = ExtendedMemTracker()
            model = TestModel(tracker)
            runner = FakeCollectiveRunner(model)
            with tracker:
                x = torch.empty(4, dtype=torch.float32)
                first = runner.all_gather(x, "test", 8, temporary_bytes=32)
                second = runner.all_reduce(x, "test", temporary_bytes=64)
                self.assertEqual(first.output.shape, (32,))
                self.assertEqual(first.communication_tensor_bytes, 128)
                self.assertEqual(tracker.report()["current"]["cpu"]["combined_bytes"], 16 + 128 + 16 + 32 + 64)
                first.wait()
                self.assertEqual(len(model.active), 1)
                first.wait()  # idempotent completion
                second.wait()
                self.assertFalse(model.active)
                self.assertEqual(tracker.report()["external_peak_bytes"]["cpu"], 96)

    def test_input_retained_until_completion(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            tracker = ExtendedMemTracker()
            runner = FakeCollectiveRunner(TestModel(tracker))
            with tracker:
                x = torch.empty(32, dtype=torch.float32)
                pending = runner.reduce_scatter(x, "test", 8)
                del x
                gc.collect()
                self.assertEqual(tracker.report()["current"]["cpu"]["tensor_bytes"], 128 + 16)
                pending.wait()
                gc.collect()
                self.assertEqual(tracker.report()["current"]["cpu"]["tensor_bytes"], 16)

    def test_dropped_handle_stays_live_until_wait_all(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            tracker = ExtendedMemTracker()
            model = TestModel(tracker)
            runner = FakeCollectiveRunner(model)
            with tracker:
                x = torch.empty(100, dtype=torch.uint8)
                pending = runner.all_gather(x, "test", 2, temporary_bytes=30)
                del x, pending
                gc.collect()
                self.assertEqual(tracker.report()["current"]["cpu"]["combined_bytes"], 330)
                self.assertEqual(len(model.active), 1)
                runner.wait_all()
                gc.collect()
                self.assertEqual(tracker.report()["current"], {})
                self.assertFalse(model.active)

    def test_real_input_rejected(self):
        runner = FakeCollectiveRunner(TestModel(ExtendedMemTracker()))
        with self.assertRaises(TypeError):
            runner.all_reduce(torch.empty(4), "test")


if __name__ == "__main__":
    unittest.main()
