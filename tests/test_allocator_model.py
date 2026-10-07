"""Pure CPU allocator scenarios with explicit backing and lifetime expectations."""
import copy
import unittest

from memtracker_nccl.allocator_model import AllocatorConfig, CachingAllocatorModel

MiB = 2**20


class AllocatorTests(unittest.TestCase):
    def model(self, expandable=False):
        return CachingAllocatorModel(AllocatorConfig(expandable_segments=expandable))

    def stats(self, model, device="cpu"):
        return model.snapshot()[device]

    def location(self, model, allocation_id):
        for device in model.snapshot().values():
            for segment in device["segments"]:
                for block in segment["blocks"]:
                    if block["allocation_id"] == allocation_id and block["state"] == "active":
                        return segment["segment_id"], block["offset_bytes"]
        self.fail(f"No active allocation {allocation_id}")

    def test_default_rounding_segment_thresholds_and_zero(self):
        for size, expected_reserved in ((1, 2*MiB), (MiB, 2*MiB),
                                        (MiB+1, 20*MiB), (10*MiB, 10*MiB),
                                        (10*MiB+1, 12*MiB)):
            model = self.model()
            model.allocate("a", size)
            stats = self.stats(model)
            self.assertEqual(stats["requested_bytes"], size)
            self.assertEqual(stats["allocated_bytes"], ((size+511)//512)*512)
            self.assertEqual(stats["reserved_bytes"], expected_reserved)
        model = self.model()
        model.allocate("zero", 0)
        self.assertEqual(self.stats(model)["reserved_bytes"], 0)
        with self.assertRaises(ValueError):
            model.allocate("zero", 0)
        model.free("zero")

    def test_best_fit_coalescing_and_fixed_segment_release(self):
        model = self.model()
        for name, size in (("a", 2), ("b", 3), ("c", 4), ("d", 5)):
            model.allocate(name, size*MiB)
        b_location = self.location(model, "b")
        model.free("b")
        model.allocate("e", 2*MiB)
        self.assertEqual(self.location(model, "e"), b_location)
        model.free("e")
        model.free("c")
        model.allocate("f", 7*MiB)
        self.assertEqual(self.location(model, "f"), b_location)
        model.empty_cache()
        self.assertEqual(self.stats(model)["reserved_bytes"], 20*MiB)
        for name in ("a", "d", "f"):
            model.free(name)
        model.empty_cache()
        self.assertEqual(self.stats(model)["reserved_bytes"], 0)

    def test_stranded_fixed_tails_vs_expandable_tail_growth(self):
        # Each fixed 16 MiB request reserves 16 MiB. Replacing them by 14 MiB
        # leaves three isolated 2 MiB tails, insufficient for a 5 MiB request.
        models = [self.model(False), self.model(True)]
        for model in models:
            for index in range(3):
                model.allocate(f"old{index}", 16*MiB)
            for index in range(3):
                model.free(f"old{index}")
            for index in range(3):
                model.allocate(f"new{index}", 14*MiB)
            before = [self.location(model, f"new{i}") for i in range(3)]
            model.allocate("extra", 5*MiB)
            self.assertEqual(before, [self.location(model, f"new{i}") for i in range(3)])
        self.assertEqual(self.stats(models[0])["reserved_bytes"], 68*MiB)
        self.assertEqual(self.stats(models[1])["reserved_bytes"], 60*MiB)
        self.assertEqual(self.stats(models[0])["segment_count"], 4)
        self.assertEqual(self.stats(models[1])["segment_count"], 1)

    def test_expandable_interior_unmap_remap_does_not_relocate_live_blocks(self):
        model = self.model(True)
        for name in ("left", "middle", "right"):
            model.allocate(name, 20*MiB)
        left = self.location(model, "left")
        middle = self.location(model, "middle")
        right = self.location(model, "right")
        model.free("middle")
        model.empty_cache()
        self.assertEqual(self.stats(model)["reserved_bytes"], 40*MiB)
        self.assertEqual(self.stats(model)["cached_bytes"], 0)
        model.allocate("replacement", 20*MiB)
        self.assertEqual(self.location(model, "replacement"), middle)
        self.assertEqual(self.location(model, "left"), left)
        self.assertEqual(self.location(model, "right"), right)
        self.assertEqual(self.stats(model)["reserved_bytes"], 60*MiB)

    def test_partial_expandable_page_cannot_be_unmapped(self):
        model = self.model(True)
        model.allocate("a", 21*MiB)
        model.empty_cache()
        self.assertEqual(self.stats(model)["reserved_bytes"], 40*MiB)
        model.free("a")
        model.empty_cache()
        self.assertEqual(self.stats(model)["reserved_bytes"], 0)
        model.allocate("b", 21*MiB)
        self.assertEqual(self.stats(model)["reserved_bytes"], 40*MiB)

    def test_deferred_free_multiple_dependencies_and_completion_while_live(self):
        for expandable in (False, True):
            model = self.model(expandable)
            model.allocate("a", 20*MiB)
            model.record_stream("a", "already-done")
            model.complete("already-done")
            model.record_stream("a", "one")
            model.record_stream("a", "two")
            model.free("a")
            model.empty_cache()
            self.assertEqual(self.stats(model)["allocated_bytes"], 0)
            self.assertEqual(self.stats(model)["pending_free_bytes"], 20*MiB)
            self.assertEqual(self.stats(model)["cached_bytes"], 0)
            model.allocate("a", 20*MiB)  # A reused logical id is a new block.
            self.assertEqual(self.stats(model)["reserved_bytes"], 40*MiB)
            model.complete("one")
            self.assertEqual(self.stats(model)["pending_free_bytes"], 20*MiB)
            model.complete("two")
            self.assertEqual(self.stats(model)["pending_free_bytes"], 0)
            model.complete("two")  # Repeated completion has no effect.
            model.free("a")
            model.empty_cache()
            self.assertEqual(self.stats(model)["reserved_bytes"], 0)

    def test_device_pool_stream_and_small_large_are_separate(self):
        for expandable in (False, True):
            model = self.model(expandable)
            model.allocate("a", 512)
            model.free("a")
            model.allocate("pool", 512, pool="private")
            model.allocate("stream", 512, stream="other")
            model.allocate("device", 512, device="cuda:1")
            model.allocate("large", 2*MiB)
            self.assertEqual(self.stats(model)["reserved_bytes"], 26*MiB)
            self.assertEqual(self.stats(model, "cuda:1")["reserved_bytes"], 2*MiB)
            self.assertEqual(self.stats(model)["pools"]["private"]["reserved_bytes"], 2*MiB)
            model.empty_cache(device="cuda:1")
            self.assertEqual(self.stats(model)["reserved_bytes"], 26*MiB)

    def test_invalid_transitions_are_atomic_and_snapshot_is_detached(self):
        model = self.model(True)
        model.allocate("a", 10)
        before = copy.deepcopy(model.snapshot())
        operations = [lambda: model.allocate("a", 20), lambda: model.allocate("b", -1),
                      lambda: model.allocate("b", True), lambda: model.allocate("b", 1.5),
                      lambda: model.allocate("b", 1, stream=""), lambda: model.free("missing"),
                      lambda: model.record_stream("missing", "x"),
                      lambda: model.record_stream("a", ""), lambda: model.empty_cache(device=4),
                      lambda: model.complete(42)]
        for operation in operations:
            with self.assertRaises((TypeError, ValueError)):
                operation()
            self.assertEqual(model.snapshot(), before)
        report = model.snapshot()
        report["cpu"]["segments"].clear()
        self.assertEqual(model.snapshot(), before)
        self.assertIn("approximation", model.describe()["confidence"])
        self.assertIn("v2.8.0", str(model.describe()))
        with self.assertRaises(TypeError):
            AllocatorConfig(expandable_segments=1)


if __name__ == "__main__":
    unittest.main()
