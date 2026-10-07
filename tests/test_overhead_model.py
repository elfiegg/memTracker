import unittest
from memtracker_nccl.overhead_model import MemoryComponent, OtherMemoryModel
from memtracker_nccl.tracker import ExtendedMemTracker


class OtherMemoryTests(unittest.TestCase):
    def test_lifetimes_provenance_and_categories(self):
        tracker = ExtendedMemTracker()
        model = OtherMemoryModel(tracker)
        context = MemoryComponent('context', 100, 'CUDA context', 'assumed', 'scenario input, not measured')
        workspace = MemoryComponent('workspace', 200, 'Library workspace', 'measured', 'measurement fixture')
        model.start(context)
        with self.assertRaises(RuntimeError):
            with model.scope(workspace):
                self.assertEqual(tracker.report()['current']['cpu']['external_bytes'], 300)
                raise RuntimeError('kernel failure')
        self.assertEqual(tracker.report()['current']['cpu']['external_bytes'], 100)
        model.stop('context')
        self.assertEqual(tracker.report()['current'], {})
        self.assertEqual(model.describe()['components']['workspace']['provenance'], 'measurement fixture')
        self.assertEqual(tracker.report()['peak']['cpu']['external_by_category']['Library workspace'], 200)

    def test_rejects_ambiguous_values_and_duplicate_lifetime(self):
        for size in (-1, True, 1.2):
            with self.assertRaises((ValueError, TypeError)):
                MemoryComponent('x', size, 'Other', 'assumed', 'fixture')
        with self.assertRaises(ValueError):
            MemoryComponent('x', 1, 'Other', 'inferred', 'fixture')
        with self.assertRaises(ValueError):
            MemoryComponent('x', 1, 'Other', 'assumed', '')
        model = OtherMemoryModel(ExtendedMemTracker())
        component = MemoryComponent('x', 1, 'Other', 'assumed', 'fixture')
        model.start(component)
        with self.assertRaises(ValueError):
            model.start(component)
        model.stop('x')
        model.start(component)
        model.stop('x')
