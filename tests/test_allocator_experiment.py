import unittest
from memtracker_nccl.allocator_experiment import combined_timeline, fragmentation

MiB = 2**20


class ExperimentTests(unittest.TestCase):
    def test_fragmentation_has_independent_expected_bytes(self):
        for expandable, expected in [(False, 68), (True, 60)]:
            report = fragmentation(expandable=expandable)
            rows = {r['phase']: r for r in report['checkpoints']}
            self.assertEqual(rows['allocator_fragmentation']['allocated_bytes'], 47*MiB)
            self.assertEqual(rows['allocator_fragmentation']['reserved_bytes'], expected*MiB)
            self.assertEqual(rows['allocator_released']['reserved_bytes'], 0)

    def test_lazy_persistence_changes_simultaneous_peak(self):
        for expandable, early_peak, late_peak in [(False, 41, 33), (True, 55, 47)]:
            early = combined_timeline(expandable=expandable, late_nccl=False)
            late = combined_timeline(expandable=expandable, late_nccl=True)
            self.assertEqual(early['resident']['peak']['cpu']['combined_bytes'], early_peak*MiB)
            self.assertEqual(late['resident']['peak']['cpu']['combined_bytes'], late_peak*MiB)
            self.assertEqual(early['resident']['current']['cpu']['combined_bytes'], 0)
            self.assertEqual(late['resident']['current']['cpu']['combined_bytes'], 0)
            self.assertEqual(early['external_peak_bytes']['cpu'], 16*MiB)
