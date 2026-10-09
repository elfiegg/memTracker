import json
from pathlib import Path
import unittest

import torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

from memtracker_nccl.k3_fullac_probe import load_functions, summarize_trace

ROOT = Path(__file__).resolve().parents[1]


class FullACProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / 'experiments/k3_fullac_probe.json').read_text())

    def test_checkpoint_keeps_inputs_and_output_but_recomputes_internal_tensors(self):
        report = self.report
        for row in report['probes']:
            if not row['full_ac']:
                continue
            h = report['tokens'] * report['dim'] * 2
            expected = (row['residual_entries'] + 2)*h + report['dim']*4
            self.assertEqual(row['after_forward_live_bytes'], expected)
            self.assertEqual(row['function_call_phases'], ['forward', 'recompute'])
            eager = next(p for p in report['probes'] if p['residual_entries'] == row['residual_entries'] and not p['full_ac'])
            self.assertGreater(eager['after_forward_live_bytes'], row['after_forward_live_bytes'])
            # Checkpointing reduces between-call retention, not this helper's
            # peak during the backward operation itself.
            self.assertEqual(eager['allocator']['fixed']['peak_live_bytes'], row['allocator']['fixed']['peak_live_bytes'])

    def test_allocator_modes_replay_identical_storage_lifetimes(self):
        for row in self.report['probes']:
            fixed = row['allocator']['fixed']; expandable = row['allocator']['expandable']
            self.assertEqual(fixed['peak_live_bytes'], expandable['peak_live_bytes'])
            self.assertGreaterEqual(expandable['peak_reserved_bytes'], expandable['peak_live_bytes'])
        # Exercise replay rather than only checking serialized expected numbers.
        events = self.report['probes'][0]['storage_events']
        self.assertEqual(summarize_trace(events), self.report['probes'][0]['allocator']['fixed'])

    def test_checkpoint_matches_actual_small_cpu_gradients(self):
        source = ROOT.parent / 'k3-fit/source'
        if not source.exists():
            self.skipTest('Optional pinned TorchTitan source checkout is not available')
        helper, policy = load_functions(source)
        torch.manual_seed(1)
        projection = torch.nn.Linear(8, 1, bias=False)
        norm = torch.nn.RMSNorm(8, eps=1e-5)
        x = torch.randn(4, 8, requires_grad=True)
        residual = torch.randn(4, 3, 8, requires_grad=True)
        params = (x, residual, projection.weight, norm.weight)
        def fn(a, b):
            return helper(a, b, projection, norm)
        eager = fn(x, residual)
        expected = torch.autograd.grad(eager.sum(), params)
        ac = checkpoint(fn, x, residual, use_reentrant=False,
                        context_fn=lambda: create_selective_checkpoint_contexts(policy))
        actual = torch.autograd.grad(ac.sum(), params)
        torch.testing.assert_close(eager, ac)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)


if __name__ == '__main__':
    unittest.main()
