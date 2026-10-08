from copy import deepcopy
import unittest
from types import SimpleNamespace

from memtracker_nccl import cuda_capture as capture
from memtracker_nccl.phase_memory import analyze_capture


class Probe:
    device = 0
    def __init__(self):
        self.events = []
        self.synchronizations = 0
        self.stopped = False
    def start(self, max_entries): self.limit = max_entries
    def stop(self): self.stopped = True
    def sample(self, synchronize=False):
        self.synchronizations += int(synchronize)
        return dict(segments=[], device_traces=[deepcopy(self.events)]), dict(
            allocated_bytes=0, active_bytes=0, reserved_bytes=0,
            device_used_bytes=100, process_used_bytes=None)


class CaptureTests(unittest.TestCase):
    def test_api_exists(self):
        self.assertTrue(callable(getattr(capture, 'PhaseRecorder', None)))

    def test_nested_phases_preserve_history_and_exception(self):
        probe = Probe()
        recorder = capture.PhaseRecorder(probe=probe, max_entries=100)
        with recorder:
            with recorder.phase('initialize'):
                probe.events += [dict(action='segment_alloc', addr=100, size=1024),
                                 dict(action='alloc', addr=100, size=512)]
            with self.assertRaisesRegex(RuntimeError, 'original'):
                with recorder.phase('optimizer', step=1):
                    with recorder.phase('communication'):
                        raise RuntimeError('original')
            probe.events += [dict(action='free_requested', addr=100, size=512),
                             dict(action='free_completed', addr=100, size=512),
                             dict(action='segment_free', addr=100, size=1024)]
        r = recorder.report()
        self.assertEqual([s['name'] for s in r['spans']], ['initialize','optimizer','communication'])
        self.assertEqual(r['spans'][1]['status'], 'error')
        self.assertEqual(r['spans'][1]['metadata']['step'], 1)
        self.assertTrue(probe.stopped)
        self.assertEqual(probe.synchronizations, 0)
        self.assertEqual(analyze_capture(r)['final']['reserved_bytes'], 0)

    def test_truncation_and_history_reset_fail_closed(self):
        for truncate in (True, False):
            probe = Probe()
            recorder = capture.PhaseRecorder(probe=probe, max_entries=3)
            with recorder:
                with recorder.phase('a'):
                    probe.events = [dict(action='snapshot')]
                with recorder.phase('b'):
                    probe.events = [dict(action='snapshot')]*3 if truncate else []
            self.assertFalse(recorder.report()['history_complete'])
            with self.assertRaises(ValueError): analyze_capture(recorder.report())

    def test_diagnostic_sampling_failure_does_not_hide_training_error(self):
        probe = Probe()
        recorder = capture.PhaseRecorder(probe=probe)
        with self.assertRaisesRegex(ValueError, 'training failure'):
            with recorder, recorder.phase('forward'):
                probe.sample = lambda **kwargs: (_ for _ in ()).throw(RuntimeError('sampling failure'))
                raise ValueError('training failure')
        self.assertFalse(recorder.report()['history_complete'])
        self.assertTrue(probe.stopped)

    def test_sync_is_opt_in(self):
        probe = Probe()
        with capture.PhaseRecorder(probe=probe, synchronize=True) as recorder:
            with recorder.phase('x'): pass
        self.assertGreater(probe.synchronizations, 0)

    def test_cuda_backend_resets_peaks_only_in_explicit_isolated_mode(self):
        for isolated in (False,True):
            calls=[]
            cuda=SimpleNamespace(is_current_stream_capturing=lambda:False,
                synchronize=lambda device:calls.append('sync'),
                reset_peak_memory_stats=lambda device:calls.append('reset'),
                mem_get_info=lambda device:(100,1000),
                memory_stats=lambda device:{'allocated_bytes.all.current':512,'active_bytes.all.current':512,
                    'reserved_bytes.all.current':2048,'allocated_bytes.all.peak':1024,
                    'active_bytes.all.peak':1536,'reserved_bytes.all.peak':4096},
                memory=SimpleNamespace(_snapshot=lambda:dict(device_traces=[[]],segments=[dict(device=0,
                    blocks=[dict(address=100,size=512,state='active_allocated')])])))
            probe=capture.TorchCudaProbe.__new__(capture.TorchCudaProbe)
            probe.torch=SimpleNamespace(cuda=cuda)
            probe.device=0
            probe.process_memory_reader=None
            probe.isolated_peaks=isolated
            _, counters=probe.sample()
            self.assertEqual(calls,['reset'] if isolated else [])
            self.assertEqual(counters['block_sizes'],[dict(address=100,size=512)])
            self.assertEqual('interval_peaks' in counters,isolated)


if __name__ == '__main__': unittest.main()
