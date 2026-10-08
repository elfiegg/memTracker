from contextlib import contextmanager
import unittest

from memtracker_nccl import torchtitan_capture as adapter


class Recorder:
    def __init__(self): self.names = []; self.metadata = {}
    @contextmanager
    def phase(self, name, **metadata):
        self.names.append((name, metadata))
        yield


class Engine:
    class Config:
        def to_dict(self): return {'model': {'dim': 42}, 'parallelism': {'pp': 1}}
    config = Config()
    num_completed_steps = 0
    def _initialize_model(self, *, marker): return marker
    def _initialize_optimizer(self): return 5
    def forward_backward_microbatch(self, **kwargs): return kwargs
    def optimizer_step(self): raise ValueError('optimizer failed')


class AdapterTests(unittest.TestCase):
    def test_adapter_exists(self):
        self.assertTrue(callable(getattr(adapter, 'instrument_engine', None)))

    def test_wraps_framework_methods_and_restores_after_exception(self):
        recorder = Recorder()
        original = Engine.optimizer_step
        with self.assertRaisesRegex(ValueError, 'optimizer failed'):
            with adapter.instrument_engine(Engine, recorder):
                engine = Engine()
                self.assertEqual(engine._initialize_model(marker='ok'), 'ok')
                self.assertEqual(engine.forward_backward_microbatch(accumulation_index=7), {'accumulation_index':7})
                engine.optimizer_step()
        self.assertIs(Engine.optimizer_step, original)
        self.assertEqual([r[0] for r in recorder.names], ['model_initialization','forward_backward','optimizer'])
        self.assertEqual(recorder.names[1][1]['accumulation_index'], 7)
        self.assertIn('effective_config_sha256', recorder.metadata)
        self.assertEqual(recorder.metadata['effective_config']['model']['dim'], 42)

    def test_unknown_framework_interface_fails_before_patching(self):
        class Other: pass
        with self.assertRaisesRegex(ValueError, 'Unsupported TorchTitan'):
            with adapter.instrument_engine(Other, Recorder()): pass
        self.assertFalse(hasattr(Other, 'optimizer_step'))

    def test_inherited_methods_restored_without_leaving_class_overrides(self):
        class Child(Engine): pass
        with adapter.instrument_engine(Child, Recorder()):
            self.assertIn('_initialize_model', Child.__dict__)
        self.assertNotIn('_initialize_model', Child.__dict__)


if __name__ == '__main__': unittest.main()
