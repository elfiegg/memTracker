"""TorchTitan lifecycle adapter; no model-specific code or shape formulas."""
import argparse
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import weakref

from .cuda_capture import PhaseRecorder, TorchCudaProbe
from .phase_memory import analyze_capture
from .report_io import write_report
from .run_metadata import add_metadata_arguments, metadata_from_args
from .tensor_inventory import tensor_inventory


PHASES = {
    '_initialize_distributed_runtime': 'distributed_initialization',
    '_initialize_model': 'model_initialization',
    '_initialize_optimizer': 'optimizer_initialization',
    '_initialize_checkpointer': 'checkpoint_initialization',
    '_initialize_forward_backward': 'execution_initialization',
    'prepare_step': 'prepare_step',
    'forward_backward_microbatch': 'forward_backward',
    'optimizer_step': 'optimizer',
}
REQUIRED = ('_initialize_model', '_initialize_optimizer', 'forward_backward_microbatch', 'optimizer_step')


@contextmanager
def instrument_engine(engine_class, recorder):
    missing = [name for name in REQUIRED if not callable(getattr(engine_class, name, None))]
    if missing:
        raise ValueError(f'Unsupported TorchTitan lifecycle interface: missing {missing}')
    originals = {}
    source = inspect.getsourcefile(engine_class)
    recorder.metadata['framework_adapter'] = dict(engine_class=engine_class.__qualname__,
        source_file=source, source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None)
    def wrap(original, phase):
        @wraps(original)
        def measured(engine, *args, **kwargs):
            if getattr(recorder, 'owner_provider', None) is None:
                reference = weakref.ref(engine)
                def owners():
                    live = reference()
                    if live is None:
                        return dict(storages=[], errors=[])
                    container = getattr(live, 'optimizers', None)
                    optimizers = getattr(container, 'optimizers', [])
                    return tensor_inventory(getattr(live, 'model_parts', []), optimizers)
                recorder.owner_provider = owners
            if 'effective_config' not in recorder.metadata:
                config = engine.config.to_dict()
                serialized = json.dumps(config, sort_keys=True, separators=(',', ':'))
                recorder.metadata.update(effective_config=config,
                    effective_config_sha256=hashlib.sha256(serialized.encode()).hexdigest())
            metadata = dict(step=getattr(engine, 'num_completed_steps', 0)+1)
            if 'accumulation_index' in kwargs:
                metadata['accumulation_index'] = kwargs['accumulation_index']
            with recorder.phase(phase, **metadata):
                return original(engine, *args, **kwargs)
        return measured
    try:
        for method, phase in PHASES.items():
            if callable(getattr(engine_class, method, None)):
                originals[method] = (method in engine_class.__dict__, getattr(engine_class, method))
                setattr(engine_class, method, wrap(originals[method][1], phase))
        yield
    finally:
        for method, (owned, original) in originals.items():
            if owned: setattr(engine_class, method, original)
            else: delattr(engine_class, method)


def _source_identity(path):
    result = {}
    try:
        def git(*args):
            return subprocess.check_output(['git', '-C', str(Path(path).parent), *args], stderr=subprocess.DEVNULL)
        result['git_head'] = git('rev-parse', 'HEAD').decode().strip()
        result['dirty'] = bool(git('status', '--porcelain'))
        result['tracked_diff_sha256'] = hashlib.sha256(git('diff', 'HEAD', '--binary')).hexdigest()
        result['scope'] = 'Git HEAD and tracked diff fingerprint; untracked files not hashed'
    except (OSError, subprocess.CalledProcessError):
        result['git_unavailable'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Per-rank capture directory')
    parser.add_argument('--max-entries', type=int, default=100000)
    parser.add_argument('--synchronize', action='store_true', help='Synchronize phase boundaries; changes overlap')
    add_metadata_arguments(parser)
    parser.add_argument('training_args', nargs=argparse.REMAINDER, help='TorchTitan arguments after --')
    args = parser.parse_args()
    import torch
    from torchtitan.training_engine import TrainingEngine
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    rank = int(os.environ.get('RANK', '0'))
    source = inspect.getsourcefile(TrainingEngine)
    runtime = dict(pytorch=str(torch.__version__), cuda=torch.version.cuda,
        nccl_api_version=torch.cuda.nccl.version(), device_name=torch.cuda.get_device_name(),
        rank=rank, world_size=int(os.environ.get('WORLD_SIZE', '1')),
        allocator_environment={k:v for k,v in os.environ.items() if k.startswith(('PYTORCH_ALLOC', 'PYTORCH_CUDA_ALLOC', 'NCCL_'))})
    recorder = PhaseRecorder(probe=TorchCudaProbe(), max_entries=args.max_entries,
        synchronize=args.synchronize, metadata=dict(run_metadata=metadata_from_args(args),
            observed_runtime=runtime, observed_training_source=_source_identity(source)))
    old_argv = sys.argv
    training_args = args.training_args[1:] if args.training_args[:1] == ['--'] else args.training_args
    try:
        sys.argv = ['torchtitan.train', *training_args]
        with recorder, instrument_engine(TrainingEngine, recorder):
            with recorder.phase('training_run'):
                runpy.run_module('torchtitan.train', run_name='__main__')
    finally:
        sys.argv = old_argv
        report = recorder.report()
        write_report(args.output/f'rank-{rank}-capture.json.gz', report)
        try:
            analysis = analyze_capture(report)
        except Exception as error:
            analysis = dict(status='attribution_unavailable', error=f'{type(error).__name__}: {error}')
        write_report(args.output/f'rank-{rank}-analysis.json.gz', analysis)
    if analysis.get('status') == 'attribution_unavailable':
        raise RuntimeError(f"Capture saved, attribution unavailable: {analysis['error']}")


if __name__ == '__main__': main()
