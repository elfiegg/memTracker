"""Opt-in CUDA allocation history capture, independent of model implementation."""
from contextlib import contextmanager
from copy import deepcopy
import os
import time


class TorchCudaProbe:
    """Private history API adapter. Use in a fresh process with no other recorder."""
    def __init__(self, device=None, *, process_memory_reader=None, isolated_peaks=False):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA capture requires a GPU; offline attribution runs on CPU')
        self.torch = torch
        self.device = torch.cuda.current_device() if device is None else device
        self.process_memory_reader = process_memory_reader
        self.isolated_peaks = isolated_peaks
        self.runtime_metadata = dict(pytorch=str(torch.__version__), cuda=torch.version.cuda,
            device_name=torch.cuda.get_device_name(self.device),
            allocator_environment={k:v for k,v in os.environ.items() if k.startswith(('PYTORCH_ALLOC','PYTORCH_CUDA_ALLOC','NCCL_'))})

    def start(self, max_entries):
        backend = self.torch.cuda.memory.get_allocator_backend()
        if backend != 'native':
            raise RuntimeError(f'Allocator history capture currently requires native backend, got {backend}')
        self.runtime_metadata['allocator_backend'] = backend
        self.torch.cuda.memory._record_memory_history(enabled='all', context='all',
            stacks='all', max_entries=max_entries, clear_history=True)

    def stop(self):
        self.torch.cuda.memory._record_memory_history(enabled=None)

    def sample(self, synchronize=False):
        cuda = self.torch.cuda
        if cuda.is_current_stream_capturing():
            raise RuntimeError('Cannot take phase checkpoints inside CUDA graph capture')
        if synchronize:
            cuda.synchronize(self.device)
        snapshot = cuda.memory._snapshot()
        stats = cuda.memory_stats(self.device)
        free, total = cuda.mem_get_info(self.device)
        row = dict(allocated_bytes=stats.get('allocated_bytes.all.current'),
            active_bytes=stats.get('active_bytes.all.current'),
            reserved_bytes=stats.get('reserved_bytes.all.current'),
            inactive_split_bytes=stats.get('inactive_split_bytes.all.current'),
            device_used_bytes=total-free, device_capacity_bytes=total,
            process_used_bytes=None, process_memory_error=None)
        row['block_sizes'] = [dict(address=b['address'], size=b['size'])
            for s in snapshot['segments'] if s['device'] == self.device
            for b in s['blocks'] if b['state'] != 'inactive']
        if self.isolated_peaks:
            row['interval_peaks'] = {f'{name}_bytes': stats.get(f'{name}_bytes.all.peak')
                for name in ('allocated', 'active', 'reserved')}
            cuda.reset_peak_memory_stats(self.device)
        if self.process_memory_reader is not None:
            try:
                row['process_used_bytes'] = self.process_memory_reader(self.device)
            except Exception as error:
                row['process_memory_error'] = f'{type(error).__name__}: {error}'
        return snapshot, row


class PhaseRecorder:
    """Record whole framework calls, preserving allocator history between calls.

    Owns memory-history recording for its context. By default it does not reset
    allocator peaks or clear cached memory. An isolated-peaks backend explicitly
    owns peak resets. This recorder is intended for one host thread per GPU.
    Diagnostic sampling failures invalidate attribution without hiding training
    exceptions. Initial setup failures abort capture explicitly.
    """
    def __init__(self, *, probe=None, max_entries=100000, synchronize=False, metadata=None, owner_provider=None):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError('max_entries must be a positive integer')
        self.probe = probe if probe is not None else TorchCudaProbe()
        self.max_entries, self.synchronize = max_entries, synchronize
        self.metadata = deepcopy(metadata or {})
        self.owner_provider = owner_provider
        self.spans, self.checkpoints, self.errors, self.stack = [], [], [], []
        self.previous = []
        self.baseline_segments, self.final_segments = [], []
        self.offset = 0
        self.started = self.closed = False

    def __enter__(self):
        if self.started:
            raise RuntimeError('PhaseRecorder is single use')
        self.probe.start(self.max_entries)
        try:
            snapshot, counters = self.probe.sample(synchronize=self.synchronize)
            self.previous = self._trace(snapshot)
            self.offset = len(self.previous)
            if self.offset >= self.max_entries:
                raise RuntimeError('History capacity exhausted before recording began')
            self.baseline_segments = snapshot['segments']
            self.final_segments = snapshot['segments']
            self.checkpoints.append(dict(trace_index=0, timestamp_ns=time.monotonic_ns(), **counters))
            self.started = True
        except BaseException:
            self.probe.stop()
            raise
        return self

    def _trace(self, snapshot):
        traces = snapshot['device_traces']
        return traces[self.probe.device] if self.probe.device < len(traces) else []

    def _checkpoint(self):
        if self.errors:
            return len(self.previous)-self.offset
        try:
            snapshot, counters = self.probe.sample(synchronize=self.synchronize)
            trace = self._trace(snapshot)
            if len(trace) >= self.max_entries:
                raise RuntimeError('History reached its capacity; truncation cannot be excluded')
            if trace[:len(self.previous)] != self.previous:
                raise RuntimeError('History was reset or overwritten')
            self.previous = trace
            self.final_segments = snapshot['segments']
            if self.owner_provider is not None:
                try:
                    counters['tensor_owners'] = self.owner_provider()
                except Exception as error:
                    counters['tensor_ownership_error'] = f'{type(error).__name__}: {error}'
            self.checkpoints.append(dict(trace_index=len(trace)-self.offset,
                timestamp_ns=time.monotonic_ns(), **counters))
        except Exception as error:
            self.errors.append(f'{type(error).__name__}: {error}')
        return len(self.previous)-self.offset

    @contextmanager
    def phase(self, name, **metadata):
        if not self.started or self.closed or not isinstance(name, str) or not name:
            raise ValueError('Phase needs an open recorder and a nonempty name')
        start = self._checkpoint()
        span = dict(id=len(self.spans), name=name, start=start, end=start,
            start_checkpoint=len(self.checkpoints)-1,
            parent=self.stack[-1]['id'] if self.stack else None, metadata=deepcopy(metadata), status='ok')
        self.spans.append(span)
        self.stack.append(span)
        try:
            yield
        except BaseException as error:
            span['status'] = 'error'
            span['error_type'] = type(error).__name__
            raise
        finally:
            span['end'] = self._checkpoint()
            span['end_checkpoint'] = len(self.checkpoints)-1
            self.stack.pop()

    def __exit__(self, exc_type, exc, tb):
        self._checkpoint()
        try:
            self.probe.stop()
        except Exception as error:
            self.errors.append(f'History shutdown: {type(error).__name__}: {error}')
        self.closed = True

    def report(self):
        return deepcopy(dict(schema_version=1, kind='cuda_framework_phase_capture',
            trace_size_semantics='requested',
            device=self.probe.device, metadata=self.metadata,
            observed_runtime=getattr(self.probe,'runtime_metadata',{}),
            history_complete=self.started and self.closed and not self.errors,
            errors=self.errors, baseline_segments=self.baseline_segments,
            final_segments=self.final_segments, events=self.previous[self.offset:],
            spans=self.spans, checkpoints=self.checkpoints,
            measurement_mode='synchronized_boundaries' if self.synchronize else 'unsynchronized_boundaries',
            isolated_peak_counters=bool(getattr(self.probe,'isolated_peaks',False)),
            limitations=['Single host thread and one selected CUDA device per recorder.',
                'Private PyTorch history API; fresh process required, no concurrent memory recorder.',
                'Snapshot/history collection perturbs host timing even without CUDA synchronization.',
                'Boundary counters are sequential samples, not an atomic device-wide snapshot.',
                'Library allocations outside PyTorch require boundary residuals or separate probes.',
                'Graph replay may have no allocator events; retained private pools remain in state.']))
