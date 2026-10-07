"""Explicit lifetimes for memory owned outside the modeled PyTorch allocator."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from uuid import uuid4

from .nccl_model import AllocationSink, _integer


@dataclass(frozen=True)
class MemoryComponent:
    name: str
    size_bytes: int
    category: str
    evidence: str
    provenance: str

    def __post_init__(self):
        _integer('size_bytes', self.size_bytes)
        if not all(isinstance(v, str) and v.strip() for v in (self.name, self.category, self.provenance)):
            raise ValueError('name, category and provenance must be nonempty strings')
        if self.evidence not in ('assumed', 'measured', 'source_formula'):
            raise ValueError('evidence must be assumed, measured or source_formula')


class OtherMemoryModel:
    """Add disjoint external backing with supplied byte sizes and lifetimes.

    A cuBLAS/cuDNN workspace obtained through PyTorch's caching allocator must
    be represented by an allocator event, not again by an external component.
    A measured residual containing NCCL cannot be added to an NCCL profile.
    No default CUDA/context/workspace constants are assumed.
    """

    def __init__(self, tracker: AllocationSink):
        self.tracker = tracker
        self._namespace = f'other-{uuid4().hex}'
        self._active: dict[str, str] = {}
        self._components: dict[str, dict] = {}

    def start(self, component: MemoryComponent, *, device: str = 'cpu') -> None:
        if component.name in self._active:
            raise ValueError(f'Component already active: {component.name}')
        allocation_id = f'{self._namespace}/{component.name}'
        self.tracker.external_alloc(allocation_id, component.size_bytes, device=device,
                                    category=component.category, metadata=asdict(component))
        self._active[component.name] = allocation_id
        self._components[component.name] = asdict(component)

    def stop(self, name: str) -> None:
        self.tracker.external_free(self._active[name])
        del self._active[name]

    @contextmanager
    def scope(self, component: MemoryComponent, *, device: str = 'cpu'):
        self.start(component, device=device)
        try:
            yield
        finally:
            self.stop(component.name)

    def describe(self) -> dict:
        return {'components': {k: dict(v) for k, v in self._components.items()},
                'active': sorted(self._active),
                'assumptions': ['All components are disjoint from tensors, allocator backing, and NCCL profiles.',
                                'Unspecified components are omitted, not predicted zero.']}
