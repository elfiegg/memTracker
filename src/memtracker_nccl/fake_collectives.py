"""Explicit CPU-only wrappers around PyTorch's functional fake collectives.

This is an opt-in scheduling API, not a global interception of arbitrary model
collectives. Caller-provided communicator names/sizes and wait points determine
modeled lifetimes; no real process group, CUDA driver, or NCCL is initialized.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed._tools.fake_collectives import CollectiveOp


@dataclass
class PendingCollective:
    output: torch.Tensor
    token: Any
    communication_tensor_bytes: int
    _model: Any = field(repr=False)
    _input: torch.Tensor | None = field(repr=False)
    _runner: Any = field(repr=False)
    _completed: bool = False

    def wait(self) -> torch.Tensor:
        """Declare simulated completion; release temporary overhead and input hold."""
        if not self._completed:
            self._model.complete_collective(self.token)
            self._input = None
            self._completed = True
            self._runner._pending.pop(self.token)
        return self.output


class FakeCollectiveRunner:
    """Run supported fake tensor collectives alongside an NcclMemoryModel."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._pending: dict[Any, PendingCollective] = {}

    def wait_all(self) -> None:
        """Complete all scheduled work, including work whose handle was dropped."""
        for pending in list(self._pending.values()):
            pending.wait()

    def _run(self, tensor: torch.Tensor, communicator: str, operation: str,
             func: Any, args: tuple, temporary_bytes: int) -> PendingCollective:
        if not isinstance(tensor, FakeTensor):
            raise TypeError("FakeCollectiveRunner accepts only FakeTensor inputs")
        if tensor.device.type != "cpu":
            raise ValueError("This CPU-only prototype supports fake CPU tensors")
        token = self.model.begin_collective(
            communicator, operation=operation,
            payload_bytes=tensor.numel() * tensor.element_size(),
            temporary_bytes=temporary_bytes,
        )
        try:
            output = func(*args)
            traffic_metadata = CollectiveOp.get_comm_tensor_size(func, output, args, {})
        except BaseException:
            self.model.complete_collective(token)
            raise
        pending = PendingCollective(output, token, traffic_metadata, self.model, tensor, self)
        self._pending[token] = pending
        return pending

    def all_gather(self, tensor: torch.Tensor, communicator: str, world_size: int,
                   *, temporary_bytes: int = 0) -> PendingCollective:
        self._validate_world_size(world_size)
        if tensor.ndim == 0:
            raise ValueError("all_gather requires at least one dimension")
        func = torch.ops._c10d_functional.all_gather_into_tensor.default
        return self._run(tensor, communicator, "all_gather", func,
                         (tensor, world_size, communicator), temporary_bytes)

    def reduce_scatter(self, tensor: torch.Tensor, communicator: str, world_size: int,
                       *, temporary_bytes: int = 0) -> PendingCollective:
        self._validate_world_size(world_size)
        if tensor.ndim == 0 or tensor.shape[0] % world_size:
            raise ValueError("First dimension must be divisible by world_size")
        func = torch.ops._c10d_functional.reduce_scatter_tensor.default
        return self._run(tensor, communicator, "reduce_scatter", func,
                         (tensor, "sum", world_size, communicator), temporary_bytes)

    def all_reduce(self, tensor: torch.Tensor, communicator: str,
                   *, temporary_bytes: int = 0) -> PendingCollective:
        func = torch.ops._c10d_functional.all_reduce.default
        return self._run(tensor, communicator, "all_reduce", func,
                         (tensor, "sum", communicator), temporary_bytes)

    @staticmethod
    def _validate_world_size(world_size: int) -> None:
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise ValueError("world_size must be a positive integer")
