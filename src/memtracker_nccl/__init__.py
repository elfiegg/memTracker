"""CPU-only experiments combining PyTorch tensor and modeled NCCL memory."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tracker import ExtendedMemTracker

__all__ = ["ExtendedMemTracker"]


def __getattr__(name: str):
    # Keep the pure allocation model usable without importing PyTorch.
    if name == "ExtendedMemTracker":
        from .tracker import ExtendedMemTracker
        return ExtendedMemTracker
    raise AttributeError(name)
