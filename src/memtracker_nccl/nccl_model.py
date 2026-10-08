"""Explicit NCCL memory scenarios, not a hardware/topology simulator.

Profiles describe locally owned device backing-buffer requests. Collective
payloads, imported peer mappings and registration of existing tensors are NOT
new allocations. Formula profiles cover specific NCCL 2.28.9 transport paths;
they do not estimate the entire NCCL footprint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol
from uuid import uuid4

NCCL_SOURCE = "https://github.com/NVIDIA/nccl/blob/v2.28.9-1/"
COMMON_EXCLUSIONS = (
    "Communicator/device metadata, work FIFO and CUDA context/library overhead.",
    "Network-plugin, NVLS, CollNet, GIN and CUDA graph allocations.",
    "Physical allocator granularity beyond the specified source request alignment.",
    "Additional transport connections, topology selection and split-communicator sharing.",
)


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _align(size: int, alignment: int) -> int:
    return ((size + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class BufferAllocation:
    name: str
    size_bytes: int
    provenance: str

    def __post_init__(self) -> None:
        _integer("size_bytes", self.size_bytes)
        if not self.name or not self.provenance:
            raise ValueError("buffer name and provenance must be nonempty")


@dataclass(frozen=True)
class CommunicatorProfile:
    name: str
    buffers: tuple[BufferAllocation, ...]
    assumptions: tuple[str, ...]
    excluded: tuple[str, ...] = COMMON_EXCLUSIONS
    confidence: str = "source formula under explicit assumptions; not GPU calibrated"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile name must be nonempty")
        names = [buffer.name for buffer in self.buffers]
        if len(names) != len(set(names)):
            raise ValueError("buffer names must be unique within a profile")

    @property
    def persistent_bytes(self) -> int:
        return sum(buffer.size_bytes for buffer in self.buffers)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "persistent_bytes": self.persistent_bytes}


def p2p_profile(
    channels: int,
    send_connections_per_channel: int = 1,
    recv_connections_per_channel: int = 1,
    read: bool = False,
    *,
    simple_bytes: int = 4 * 2**20,
    ll_bytes: int = 512 * 2**10,
    ll128_bytes: int = 120 * 640 * 8 * 8,
) -> CommunicatorProfile:
    """Local P2P transport requests for an explicitly supplied connection count.

    NCCL 2.28.9 p2pSendSetup/p2pRecvSetup request 4 KiB headers and
    protocol buffers rounded separately to CUDA_IPC_MIN (2 MiB).
    Do not apply this profile to NET/NVLS/shared-memory transports.
    """
    _integer("channels", channels, 1)
    for name, value in (("send_connections_per_channel", send_connections_per_channel),
                        ("recv_connections_per_channel", recv_connections_per_channel),
                        ("simple_bytes", simple_bytes), ("ll_bytes", ll_bytes),
                        ("ll128_bytes", ll128_bytes)):
        _integer(name, value)
    if not isinstance(read, bool):
        raise TypeError("read must be bool")
    send_size = _align(4096 + (simple_bytes if read else 0), 2**21)
    recv_size = _align(4096 + ll_bytes + ll128_bytes + (0 if read else simple_bytes), 2**21)
    provenance = NCCL_SOURCE + "src/transport/p2p.cc#L346-L349; " + NCCL_SOURCE + "src/transport/p2p.cc#L408-L411"
    buffers = tuple(
        BufferAllocation(f"channel-{channel}/{direction}-{connection}", size, provenance)
        for channel in range(channels)
        for direction, count, size in (
            ("send", send_connections_per_channel, send_size),
            ("recv", recv_connections_per_channel, recv_size),
        )
        for connection in range(count)
    )
    return CommunicatorProfile(
        name="nccl-2.28.9-local-p2p",
        buffers=buffers,
        assumptions=(
            f"Caller supplies channels={channels}, local send connections/channel={send_connections_per_channel}, receive connections/channel={recv_connections_per_channel}.",
            f"P2P read={read}; direct/IPC/cuMem P2P setup path, no intermediate rank or memcpy staging.",
            f"Protocol bytes: SIMPLE={simple_bytes}, LL={ll_bytes}, LL128={ll128_bytes}; defaults from {NCCL_SOURCE}src/init.cc#L636-L650 and src/include/device.h.",
            f"Header=4096 and request alignment=2097152 from {NCCL_SOURCE}src/include/comm.h#L32-L63.",
            "All declared connections become resident at first modeled use and persist to communicator destruction.",
            "Only locally owned storage is counted; imported remote mappings and existing user buffers are excluded from allocation events.",
        ),
    )


def shared_net_profile(
    channels: int,
    chunk_bytes: int = 128 * 1024,
    send_pool: bool = True,
    recv_pool: bool = True,
) -> CommunicatorProfile:
    """GDR P2P NET shared pools: channels * 16 slots * chunk per direction.

    Count once per top-parent local rank and direction, NOT once per remote peer.
    The caller must supply the effective chunk size after NCCL's size cap.
    """
    _integer("channels", channels, 1)
    _integer("chunk_bytes", chunk_bytes, 1)
    if not isinstance(send_pool, bool) or not isinstance(recv_pool, bool):
        raise TypeError("send_pool and recv_pool must be bool")
    size = channels * 16 * chunk_bytes
    return CommunicatorProfile(
        name="nccl-2.28.9-gdr-shared-net",
        buffers=tuple(
            BufferAllocation(f"shared-{direction}", size, NCCL_SOURCE + "src/transport/net.cc#L572-L607")
            for direction, enabled in (("send", send_pool), ("recv", recv_pool)) if enabled
        ),
        assumptions=(
            f"Effective P2P channels={channels}, chunk_bytes={chunk_bytes}; 16 shared slots per channel.",
            "GDR enabled, NET shared P2P buffer path; separate send/receive pools for one top-parent local rank.",
            "The 128 KiB default is the multi-node P2P chunk default; caller accounts for overrides and the SIMPLE-buffer cap.",
            "No per-peer or payload-size multiplier; all declared pools persist after first use.",
        ),
        excluded=COMMON_EXCLUSIONS + (
            "Dedicated ring/tree NET buffers and local P2P buffers must be modeled separately.",
            "Host staging/control memory and optional LL buffers.",
        ),
    )


def calibrated_profile(persistent_bytes: int, provenance: str) -> CommunicatorProfile:
    """An explicit observed or assumed allowance; provenance must say which.

    A zero allowance means omitted/unmodeled overhead, not a zero NCCL footprint.
    """
    _integer("persistent_bytes", persistent_bytes)
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("provenance must describe measurement or assumption")
    return CommunicatorProfile(
        name="explicit-allowance",
        buffers=(BufferAllocation("persistent-allowance", persistent_bytes, provenance),),
        assumptions=("Caller provides locally owned persistent bytes, excluding existing tensor storage.",
                     "A zero allowance omits NCCL overhead; it is not evidence of zero usage."),
        excluded=("Anything absent from the supplied measurement/assumption; no automatic topology extrapolation.",),
        confidence="user-supplied; accuracy depends on provenance",
    )


class AllocationSink(Protocol):
    def external_alloc(self, allocation_id: str, size_bytes: int, device: str = "cpu",
                       category: str = "NCCL", metadata: dict | None = None) -> Any: ...
    def external_free(self, allocation_id: str) -> Any: ...


@dataclass
class _Communicator:
    profile: CommunicatorProfile
    device: str
    allocation_ids: list[str] = field(default_factory=list)
    initialized: bool = False


class NcclMemoryModel:
    """Lifecycle adapter from explicit communicator profiles to an allocation sink.

    This deliberately does not choose NCCL algorithms or infer topology. Fake
    collective wrappers call begin/complete; payload tensors remain MemTracker's
    responsibility. Temporaries are an explicit scenario input, never inferred
    as a copy of collective payload bytes.
    """

    def __init__(self, tracker: AllocationSink):
        self.tracker = tracker
        self._prefix = f"nccl/{uuid4().hex}"
        self._communicators: dict[str, _Communicator] = {}
        self._pending: dict[str, tuple[str, str | None]] = {}
        self._next_id = 0

    def register_communicator(self, name: str, profile: CommunicatorProfile,
                              device: str = "cpu") -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("communicator name must be nonempty")
        if name in self._communicators:
            raise ValueError(f"communicator {name!r} already registered")
        self._communicators[name] = _Communicator(profile, device)

    def _id(self) -> str:
        self._next_id += 1
        return f"{self._prefix}/{self._next_id}"

    def initialize_communicator(self, name: str) -> None:
        """Materialize declared persistent backing without starting a collective.

        Idempotent; use before model allocation to model eager communication
        initialization. This does not allocate collective payloads/temporaries.
        It is a simulation event, not a call to real NCCL initialization.
        """
        comm = self._communicators[name]
        if not comm.initialized:
            allocated = []
            try:
                for buffer in comm.profile.buffers:
                    if not buffer.size_bytes:
                        continue
                    allocation_id = self._id()
                    self.tracker.external_alloc(
                        allocation_id, buffer.size_bytes, device=comm.device,
                        category="NCCL", metadata={"communicator": name,
                            "buffer": buffer.name, "lifetime": "persistent",
                            "provenance": buffer.provenance,
                            "confidence": comm.profile.confidence},
                    )
                    allocated.append(allocation_id)
            except Exception:
                for allocation_id in reversed(allocated):
                    self.tracker.external_free(allocation_id)
                raise
            comm.allocation_ids = allocated
            comm.initialized = True

    def begin_collective(self, name: str, *, operation: str = "all_reduce",
                         payload_bytes: int = 0, temporary_bytes: int = 0) -> str:
        _integer("payload_bytes", payload_bytes)
        _integer("temporary_bytes", temporary_bytes)
        self.initialize_communicator(name)
        comm = self._communicators[name]
        token = self._id()
        temporary_id = None
        if temporary_bytes:
            temporary_id = token + "/temporary"
            self.tracker.external_alloc(
                temporary_id, temporary_bytes, device=comm.device, category="NCCL",
                metadata={"communicator": name, "operation": operation,
                          "lifetime": "until_explicit_completion",
                          "payload_bytes": payload_bytes,
                          "provenance": "explicit user-supplied temporary allowance; not an NCCL source formula"},
            )
        self._pending[token] = (name, temporary_id)
        return token

    def complete_collective(self, token: str) -> None:
        _, temporary_id = self._pending[token]
        if temporary_id is not None:
            self.tracker.external_free(temporary_id)
        del self._pending[token]

    def destroy_communicator(self, name: str) -> None:
        comm = self._communicators[name]
        if any(pending_name == name for pending_name, _ in self._pending.values()):
            raise RuntimeError(f"communicator {name!r} has pending collectives")
        for allocation_id in reversed(comm.allocation_ids):
            self.tracker.external_free(allocation_id)
        del self._communicators[name]

    def describe(self) -> dict[str, Any]:
        return {name: {"device": str(comm.device), "initialized": comm.initialized,
                       "profile": comm.profile.to_dict()}
                for name, comm in self._communicators.items()}
