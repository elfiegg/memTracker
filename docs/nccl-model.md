# What the NCCL model counts

This is a CPU-side memory accounting prototype. It generates allocation events
from an explicit profile; it does not run NCCL, discover a GPU topology, or
predict NCCL's algorithm selection. Profiles count locally owned device-buffer
requests. A collective's payload is already tensor memory. Registering that
tensor or mapping a peer's buffer does not allocate another local copy.

## Source-backed profiles

The formulas are pinned to **NCCL v2.28.9-1**. They are not a claim about every
NCCL version or configuration.

### Local P2P

`p2p_profile(channels, send_connections_per_channel=1,
recv_connections_per_channel=1, read=False)` models the local P2P transport
setup path, without an intermediate rank or memcpy staging.

For each specified local connection:

```
send = round_up(4096 + (SIMPLE if read else 0), 2 MiB)
recv = round_up(4096 + LL + LL128 + (0 if read else SIMPLE), 2 MiB)
```

The request construction is in
[p2pSendSetup/p2pRecvSetup](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/transport/p2p.cc#L346-L411).
The headers and alignment are defined in
[comm.h](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/include/comm.h#L32-L63).
Protocol defaults are SIMPLE=4 MiB, LL=512 KiB, LL128=4.6875 MiB, from
[computeBuffSizes](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/init.cc#L636-L650)
and [device constants](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/include/device.h#L64-L102).

With one send and receive connection, this is 12 MiB per channel. The caller
supplies the channel/connection counts; the model does not derive them from
world size. This profile is inappropriate for NET or NVLS buffers.

### Shared NET P2P pools

`shared_net_profile(channels, chunk_bytes=128*1024)` models GDR-enabled shared
NET P2P pools. Each enabled direction owns `channels * 16 * chunk_bytes` bytes.
It is one pool per top-parent local rank and direction, **not per remote peer**.
The formula and refcounted reuse are in
[sharedNetBuffersInit/Destroy](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/transport/net.cc#L572-L629).
128 KiB is the multi-node chunk default; the caller must supply the effective
chunk after overrides/capping. Dedicated ring/tree NET buffers, optional LL,
host memory and local P2P buffers are separate components and are excluded.

## Assumptions and lifecycle

- First collective use allocates the profile's persistent buffers. Subsequent
  collectives reuse them, independently of payload size. Destruction frees them.
- `temporary_bytes` is an explicit user allowance, not a source-derived NCCL
  workspace formula. It remains live until `complete_collective(token)`; multiple
  outstanding operations coexist. Destroying a communicator with pending work
  raises an error.
- Different modeled communicators own separate pools. Split communicators that
  share resources need a custom shared-resource model; blindly adding profiles
  can overestimate their memory.
- `calibrated_profile(bytes, provenance)` accepts a measurement or a declared
  sensitivity assumption. Zero means **overhead omitted**, not measured zero.

Every profile serializes its formulas' provenance, assumptions, confidence and
exclusions. Counts are device allocation requests under those assumptions, not
guaranteed physical consumption or an OOM bound. Communicator metadata, work
FIFOs, allocator rounding beyond modeled request alignment, CUDA context,
network plugins, NVLS/CollNet/GIN and graph allocations remain unmodeled.

## Calibration

Collect persistent and transient allocations on representative hardware, using
the same NCCL version, environment, communicator layout and workload. Separate
new backing allocations from virtual mappings and registrations. Supply the
observed **NCCL-only** bytes as a profile; do not add memory already tracked as
PyTorch tensor storage. Keep unknown components visible in the report.
