# Inputs and scope

## Training configuration

Obtain the effective configuration after defaults and CLI overrides. Preserve
its serialized form and fingerprint alongside every report.

| Area | Required details |
|---|---|
| Model | Actual constructors, parameter shapes, tied weights, buffers, attention/MoE implementation, expert count/top-k, routing and padding |
| Batch | Sequence length, microbatch size/count, accumulation, variable-length policy |
| Parallelism | PP/DP/FSDP/EP/TP/CP, replication, rank mesh/order, stage splits, virtual stages, exact pipeline schedule |
| Precision | Parameter, activation, reduction, optimizer-state and master-weight dtypes; quantization |
| Optimizer | Parameter groups and actual implementation, sharding/owner assignment, buckets, foreach/fused settings, zero-grad policy |
| Lifetimes | Checkpoint policy, FSDP reshard/prefetch, eager/lazy PP buffers and communicator connections, async overlap |
| Runtime | Compile/graphs/warmup, allocator backend/config, NCCL settings/topology/transport, kernel choices |
| Hardware | GPU type, usable per-device capacity, sharing with other processes |
| Measurement | Rank, phase/step, peak-reset window, units, success/OOM, log/snapshot provenance |

Do not infer dimensions from a model name, treat DP and EP as disjoint GPU
multipliers without reading the mesh, or infer activation residency from the
microbatch count alone. Interleaved schedules require stage ownership and action
order. If exact details are absent, report explicit scenarios and conditional
results instead of choosing a hidden default.

## One public identity field per component

The branch accepts these JSON/API keys and equivalent CLI flags:

| JSON/API | CLI | Value |
|---|---|---|
| `training_code` | `--training-code` | Full 40-character training checkout/base commit |
| `torchao` | `--torchao` | Full 40-character TorchAO commit |
| `pytorch` | `--pytorch` | Full PyTorch version/build string |
| `cuda` | `--cuda` | CUDA version reported by PyTorch |
| `nccl` | `--nccl` | One full version/build string |
| `container` | `--container` | One path, tag, URI, or immutable digest |

Use `--run-metadata /path/run.json` where supported. Optional defaults use
`--runtime-profile /path/profile.json`, with this structure:

```json
{
  "schema_version": 1,
  "name": "my-target-runtime",
  "values": {"cuda": "13.0"}
}
```

Precedence: explicit flags > run metadata > profile > unknown. Fallback values
are assumed; explicit null suppresses fallback. Do not silently apply the saved
K3 profile to a new model. Consult repository `docs/run-metadata.md` for the
selected command's behavior.

A base commit may have local patches. Capture observed checkout/dirty state and
available source hashes automatically; do not add extra required user fields.
A container path is an identifier, not proof of immutable contents. Keep declared
target identity, observed execution runtime, and CPU simulator runtime separate.
Metadata does not download dependencies or switch the modeled implementation.

## Coverage inventory

For each row below, record `executed`, `source modeled`, `calibrated`, or `unknown`,
with the evidence and lifetime assumptions:

- Parameters, gradients, optimizer state, tied/shared storage.
- Checkpoint inputs, residuals, recomputation and backward intermediates.
- FSDP gathered weights, prefetch overlap, PP send/receive buffers.
- Optimizer momentum, Newton–Schulz intermediates, update/redistribution scratch,
  owner-rank all-to-all, and buffers outside optimizer `.state`.
- Attention/GEMM/MoE kernel workspaces and compile/graph-private pools.
- Allocator rounding, caching, expandable backing and deferred frees.
- Communicator buffers, connection initialization, context/library allocations.

CPU FakeTensor records visible tensor operations; opaque CUDA kernel allocations
and transport behavior require additional modeling or measurement. A missing
entry is not evidence of zero cost.
