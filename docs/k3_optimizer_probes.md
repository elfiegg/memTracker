# K3 optimizer memory probes

These CPU-only probes separate persistent optimizer storage from transient step
allocations. They do not predict a complete training peak or launch GPU work.
They require the retained TorchTitan `b4d5b404bd30dec67f86873203fbd4ba5f457531`
source archive or a clean checkout of that commit, plus the existing PP8/VPP2
schedule JSON. Source files are checked against the archive/checkout before use.

```sh
PYTHONPATH=src python -m memtracker_nccl.k3_muon_probe \
  --source ../k3-fit/source --schedule experiments/k3_interleaved_schedule.json \
  --output experiments/k3_muon_source_probe.json.gz
PYTHONPATH=src python -m memtracker_nccl.k3_adamw_probe \
  --source ../k3-fit/source --schedule experiments/k3_interleaved_schedule.json \
  --output experiments/k3_adamw_source_probe.json.gz
```

The Muon probe imports the actual FlexShard source modules in an isolated
package. It executes the K3 optimizer configuration function, creates real
CPU DeviceMeshes with FakeStore, builds source DistMuon plans from meta DTensor
parameters, and uses the source packed schedule builder for all 32 participants.
Only the cross-process hash validation is skipped. The source runtime reserves
its two pipeline slots and separate local scratch under FakeTensor with no-op
CPU stream/event hooks. Owner balancing is performed by the source planner,
independently for each virtual model part.

All configured Muon parameters get source `zeros_like(grad)` momentum, including
the dense matrices. The source prepare, five Newton–Schulz iterations,
direction copy-back and final update execute on real-size FakeTensors. Exact
source pack/unpack operations also execute, with callback kernels accounted for
by their separate probes. Numerical all-to-all communication is not executed.

The AdamW probe uses explicit non-Muon parameter shapes, including empty
FSDP shards of single-row residual projections. The shapes reconcile with the
independent aggregate inventory minus Muon, and all vision shapes reconcile
with the vision aggregate. Actual `torch.optim.AdamW(foreach=True,fused=False)`
executes first and steady steps. With BF16 parameters and gradients, both moments
are BF16. CPU step scalars are reported separately from persistent GPU moments.

The default K3 text-only path **still runs the vision encoder** on dummy inputs
and adds a zero-valued dependency. Its gradients are zero but not `None`, so
AdamW initializes vision moments. `--without-vision` is appropriate only when
the model configuration explicitly removes the encoder.

Each `stage_owners` row includes `persistent_allocations` with allocation IDs,
bytes, pool and stream. Muon rows include per-bucket exchange/scratch sizes,
actual owner split sizes and references to shared kernel probes. Kernel
`storage_events` contain allocation/free operations and storage IDs for allocator
replay. Newton–Schulz events drain to zero; AdamW first-step moment allocations
persist, while its steady-step transient events drain. `bf16_optimizer_gradient_bytes`
describes reduced gradients at optimizer time; it must replace, rather than be
blindly added to, full FP32 pre-reduction accumulated gradients.

In the generated PP8/VPP2 report, Muon runtime buffers occupy 2.48–4.83 GiB per
GPU, depending on pipeline rank and DP owner. Each GPU's largest eager
Newton–Schulz transient is 5.4140625 GiB for the local expert w13 batch. These
temporaries occur during optimizer execution, not necessarily at the training
activation/gradient peak. Each model part's reservations remain persistent.

Limits: parameter shapes are source-transcribed, not obtained by constructing
the full K3 model. The local helper runtime is PyTorch 2.14.1; source provenance
and runtime version are in every report. CUDA library workspace, NCCL private
memory, asynchronous allocator reuse, FSDP storage padding and the complete
training timeline require separate models. CPU scalar temporaries in the AdamW
trace conservatively overcount GPU usage by a few bytes.
