# K3 Interleaved1F1B: optimizer, FSDP and recomputation lifetimes

The expanded model estimates **186.084 GiB of simultaneously live tensors on
PP rank 4 / DP owner 0**, exceeding GB200 capacity **184.346 GiB** by **1.738 GiB
before NCCL or other external allocations**. With the previously used **16 GiB
resident communication allowance**, the subtotal is **202.084 GiB**. This
supersedes the earlier 142.475 GiB boundary-only subtotal. It is a **modeled
no-fit under the stated lifetimes**, not a measured full-training OOM or a
certified lower bound.

Recipe: PP8/FSDP32/EP32/TP1/CP1, Interleaved1F1B with two virtual stages per rank,
4068 tokens, microbatch 2 × 64, BF16 parameters, FP32 accumulated/reduction
gradients, DistMuon + AdamW, FullAC, HybridEP, CUDA graphs off. Communication
backing is resident throughout; no lazy initialization discount. No 256-GPU job
is used for this analysis.

## Included components

| Component | Evidence and lifetime |
|---|---|
| Muon momentum | Actual source `zeros_like(grad)` in storage layout for every dense/expert Muon parameter; persistent across iterations. |
| Owner-rank redistribution | Actual source owner planner and both packed all-to-all directions for all 32 DP owners and 16 virtual stages. Two rolling exchange/scratch slots plus local expert scratch reserved at construction. Both local stages' buffers persist together. |
| Newton–Schulz/update | Source prepare, five NS iterations, direction copy-back and final update execute under FakeTensor. Eighteen kernel traces; largest transient 5.414 GiB. Source pack/unpack executes in 292 distinct cases with zero additional tensor storage. Numerical network transfers/private NCCL buffers are not executed. |
| AdamW | Real foreach first/steady steps on source-derived local parameter shapes; BF16 moments for embeddings, head, norms, residual projections, KDA non-Muon tensors and vision. |
| FSDP | Dense gathered weights **and EDP1 expert copies**, gather staging, scheduled unshard waits, forward reshard, backward retention and reduction-time reshard. |
| FullAC | Unique checkpoint inputs persist until layer backward; shared residual stacks count once. Original forward helper temporaries are freed and recreated for backward. |
| Attention residual | Exact FP32 values, variance, normalized keys, scores, probabilities and autograd intermediates, replayed within the layer timeline. |
| Allocator | Same allocation/free events replayed with fixed/expandable segments, explicit compute/gather/PP/optimizer-transfer domains, rounding and cached backing. |

Muon selection follows parameter names, not simply “dense versus expert.”
AdamW handles the complement. Text-only K3 still runs dummy vision computation
and adds a zero-valued autograd dependency. Gradients are zero but not `None`,
so vision AdamW moments count.

## Simultaneous peak

The largest live peak is **virtual stage 12, microbatch 8, layer 71
recomputation/backward**. These values coexist at that event; independent
category maxima are not summed afterward.

| Live component | GiB |
|---|---:|
| Sharded parameters | 21.066 |
| Muon momentum | 21.065 |
| Muon persistent redistribution/scratch | 3.939 |
| Router buffers + AdamW moments | 0.042 |
| FSDP unsharded parameter copies | 33.260 |
| FP32 accumulated gradients | 66.521 |
| Retained checkpoint inputs | 22.703 |
| Retained pipeline outputs | 7.278 |
| Pending pipeline receive payloads | 0.978 |
| Attention-residual temporaries | 9.233 |
| **Total modeled live** | **186.084** |

The 5.414 GiB Newton–Schulz transient occurs later, during optimizer execution.
Adding it to this backward peak would incorrectly combine separate events.
Rank 4's modeled optimizer-phase live peak is **72.591 GiB**, including persistent
state and reduced BF16 gradients.

| PP rank | Live peak, no external memory | With 16 GiB communication |
|---|---:|---:|
| 0 | 170.182 | 186.182 |
| 1 | 176.983 | 192.983 |
| 2 | 182.825 | 198.825 |
| 3 | 180.894 | 196.894 |
| 4 | 186.084 | 202.084 |
| 5 | 179.807 | 195.807 |
| 6 | 180.110 | 196.110 |
| 7 | 155.807 | 171.807 |

## FullAC release/recompute

Original forward retains checkpoint inputs and observable outputs instead of
every internal activation. It cannot free input storage still referenced by
another checkpoint/microbatch. At layer backward, it reruns the needed forward
operations and consumes recreated saved tensors during backward. With
`early_stop=True`, recomputation stops once the required saved tensors exist.
K3 checkpoints a **whole decoder block**; the exact-function probe isolates its
attention-residual helper.
[PyTorch checkpoint behavior](https://docs.pytorch.org/docs/2.14/checkpoint.html)

At 8136 tokens and eight residual entries, the isolated helper retains
**4.998 GiB after eager forward**, versus **1.086 GiB after checkpointed forward**
(inputs, weights and output included). Both peak near **12.818 GiB** in backward.
FullAC reduces retention between operations, while recomputation still needs
these temporaries.

## Expandable segments

The modeled flag is **`expandable_segments=True`**, configured through
`PYTORCH_ALLOC_CONF=backend:native,expandable_segments:True`. A custom
`enable_segmentation` wrapper must map to that option for this comparison to
apply. Expandable segments grow mapped backing and reuse suitable gaps; they
do not shrink simultaneously live tensors.
[PyTorch allocator configuration](https://docs.pytorch.org/docs/2.14/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf)

Rank 4 replay: **211.166 GiB fixed versus 203.779 GiB expandable** reserved
backing before external communication; live peak **186.084 GiB in both cases**.
This reserved comparison is approximate. Stream delays, allocation-failure
cache release/retry and exact per-parameter gather padding are not simulated.
Reserved above capacity alone would not prove OOM: real allocation retries can
release unused cache.

## Remaining gaps and calibration status

This is **not yet a complete CUDA training-memory simulator**;
`full_model_fit_verified=false` remains explicit. Uncovered components are:

- Whole MLA/KDA/MoE activation graphs, CUDA kernel-internal workspaces and
  FullAC registered effects that preserve additional values.
- Whole-block recompute overlap: two residual helpers, attention and MoE can
  retain saved values together. One isolated helper per layer is replayed.
- Residual cache/gradient deposits beyond counted checkpoint/wire aliases;
  output/loss and vision activation intermediates.
- FSDP shard-reorder buffers, reduction packing/casts, exact padding and
  delayed stream reuse. Gather lifetimes use source rules with aggregate groups.
- Topology-specific NCCL/HybridEP private buffers and CUDA context/library
  memory. The 16 GiB allowance is **not a measurement**.

These gaps prevent a measured OOM claim, and prevent treating a below-capacity
subtotal as proof of fit. The one-device component calibration script is ready
and locally smoke-tested; automatic approval review blocked code upload to
Polyphe pending explicit transfer authorization. **No new GPU job was submitted.**
Its status is recorded in `experiments/k3_component_cuda_calibration_status.json`.
Even successful component calibration would not validate 32-rank NCCL/HybridEP
topology or the complete pipeline's asynchronous overlap.

## Reproduction and validation

TorchTitan source: `b4d5b404bd30dec67f86873203fbd4ba5f457531`. Target pipeline/FSDP:
PyTorch `2.15.0.dev20260928+cu130`. CPU probes: PyTorch 2.14.1. Evidence records
versions and source hashes. See [optimizer probes](k3_optimizer_probes.md) to
generate the compressed optimizer inputs.

```sh
PYTHONPATH=src python -m memtracker_nccl.k3_fullac_probe ../k3-fit/source \
  --output experiments/k3_fullac_probe.json
PYTHONPATH=src python -m memtracker_nccl.k3_training_lifetimes \
  --schedule experiments/k3_interleaved_schedule.json \
  --muon experiments/k3_muon_source_probe.json.gz \
  --adam experiments/k3_adamw_source_probe.json.gz \
  --residual experiments/k3_fullac_probe.json \
  --allocator expandable --communication-gib 16 \
  --output experiments/k3_training_lifetimes_expandable.json
```

Use `--allocator fixed` for comparison. **74 CPU tests pass**, including source
owner-route conservation, reservations, packing, NS lifetimes, AdamW first/steady
states, dummy-vision gradients, FullAC retention/recompute and small real CPU
gradient equivalence, schedule lifetime drainage and allocator behavior.
Independent spec and quality reviews passed for the explicitly partial scope.
