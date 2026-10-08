# K3 Interleaved1F1B: CPU estimate

**Superseded for fit assessment:** the [expanded lifetime analysis](k3-training-lifetimes.md)
adds optimizer runtime buffers, FSDP copies and attention-residual recomputation.
The figures below remain the earlier partial boundary subtotal.

For the requested PP8/FSDP32/EP32, S4068, microbatch 2 × 64, BF16, FP32 reduction,
FullAC, DistMuon+AdamW, HybridEP, CUDA-graph-off configuration, the largest
**modeled partial peak is 142.48 GiB/GPU**, on PP rank 2. Communication is assumed
fully initialized and resident from startup; its byte size remains an explicit
sensitivity input.

| Assumed persistent communication/GPU | Largest partial subtotal | Remaining out of 184.35 GiB |
|---|---:|---:|
|0GiB (omitted baseline)|142.48GiB|41.87GiB|
|8GiB|150.48GiB|33.87GiB|
|16GiB|158.48GiB|25.87GiB|
|32GiB|174.48GiB|9.87GiB|

**This does not establish full-model fit.** The remaining budget must cover
additional gathered dense weights, dense optimizer state, recompute/backward
workspaces, residual-cache/deposit allocations beyond counted aliases, loss
intermediates, CUDA/library memory, communication temporaries, and allocator
rounding, cache, fragmentation, and pending frees. The model samples completed
action boundaries and does not capture intra-kernel peaks.

## Actual virtual-stage layout and liveness

The inspected TorchTitan recipe uses the default **two virtual stages per PP
rank**: 16 logical stages across eight physical PP ranks. Rank r owns stages r and
r+8. The recipe found on the cluster had a different batch shape; only its
Interleaved1F1B selection and default stage-splitting policy are reused here.
The estimate preserves the user's **4068, 2×64** inputs.

The native layer split is 5/6/6/6/6/6/6/6/6/6/6/6/6/6/6/4. K3's rank-local
attention-residual cache is enabled, so second-pass wire payloads carry only
blocks missing on the destination rank. Full assembled stacks remain part of
checkpoint inputs; reduced wire payload does not eliminate that storage.

| PP rank | Virtual stages | Decoder layers | Peak live stage/microbatch pairs | Partial simultaneous peak GiB |
|---|---|---|---:|---:|
|0|0,8|0–4;47–52|23|130.34|
|1|1,9|5–10;53–58|21|136.26|
|2|2,10|11–16;59–64|19|142.48|
|3|3,11|17–22;65–70|17|135.28|
|4|4,12|23–28;71–76|15|140.52|
|5|5,13|29–34;77–82|13|131.37|
|6|6,14|35–40;83–88|11|133.35|
|7|7,15|41–46;89–92|9|110.02|

A pair means one microbatch's saved state for **one virtual stage**, not a full
model activation copy. The two virtual stages can hold the same microbatch at
once. Per-stage maxima are not additive: rank 0 separately reaches 16 and 8, but
its simultaneous peak is 23, not 24. Warmup-only forward counts are 22, 20, …, 8;
the next forward before the first backward raises live counts to 23, 21, …, 9.

## What the replay executes and models

We execute the pinned runtime's pure Interleaved1F1B schedule generator and
lowering helpers on CPU, plus the actual TorchTitan layer partition and K3
residual-layout helpers. No process group or GPU allocation is created.
The target runtime is 2.15.0.dev20260928+cu130; TorchTitan source is
b4d5b404bd30dec67f86873203fbd4ba5f457531. Source hashes and generated actions are
saved in [the schedule fixture](../experiments/k3_interleaved_schedule.json).

The lowering uses max_active_stages=2 and resolved unshard_lookahead=2 (the
recipe's auto setting), with defer_pp_recv=False. Consequently the replay
includes receives posted ahead of compute, not just tensors already consumed
by a forward. UNSHARD/RESHARD events establish ordering; their allocation bytes
are still excluded because module-level FSDP lifetimes need separate modeling.

Counted storage includes:

- BF16 sharded parameters, including stage 0 embeddings/vision and stage 15 output.
  Text-only execution omits vision gradients.
- Persistent BF16 expert momentum, assuming a later iteration after initialization.
- Full FP32 gradient allocations after each virtual stage's first backward,
  released at that stage's scheduled REDUCE_GRAD. Reduced local gradients omitted.
- FullAC layer hidden inputs and distinct residual-stack inputs, retained from
  forward through backward. Unchanged stack aliases are counted once per stage
  and microbatch; stacks created by torch.cat get new storage.
- Nonfinal-stage hidden outputs and separately packed residual payloads retained
  in the pipeline forward cache, plus pending forward/backward receive buffers.

Rank 2's sampled peak occurs with 11 live pairs on stage 2 and 8 on stage 10, plus
both stages' accumulated gradients. The report preserves this common-timeline
breakdown; it does not sum independent maxima.

NCCL_RUNTIME_CONNECT changes NCCL connection initialization, not the schedule's
receive-buffer policy. These estimates take no discount for lazy communication
initialization and do not assume =0 and warmed-up =1 have equal footprints.
Communication allowances must be locally owned backing disjoint from already
counted payload tensors, counted once per GPU.

## Reproduce

The saved action fixture allows replay without the GPU runtime:

```bash
PYTHONPATH=src python -m memtracker_nccl.k3_interleaved \
  --schedule experiments/k3_interleaved_schedule.json \
  --sequence 4068 --microbatch-size 2 \
  --output experiments/k3_interleaved_fit.json
```

Regenerate the fixture from the recorded local sources:

```bash
python scripts/extract_interleaved_schedule.py \
  --pytorch-schedules /path/to/recorded/torch/distributed/pipelining/schedules.py \
  --torchtitan-source /path/to/recorded/torchtitan-checkout \
  --output experiments/k3_interleaved_schedule.json
```

The extractor checks pinned source hashes before executing selected AST
functions. It does not import full GPU-dependent packages. Adapting to a different
runtime requires reviewing its allocation/lowering behavior, not merely changing
the version label. Results are in [the storage replay](../experiments/k3_interleaved_fit.json).
No 256-GPU job, or new GPU job of any size, was submitted for this estimate.
