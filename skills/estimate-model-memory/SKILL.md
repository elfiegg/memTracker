---
name: estimate-model-memory
description: Estimates training memory with memTracker's FakeTensor, caching allocator, communication models, and TorchTitan phase capture. Use when estimating whether a model fits GPU memory, comparing pipeline or optimizer configurations, explaining allocated versus reserved memory, or validating predictions against training logs and CUDA captures.
---

# Estimate model memory

Use `elfiegg/memTracker`, branch `feat/allocator-overhead-experiments`:
https://github.com/elfiegg/memTracker/tree/feat/allocator-overhead-experiments

The generic core executes tensor lifetimes and models allocator backing. The
TorchTitan adapter captures real execution phases. There is no universal CLI
that converts an arbitrary TorchTitan config into a complete distributed estimate.
The saved K3 workload adapter is a separate, partial, source-pinned implementation.

## Workflow

1. Locate the checkout and read its `README.md` and applicable `AGENTS.md`.
   Record branch, commit, dirty state, and simulator runtime. Use an existing
   suitable checkout or a separate clone; preserve unrelated changes. Verify
   the interfaces below against that revision before running commands.
2. Collect the effective model/training config and source. Prefer existing run
   artifacts; ask only for missing inputs that materially affect the result.
   Read [INPUTS.md](INPUTS.md) for the required configuration and identity fields.
3. Choose the supported route in [WORKFLOWS.md](WORKFLOWS.md):
   - Generic FakeTensor harness for executable model/optimizer code.
   - Real TorchTitan phase capture for measured lifetimes and attribution.
   - Historical workload replay only for the adapter's supported configurations.
   State which components each route covers before interpreting a total.
4. Execute CPU estimates first. Include initialization, first optimizer update,
   and repeated steps; continue until the relevant lifetime pattern is covered.
   Preserve allocation state across phases and match checkpoint/gradient lifetimes.
   Use the actual schedule for each PP rank and virtual stage.
5. Model allocator and external memory with explicit provenance and lifetimes.
   Read [VALIDATION.md](VALIDATION.md) before combining counters. Unsupported
   operations and unmeasured components stay explicit unknowns, never zero.
6. If measurement is needed, use existing traces or a small authorized GPU probe
   in the target runtime. Do not request a full training allocation merely to
   gauge memory. A reduced topology validates components, not full-topology fit.
7. Compare matching ranks, phases, reset windows, and configurations. Separate
   calibration recipes from held-out validation recipes. Retain raw captures,
   predictions, effective configs, identities, and comparison reports.
8. Report the peak phase/rank, covered components, remaining gaps, and conclusion:
   measured result, conditional modeled result, or `fit_unproven`. Partial memory
   below capacity does not establish fit. Keep the user-facing report concise.

## Quick smoke check

From the repository root, with its compatible CPU PyTorch environment:

```sh
PYTHONPATH=src python examples/framework_phase_demo.py \
  --mode fake --family mlp --width 32 --batch 2 --steps 3 \
  --full-ac --expandable --output /tmp/memtracker-smoke.json.gz
```

This checks the tooling on a small fixture; it does not estimate the user's model.
Use the real model code and effective config for that estimate.

## Required accounting rules

- Distinguish tensors, rounded allocated blocks, active blocks awaiting reuse,
  reserved backing, and external allocations. Do not sum independent peaks.
- Include the actual optimizer's state, update scratch, persistent private
  buffers, and owner-rank communication. An optimizer name is insufficient.
- Derive FullAC, FSDP gather/prefetch, residual, and PP buffer lifetimes from code
  or traces. Interleaved 1F1B does not imply all microbatches are live together.
- Resolve an application's segmentation flag to the actual allocator setting.
  `AllocatorConfig(expandable_segments=True)` models native expandable segments;
  it is an approximation, not a version-selected CUDA allocator implementation.
- NCCL payload tensors are already tensor allocations. External memory can also
  include context, libraries, and other processes; do not label all of it NCCL.
- Keep models framework/config driven. Do not insert model-name constants or
  fit unexplained offsets to the same workloads used to claim validation.

## Deliverable

Provide reproducible commands and artifact paths, plus a compact table:
`rank / phase | metric | estimate | observed | gap | coverage`.
Use GiB or MiB consistently; define gap as estimate minus observed. For OOMs,
report the failed phase and successful allocations before failure, not an
ordinary successful-step accuracy percentage. State what evidence would close
each material unknown.
