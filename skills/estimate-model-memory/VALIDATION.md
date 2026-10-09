# Interpretation and success criteria

## Compare the same quantities

| Quantity | Meaning |
|---|---|
| Requested | Requested live allocator bytes; distinct from block rounding |
| Allocated | Live physical allocator blocks |
| Active | Allocated blocks plus blocks pending stream completion |
| Reserved | Allocator backing, including reusable cache |
| Cached | Reserved minus active; not automatically fragmentation |
| External | Allocations outside the tracked allocator, with stated ownership |
| Resident model peak | Maximum simultaneous reserved plus disjoint external bytes |

The estimate's `phases[].devices` includes `tensor_peak_bytes`,
`requested_allocated_peak_bytes`, `allocated_peak_bytes`, `active_peak_bytes`,
`pending_free_peak_bytes`, `reserved_peak_bytes`, `external_peak_bytes`, and
`resident_peak_bytes`. These independent peaks need not occur together.
Use `resident_peak_bytes` for the modeled simultaneous total. Adding tensor
memory to reserved memory double-counts allocator-managed storage.

Native history allocation sizes are requested bytes. The analyzer uses snapshots
to associate physical blocks with allocation generations. Transient allocations
may die before a snapshot: physical peaks can then be null with lower bounds.
Do not substitute rounded requests for measured physical block sizes. Reused
addresses do not prove reused allocation identities.

External memory sampled at boundaries is not a measured external peak. The
default device-wide residual can include other processes. A process-specific
reader improves scope but still needs matching timestamps. Never add an
independent external maximum to an allocator maximum or call the residual NCCL
without attribution evidence.

## Explain a gap

Compare requested, active, reserved, and external measurements separately. Find
the first phase where they diverge; inspect live allocation origins, stack traces,
ownership inventories, and pending-free events there. A phase that allocated a
buffer may differ from the phase where that buffer contributes to the peak.

- Requested-byte gaps suggest missing storage or incorrect lifetimes/shapes.
- Active/requested differences can involve rounding and deferred frees.
- Reserved/active differences can involve reusable backing, splitting, pool or
  stream separation, and expandable mappings. They are not all fragmentation.
- External gaps require context/library/communicator attribution and isolation.

These are investigation directions, not proofs from aggregate counters alone.
Optimizer `.state` inventory does not include every private buffer; inspect code
and allocation stacks for persistent redistribution and update scratch. Kernel
workspaces may be allocator-managed or external. Count each allocation once.

For NCCL eager/lazy connection experiments, preserve schedule, tensor payloads,
topology and transport settings. Resolve the actual runtime flag spelling and
semantics from that version. Warmup may move allocation earlier and increase
overlap; do not infer an unchanged peak from identical steady-state totals.

## Validation gates

1. **Accounting:** reconcile boundary allocated/active/reserved counters. Reject
   truncated history and mismatches before claiming peak attribution. Check
   allocator schema compatibility in the target runtime.
2. **Identity:** establish equivalent effective config, source including relevant
   patches, runtime, rank mesh, kernels, and logging/reset windows. A base commit
   or container path alone is insufficient proof.
3. **Coverage:** check startup, first lazy optimizer-state creation, repeated
   execution, checkpoint recomputation, overlap, and applicable graph warmup.
   Preserve allocator history between windows when training does so.
4. **Prediction:** compare like-for-like phase/rank metrics with explicit unknowns.
   Agree on tolerances for the intended use before evaluating held-out runs.
   Keep repeated configurations together across calibration/validation splits.
5. **Fit conclusion:** a partial estimate below capacity stays `fit_unproven`.
   A known simultaneous lower bound above capacity establishes modeled failure
   only under its stated assumptions. A successful reduced run does not prove
   full-topology fit. Report the worst covered rank and unobserved ranks.

OOM observations are censored. Failed allocation requests are not live memory;
do not add them to observed successful allocation totals or compute ordinary
successful-step peak error. Source-matched OOM evidence can constrain failure
phase and needed headroom, with allocator/external uncertainty retained.

CPU bookkeeping tests and exact snapshot reconciliation validate those layers;
they do not establish full-model peak accuracy or live GPU-hook compatibility.
Read the checkout's current evidence and reproduce relevant checks before making
claims. Unknown values remain null/unknown rather than becoming zero in tables.

## Concise result format

State scope and verdict first, then use a table such as:

| Rank / phase | Metric | Estimate GiB | Observed GiB | Gap GiB | Coverage |
|---|---|---:|---:|---:|---|
| 0 / optimizer step 1 | Active peak | … | … | … | Matched / partial / unknown |

Gap = estimate minus observed; negative means underestimation. Use relative error
only when a matching, nonzero, uncensored measurement exists. Link raw data and
reproduction commands. Name missing evidence directly; avoid replacing an
unexplained gap with a universal per-model or per-layer correction.
