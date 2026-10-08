"""Compose known K3 allocations on the source Interleaved1F1B action sequence.

This is a storage-lifetime model, not a full CUDA execution. In particular, one
exact attention-residual helper is replayed per layer; the rest of each layer's
activation graph is explicitly uncovered. A below-capacity result is not a fit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path

from .allocator_model import AllocatorConfig, CachingAllocatorModel
from .k3_experiment import inventory
from .k3_interleaved import parse_action, run as boundary_report
from .report_io import read_report, write_report

GiB = 2**30


class Ledger:
    def __init__(self, expandable):
        self.allocator = CachingAllocatorModel(AllocatorConfig(expandable_segments=expandable))
        self.live = {}
        self.categories = Counter()
        self.peak = dict(bytes=0)
        self.reserved_peak = dict(bytes=0)
        self.event = "initialize"
        self.sequence = 0
        self.phase = "initialization"
        self.phase_peaks = {}

    def sample(self):
        self.sequence += 1
        total = sum(self.categories.values())
        # Avoid constructing synthetic addresses for every block at every event.
        reserved = self.allocator.reserved_bytes()
        if total > self.peak["bytes"]:
            self.peak = dict(bytes=total, event=self.event, categories=dict(self.categories))
        if total > self.phase_peaks.get(self.phase, {}).get("bytes", -1):
            self.phase_peaks[self.phase] = dict(bytes=total, event=self.event, categories=dict(self.categories))
        if reserved > self.reserved_peak["bytes"]:
            self.reserved_peak = dict(bytes=reserved, event=self.event, live_bytes=total,
                                      categories=dict(self.categories))

    def alloc(self, name, size, category, stream="compute"):
        if name in self.live:
            raise ValueError(f"Duplicate allocation {name}")
        self.live[name] = (size, category)
        self.categories[category] += size
        self.allocator.allocate(name, size, stream=stream)
        self.sample()

    def free(self, name):
        size, category = self.live.pop(name)
        self.categories[category] -= size
        self.allocator.free(name)
        self.sample()

    def trace(self, events, prefix, *, phases=None, category="attention_residual"):
        allocated = set()
        for event in events:
            name = prefix + "/" + str(event["storage_id"])
            if event["op"] == "alloc" and event["phase"] != "inputs" and (phases is None or event["phase"] in phases):
                self.alloc(name, event["bytes"], category)
                allocated.add(name)
            elif event["op"] == "free" and name in allocated:
                self.free(name); allocated.remove(name)
        # Outputs/saved values of the isolated helper are consumed by the rest
        # of the block, which is outside this probe. Do not retain into next MB.
        for name in sorted(allocated):
            self.free(name)


def run(schedule, muon, residual, adam, *, expandable=True, communication_gib=16,
        sequence=4068, microbatch_size=2, capacity_bytes=197940150272):
    if sequence * microbatch_size != residual["tokens"] or residual["dim"] != 7168:
        raise ValueError("Residual probe shape does not match training tokens/dim")
    if isinstance(communication_gib, bool) or not math.isfinite(communication_gib) or communication_gib < 0:
        raise ValueError("communication_gib must be nonnegative")
    if not (muon["source_commit"] == adam["source_commit"] == residual["source_commit"] == schedule["training_source"]):
        raise ValueError("Source probe revisions must match the schedule")
    if not adam.get("vision_encoder_present"):
        raise ValueError("Full-model recipe requires the vision encoder and its AdamW states")
    baseline = boundary_report(schedule, sequence=sequence, microbatch_size=microbatch_size,
                               capacity_bytes=capacity_bytes)
    stage_info = {s["stage"]: s for s in baseline["stages"]}
    item_info = {r["layer"]: r for r in inventory() if r["layer"] is not None}
    probes = {r["residual_entries"]: r for r in residual["probes"] if not r["full_ac"]}
    forward_probes = {r["residual_entries"]: r for r in residual["probes"] if r["full_ac"]}
    h = sequence * microbatch_size * 7168 * 2
    ranks = []
    for rank in range(8):
        owners = [r for r in muon["ranks"] if r["pp_rank"] == rank]
        if {r["dp_owner"] for r in owners} != set(range(32)):
            raise ValueError("Need all 32 Muon owner ranks for each PP rank")
        # Dense momentum/state are owner-independent. Runtime reserve differs.
        owner = max(owners, key=lambda r: r["reserved_buffer_bytes"])["dp_owner"]
        chosen = [r for r in muon["stage_owners"] if r["pp_rank"] == rank and r["dp_owner"] == owner]
        if len(chosen) != 2:
            raise ValueError("Need both virtual stages per physical PP rank")
        ledger = Ledger(expandable)
        layers = {}; groups = {}; pending = set(); gathered = set(); active = {}; gradients = set()
        local_gradients = Counter()
        def group_for_parameter(st, fqn):
            if fqn.startswith("layers."):
                return (st, int(fqn.split(".")[1]))
            if fqn.startswith("vision_encoder."):
                return (st, "vision_encoder")
            return (st, "tok_embeddings" if fqn.startswith("tok_embeddings.") else "output")
        adam_chosen = {(r["stage"], r["dp_owner"]): r for r in adam["stage_owners"]}
        for m in chosen:
            st = m["stage"]; info = stage_info[st]
            a = adam_chosen[(st, owner)]
            layers[st] = list(range(info["first_layer"], info["last_layer"]+1))
            for allocation in m["persistent_allocations"]:
                if allocation["kind"] == "momentum":
                    name = allocation["allocation_id"]
                    ledger.alloc(f"params/{name}", allocation["bytes"], "sharded_parameters")
                    ledger.alloc(f"momentum/{name}", allocation["bytes"], "muon_momentum")
                    local_gradients[group_for_parameter(st, allocation["fqn"])] += allocation["bytes"]
            for parameter in a["local_parameter_inventory"]:
                ledger.alloc(f"params/{st}/{parameter['fqn']}", 2*parameter["local_numel"], "sharded_parameters")
                local_gradients[group_for_parameter(st, parameter["fqn"])] += 2*parameter["local_numel"]
            for i, buf in enumerate(m["reserved_buffers"]):
                stream = "compute" if buf["buffer"] == "compute_scratch" else "muon_transfer"
                ledger.alloc(f"reserve/{st}/{i}", buf["bytes"], "muon_reserved_buffers", stream)
            for layer in layers[st]:
                row = item_info[layer]
                # Distinct output copies even at EDP1: target wait_for_unshard
                # calls alloc_storage + tensor.copy_(all_gather_input).
                groups[(st, layer)] = (row["dense_parameters"]*2, row["expert_parameters"]//32*2)
                ledger.alloc(f"router/{layer}", row["replicated_buffer_bytes"], "router_buffers")
            extras = [r for r in inventory() if r["layer"] is None and
                      ((st == 0 and r["name"] in ("tok_embeddings", "vision_encoder")) or (st == 15 and r["name"] == "output"))]
            for row in extras:
                groups[(st, row["name"])] = (row["dense_parameters"]*2, 0)
            for allocation in a["persistent_allocations"]:
                ledger.alloc(allocation["allocation_id"], allocation["bytes"], "adamw_moments")

        def start_gather(key):
            if key in gathered or key in pending:
                return
            pending.add(key)
            dense, _ = groups[key]
            ledger.alloc(f"gather_flat/{key}", dense, "fsdp_gather_staging", "fsdp_gather")

        def wait_gather(key):
            if key in gathered:
                return
            start_gather(key)
            dense, expert = groups[key]
            ledger.alloc(f"gathered/{key}", dense+expert, "fsdp_unsharded_weights")
            ledger.free(f"gather_flat/{key}")
            pending.remove(key); gathered.add(key)

        def release_gather(key):
            if key in gathered:
                ledger.free(f"gathered/{key}"); gathered.remove(key)
            if key in pending:
                ledger.free(f"gather_flat/{key}"); pending.remove(key)

        ledger.phase = "training"
        for position, action in enumerate(schedule["actions"][str(rank)]):
            st, op, mb = parse_action(action); info = stage_info[st]
            ledger.event = action
            key = (st, mb)
            stage_groups = [g for g in groups if g[0] == st]
            if op in ("RECV_F", "RECV_B"):
                size = info["forward_receive_bytes" if op == "RECV_F" else "backward_receive_bytes"]
                ledger.alloc(f"{op}/{key}", size, "pipeline_receive", "pipeline")
            elif op == "UNSHARD":
                for group in stage_groups:
                    start_gather(group)
            elif op == "RESHARD":
                for group in stage_groups:
                    release_gather(group)
            elif op == "F":
                if st:
                    ledger.free(f"RECV_F/{key}")
                # Pipeline _assert_unsharded waits each explicitly scheduled
                # stage-wide handle before executing the stage.
                for group in stage_groups:
                    if group in pending:
                        wait_gather(group)
                if st == 0:
                    wait_gather((st, "vision_encoder"))
                    wait_gather((st, "tok_embeddings"))
                    release_gather((st, "tok_embeddings"))
                refs = Counter((layer+11)//12 for layer in layers[st])
                active[key] = refs
                for layer in layers[st]:
                    ledger.event = f"{action}/layer{layer}/forward"
                    prefix = (layer+11)//12
                    ledger.alloc(f"hidden/{key}/{layer}", h, "checkpoint_inputs")
                    stack_name = f"stack/{key}/{prefix}"
                    if stack_name not in ledger.live:
                        ledger.alloc(stack_name, prefix*h, "checkpoint_inputs")
                    wait_gather((st, layer))
                    index = layers[st].index(layer)
                    if index+1 < len(layers[st]):
                        start_gather((st, layers[st][index+1]))
                    probe = forward_probes[layer//12+1]
                    ledger.trace(probe["storage_events"], f"helper/{position}/{layer}", phases={"forward"})
                    release_gather((st, layer))  # reshard_after_forward='always'
                for group in stage_groups:
                    if not isinstance(group[1], int) and group[1] != "tok_embeddings":
                        wait_gather(group); release_gather(group)
                ledger.alloc(f"output/{key}", info["retained_boundary_bytes"]-info["checkpoint_input_bytes"], "pipeline_output")
            elif op == "B":
                # FP32 accumulated gradients remain until REDUCE_GRAD. Allocate
                # each layer on its first backward, rather than all at stage entry.
                for group in stage_groups:
                    if group in pending:
                        wait_gather(group)
                if st == 0:
                    wait_gather((st, "vision_encoder"))
                if st == 15:
                    wait_gather((st, "output"))
                    if (st, "output") not in gradients:
                        ledger.alloc(f"grad/{st}/output", 2*sum(groups[(st, "output")]), "fp32_accumulated_gradients")
                        gradients.add((st, "output"))
                refs = active[key]
                for layer in reversed(layers[st]):
                    ledger.event = f"{action}/layer{layer}/recompute_backward"
                    wait_gather((st, layer))
                    ledger.trace(probes[layer//12+1]["storage_events"], f"helper/{position}/{layer}")
                    if (st, layer) not in gradients:
                        dense, expert = groups[(st, layer)]
                        ledger.alloc(f"grad/{st}/{layer}", 2*(dense+expert), "fp32_accumulated_gradients")
                        gradients.add((st, layer))
                    ledger.free(f"hidden/{key}/{layer}")
                    prefix = (layer+11)//12; refs[prefix] -= 1
                    if not refs[prefix]:
                        ledger.free(f"stack/{key}/{prefix}")
                    # set_reshard_after_backward(False): full parameters persist.
                for group in stage_groups:
                    if not isinstance(group[1], int):
                        wait_gather(group)
                        if group not in gradients:
                            ledger.alloc(f"grad/{st}/{group[1]}", 2*sum(groups[group]), "fp32_accumulated_gradients")
                            gradients.add(group)
                if st != 15:
                    ledger.free(f"RECV_B/{key}")
                ledger.free(f"output/{key}"); del active[key]
            elif op == "REDUCE_GRAD":
                # The local BF16 result coexists with the FP32 input. Collective
                # packing/casts/reduction intermediates are still uncovered.
                for group in sorted(gradients, key=str):
                    if group[0] == st:
                        release_gather(group)
                        ledger.alloc(f"optimizer_grad/{group}", local_gradients[group], "bf16_optimizer_gradients")
                        ledger.free(f"grad/{st}/{group[1]}"); gradients.remove(group)
        if active or pending or gathered or gradients:
            raise ValueError("Schedule failed to drain modeled training lifetimes")
        training_peak = dict(ledger.peak)
        after_schedule_categories = dict(ledger.categories)
        ledger.phase = "optimizer"
        for m in chosen:
            for bucket in m["buckets"]:
                for computation in bucket["computations"]:
                    ledger.event = f"optimizer/stage{m['stage']}/{computation['fqn']}"
                    ledger.trace(muon["kernel_probes"][computation["probe"]]["storage_events"],
                                 "muon_kernel", category="muon_kernel_temporary")
            a = adam_chosen[(m["stage"], owner)]
            ledger.event = f"optimizer/stage{m['stage']}/adamw_foreach"
            ledger.trace(adam["kernel_probes"][a["probe"]]["steady_step"]["storage_events"],
                         "adamw_kernel", category="adamw_kernel_temporary")
        comm = int(communication_gib*GiB)
        ranks.append(dict(pp_rank=rank, dp_owner=owner, training_live_peak=training_peak,
                          phase_peaks=ledger.phase_peaks, after_schedule_live_by_category=after_schedule_categories,
                          modeled_live_peak=ledger.peak, modeled_reserved_peak=ledger.reserved_peak,
                          live_plus_communication_bytes=ledger.peak["bytes"]+comm,
                          reserved_plus_communication_bytes=ledger.reserved_peak["bytes"]+comm,
                          capacity_bytes=capacity_bytes, remaining_live_bytes=capacity_bytes-ledger.peak["bytes"]-comm,
                          verdict="modeled_live_exceeds_capacity" if ledger.peak["bytes"]+comm>capacity_bytes else "fit_unproven",
                          communication_sensitivity=[dict(communication_gib=c,
                              live_plus_communication_bytes=ledger.peak["bytes"]+c*GiB,
                              remaining_bytes=capacity_bytes-ledger.peak["bytes"]-c*GiB) for c in (0,8,16,32)]))
    return dict(schema_version=1, scenario=baseline["scenario"], allocator_mode="expandable" if expandable else "fixed",
                communication_gib=communication_gib, ranks=ranks, full_model_fit_verified=False,
                coverage=dict(muon_momentum=True, muon_persistent_redistribution_buffers=True,
                              muon_newton_schulz_and_update=True, adamw_bf16_moments=True,
                              fsdp_dense_and_edp1_expert_unsharded_copies=True,
                              fullac_checkpoint_input_release=True, attention_residual_helper=True,
                              allocator_lifetime_replay=True),
                assumptions=["Worst persistent-buffer Muon DP owner selected independently for each PP rank.",
                             "FSDP layer groups model source UNSHARD, wait/copy-out, forward reshard and backward retention.",
                             "EDP1 copies expert parameters in target wait_for_unshard despite requiring no network all-gather.",
                             "FullAC checkpoint inputs freed per layer in backward; unchanged residual stacks share storage.",
                             "One exact attention-residual helper per layer: original forward temporaries released, backward recreates them.",
                             "FullAC whole-block recompute may retain multiple helpers and other kernel tensors concurrently.",
                             "Allocator frees complete immediately within explicit compute/gather/pipeline/transfer domains.",
                             "Communication is an uncalibrated aggregate allowance resident throughout, disjoint from tensor payloads.",
                             "No allocator allocation-failure retry/empty-cache simulation; reserved above capacity alone is not OOM proof."],
                excluded=["Full MLA/KDA/MoE activation graph, kernel-internal workspaces, registered FullAC effects",
                          "Residual cache/deposit lifetimes beyond existing checkpoint/wire aliases",
                          "Loss logits/intermediates and root-only small parameters beyond inventory aggregates",
                          "FSDP shard-reorder temporaries, exact padding, reduction packing/casts and delayed stream frees",
                          "Vision forward/backward activation temporaries; CUDA/library external allocations"],
                source_commit=muon["source_commit"],
                source_sha256={"target_fsdp_param_group.py": "4901c6bd3ededb5fe4ac33e5ef6c03283139386bb91247be726f53cab4b7dd64",
                               "target_fsdp_collectives.py": "e7e9d6be8cb35969b4665ef58b7d9a8c3ed68cacce85f544928294d464513118",
                               "target_pipeline_stage.py": "bb0ece062dd3200b6d666b3be1ea9c1f4cb85cb9b556f069d94f53e38611f717",
                               **schedule["source_sha256"]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule",type=Path,required=True)
    parser.add_argument("--muon",type=Path,required=True)
    parser.add_argument("--residual",type=Path,required=True)
    parser.add_argument("--adam",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allocator",choices=("fixed","expandable"),default="expandable")
    parser.add_argument("--communication-gib",type=float,default=16)
    args=parser.parse_args()
    result=run(read_report(args.schedule),read_report(args.muon),
               read_report(args.residual),read_report(args.adam),expandable=args.allocator=="expandable",
               communication_gib=args.communication_gib)
    write_report(args.output,result)
    for row in result["ranks"]:
        print(row["pp_rank"], row["dp_owner"], row["live_plus_communication_bytes"]/GiB,
              row["reserved_plus_communication_bytes"]/GiB, row["modeled_live_peak"]["event"])


if __name__=="__main__":
    main()
