"""CPU/FakeTensor execution of pinned K3 DistMuon planning and tensor kernels.

This is an optimizer-only probe. Actual source planners select owners and packed
routes; source runtime reserves buffers. CUDA stream/event and cross-rank plan
validation are substituted, never a CUDA/NCCL execution claim.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
from dataclasses import replace
from enum import Enum
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed._tools.mem_tracker import _UpdateType

from .tracker import ExtendedMemTracker
from .report_io import read_report, write_report

COMMIT = "b4d5b404bd30dec67f86873203fbd4ba5f457531"
FILES = ["torchtitan/distributed/flex_shard/" + name + ".py" for name in (
    "optimizer_reshard", "_optimizer_reshard_schedule", "_optimizer_reshard_runtime", "dist_muon")]
FILES += ["torchtitan/models/kimi_k3/config_registry.py", "torchtitan/models/kimi_k2_7/config_registry.py",
          "torchtitan/models/kimi_k3/__init__.py", "torchtitan/distributed/fsdp.py",
          "torchtitan/models/common/linear.py", "torchtitan/models/common/config_utils.py",
          "torchtitan/models/kimi_k3/kda.py", "torchtitan/components/optimizer/optimizer.py",
          "torchtitan/models/kimi_k3/model.py", "torchtitan/models/common/multimodal.py",
          "torchtitan/models/kimi_k3/vision_encoder.py", "torchtitan/models/kimi_k2_7/vision_encoder.py",
          "torchtitan/models/common/vision_encoder.py"]


def load_source(source: Path):
    """Import untouched source modules under an isolated package, no CUDA imports."""
    archive = None
    if (source / ".git").exists():
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    else:
        revision = json.loads((source.parent / "job-record.json").read_text())["source_commit"]
        archive = tarfile.open(source.parent / "polyphe-torchtitan-b4d5b404.tar.gz")
    if revision != COMMIT:
        raise ValueError(f"Expected {COMMIT}, got {revision}")
    hashes = {}
    for relative in FILES:
        data = (source / relative).read_bytes()
        original = (archive.extractfile(relative).read() if archive else
                    subprocess.check_output(["git", "show", f"{COMMIT}:{relative}"], cwd=source))
        if data != original:
            raise ValueError(f"Source differs from pinned commit: {relative}")
        hashes[relative] = hashlib.sha256(data).hexdigest()
    if archive:
        archive.close()
    package = "_k3_muon_pinned"
    mod = ModuleType(package)
    mod.__path__ = [str(source / "torchtitan/distributed/flex_shard")]
    sys.modules[package] = mod
    modules = []
    for relative in FILES[:4]:
        name = package + "." + Path(relative).stem
        spec = importlib.util.spec_from_file_location(name, source / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return (*modules, hashes)


def extract_function(source: Path, relative: str, name: str, namespace: dict):
    tree = ast.parse((source / relative).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    # Future annotations avoid importing model/config classes; body is unmodified.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source / relative), "exec"), namespace)
    return namespace[name]


def optimizer_config(source: Path, config):
    class Axes(Enum):
        DP_SHARD = "dp_shard"
        EDP_SHARD = "edp_shard"
        EP = "ep"
    ns = {"ComputeLayout": config.ComputeLayout, "BlockShard": config.BlockShard,
          "Owned": config.Owned, "BucketConfig": config.BucketConfig,
          "Shard": Shard, "MeshAxisName": Axes, "cast": lambda kind, value: value,
          "KimiMLAAttention": NS(Config=object),
          "DistMuon": NS(Config=NS), "AdamW": NS(Config=NS), "OptimizersContainer": NS(Config=NS)}
    extract_function(source, "torchtitan/models/kimi_k2_7/config_registry.py", "_per_expert_compute_layout", ns)
    fn = extract_function(source, "torchtitan/models/kimi_k3/config_registry.py", "_dist_muon_optimizer", ns)
    attention = NS(qk_nope_head_dim=128, qk_rope_head_dim=64, kv_lora_rank=512, v_head_dim=128)
    layers = [NS(attention=attention if i % 4 == 3 or i == 92 else None,
                 delta_attention=NS(head_dim=128) if i % 4 != 3 and i != 92 else None,
                 feed_forward=object() if i == 0 else None) for i in range(93)]
    return fn(NS(first_attention=attention, layers=layers), muon_lr=8e-4, adamw_lr=8e-4,
              parallelism=NS(expert_parallel_degree=32)).optimizers[0]


def muon_inventory() -> list[dict]:
    """Exact matrix shapes from pinned K3/Linear/GroupedLinear config constructors.

    Stacked dense w13 uses [2,F,D], FSDP Shard(1); expert w13 uses
    [E,2,F,D], EP Shard(0). Ordinary dense matrices use FSDP Shard(0).
    Every entry is cross-checked against the executed K3 optimizer config.
    """
    result = []
    for layer in range(93):
        prefix = f"layers.{layer}."
        if layer % 4 == 3 or layer == 92:
            matrices = {"attention.wq_a": (1536,7168), "attention.wq_b": (18432,1536),
                        "attention.wkv_a": (576,7168), "attention.wkv_b": (24576,512),
                        "attention.gate": (12288,7168), "attention.wo": (7168,12288)}
        else:
            matrices = {"delta_attention."+p: (12288,7168) for p in ("q_proj","k_proj","v_proj","output_gate")}
            matrices.update({"delta_attention.forget_a": (128,7168), "delta_attention.forget_b": (12288,128),
                             "delta_attention.output_proj": (7168,12288)})
        if layer == 0:
            matrices.update({"feed_forward.w13": (2,33792,7168), "feed_forward.w2": (7168,33792)})
        else:
            matrices.update({"moe.routed_experts.w13": (896,2,3072,3584),
                             "moe.routed_experts.w2": (896,3584,3072),
                             "moe.router.gate": (896,7168), "moe.shared_experts.w13": (2,6144,7168),
                             "moe.shared_experts.w2": (7168,6144),
                             "moe.routed_down": (3584,7168), "moe.routed_up": (7168,3584)})
        for name, shape in matrices.items():
            expert = "routed_experts" in name
            dim = 1 if name.endswith("w13") and not expert else 0
            result.append(dict(fqn=prefix+name+".weight", layer=layer, shape=list(shape),
                               expert=expert, storage_shard_dim=dim, numel=math.prod(shape)))
    return result


class _Stream:
    def wait_event(self, *args): pass
    def wait_stream(self, *args): pass


class _Event:
    def record(self, *args): pass


class _CPUStreams:
    """Only host scheduling hooks are no-ops. All buffer tensor allocations run."""
    Event = _Event
    _stream = _Stream()
    @staticmethod
    def current_stream(*args): return _CPUStreams._stream
    @staticmethod
    def Stream(**kwargs): return _Stream()
    @staticmethod
    def stream(value): return nullcontext()


class AllocationTrace(ExtendedMemTracker):
    """Actual storage create/delete events, with stable IDs for allocator replay."""
    def __init__(self):
        self.storage_events = []
        self.phase = "inputs"
        self.handles = {}
        self.next_handle = 0
        super().__init__()

    def _update_snap(self, u_type, winfo, *args, **kwargs):
        if u_type in (_UpdateType.ADD, _UpdateType.DEL):
            if u_type == _UpdateType.ADD:
                self.next_handle += 1
                self.handles[winfo] = self.next_handle
            self.storage_events.append(dict(op="alloc" if u_type == _UpdateType.ADD else "free",
                storage_id=self.handles[winfo], bytes=winfo.size*winfo.element_size, phase=self.phase,
                pool="default", stream="compute"))
        super()._update_snap(u_type, winfo, *args, **kwargs)


def kernel_probe(muon, shape, views, reference_shape, *, ns_steps=5):
    """Execute exact prepare, all NS iterations, copy-back, and final update."""
    with FakeTensorMode(allow_fallback_kernels=False):
        tracker = AllocationTrace()
        with tracker:
            grad = torch.empty(shape, dtype=torch.bfloat16)
            momentum = torch.empty_like(grad)
            parameter = torch.empty_like(grad)
            prepared = torch.empty_like(grad)
            baseline = 4*grad.numel()*grad.element_size()
            tracker.phase = "prepare"
            muon._prepare_muon_input(grad, momentum, momentum=.95, nesterov=True, out=prepared)
            tracker.phase = "newton_schulz"
            muon._compute_muon_direction(prepared, matrix_views=views, lr_reference_shape=reference_shape,
                adjust_lr_fn="match_rms_adamw", ns_coefficients=(3.4445,-4.7750,2.0315), ns_steps=ns_steps, eps=1e-7)
            tracker.phase = "final_update"
            muon._apply_muon_update(parameter, prepared, lr=8e-4, weight_decay=0,
                adjust_lr_fn="match_rms_adamw", compute_matrix_shape=reference_shape)
            report = tracker.report()
        return dict(shape=list(shape), views=[dict(shape=list(v.shape), strides=list(v.strides), offset=v.offset) for v in views],
                    input_bytes=baseline, transient_peak_bytes=report["peak"]["cpu"]["tensor_bytes"]-baseline,
                    storage_events=[e for e in tracker.storage_events if e["phase"] != "inputs"],
                    peak=report["peak"]["cpu"], ns_steps=ns_steps)


def rank_plans(plans, participant, schedule):
    """Use source's packed schedule builder for every owner; no guessed splits."""
    result = []
    for plan in plans:
        if isinstance(plan, schedule._LocalBucketPlan):
            result.append(plan)
            continue
        kwargs = dict(redistribution_plans=plan.redistribution_plans, process_group=plan.group.process_group,
                      local_participant=participant)
        result.append(replace(plan, device=torch.device("cpu"), group=replace(plan.group, local_participant=participant),
            storage_to_compute_schedule=schedule._build_packed_all_to_all_schedule(
                **kwargs, direction=schedule._RedistributionDirection.STORAGE_TO_COMPUTE),
            compute_to_storage_schedule=schedule._build_packed_all_to_all_schedule(
                **kwargs, direction=schedule._RedistributionDirection.COMPUTE_TO_STORAGE)))
    return result


def reserved_rows(runtime):
    slots = [("local", runtime._local_slot)]
    if runtime._context:
        slots += [(str(i), s.buffers) for i,s in enumerate(runtime._context.slots)]
    rows = []
    for slot, buf in slots:
        for (_, dtype), reserved in buf.buffers.items():
            for name in ("storage_exchange", "compute_exchange", "compute_scratch", "storage_scratch"):
                tensor = getattr(reserved, name)
                if tensor is not None:
                    rows.append(dict(slot=slot, buffer=name, dtype=str(dtype), bytes=tensor.numel()*tensor.element_size(),
                                     pool="default", stream="compute" if name=="compute_scratch" else "transfer"))
    return rows


def _views(muon, item, partition):
    shape = torch.Size(partition.tensor_shape) if partition is not None else item.param.to_local().shape
    if not shape.numel(): return shape, ()
    if type(item.compute_sharding) is muon.BlockShard:
        start = partition.logical_regions[0].offsets[0] if partition is not None else 0
        return shape, muon._matrix_batch_views_from_shape(shape, matrix_row_sizes=item.compute_sharding.block_sizes,
                                                        logical_row_start=start)
    return shape, (muon._MatrixBatchView(shape, tuple(math.prod(shape[i+1:]) for i in range(len(shape))), 0),)


def packing_probe(plan, runtime, local_specs):
    """Run exact pack/unpack region reshapes and copies, excluding callbacks.

    Collective data are undefined FakeTensor values; source all-to-all plans
    determine the allocated input/output spans. Kernel callbacks are separately
    traced by kernel_probe. This catches reshape copies that size-only plans miss.
    """
    with FakeTensorMode(allow_fallback_kernels=False), patch.object(torch,"get_device_module",return_value=_CPUStreams):
        rt=runtime._BucketedRedistributionRuntime(torch.device("cpu"))
        rt.reserve_buffers([plan],local_tensor_spec=lambda item:(*local_specs[item.fqn][:2],torch.device("cpu")))
        slot=rt._context.slots[0]
        storage,compute=slot.buffers.communication_buffers(plan)
        work=runtime._BucketWork(plan,slot,storage,compute)
        tensors=[getattr(r,name) for r in slot.buffers.buffers.values()
                 for name in ("storage_exchange","compute_exchange","compute_scratch","storage_scratch")
                 if getattr(r,name) is not None]
        tracker=AllocationTrace()
        tracker.track_external(*tensors)
        baseline=sum(t.numel()*t.element_size() for t in tensors)
        tracker.phase="packing"
        with tracker:
            runtime._prepare_redistributed(plan,slot.buffers,storage,prepare=lambda item,out:None)
            runtime._compute_redistributed(work,slot.buffers,compute=lambda item,out:None)
            runtime._finalize_redistributed(work,slot.buffers,finalize=lambda item,out:None)
            peak=tracker.report()["peak"]["cpu"]["tensor_bytes"]
        return dict(transient_peak_bytes=peak-baseline,
                    storage_events=[e for e in tracker.storage_events if e["phase"]=="packing"])


def run(source: Path, stages: list[dict], *, owners=range(32)) -> dict:
    owners = tuple(owners)
    if not owners or len(set(owners)) != len(owners) or any(type(o) is not int or not 0 <= o < 32 for o in owners):
        raise ValueError("owners must contain unique integers in [0,32)")
    if not stages:
        raise ValueError("At least one stage is required")
    config, schedule, runtime, muon, hashes = load_source(source)
    opt_config = optimizer_config(source, config)
    inventory = muon_inventory()
    if {r["fqn"] for r in inventory} != set(opt_config.compute_sharding_by_fqn):
        raise AssertionError("Muon shape inventory and exact source config disagree")
    from torch.testing._internal.distributed.fake_pg import FakeStore
    if dist.is_initialized():
        raise RuntimeError("Run optimizer probe in its own process")
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=32)
    outputs, probes, packing_probes = [], {}, {}
    try:
        dense_mesh = init_device_mesh("cpu", (32,), mesh_dim_names=("dp_shard",))
        expert_mesh = init_device_mesh("cpu", (1,32), mesh_dim_names=("edp_shard","ep"))
        for stage in stages:
            rows = [r for r in inventory if stage["first_layer"] <= r["layer"] <= stage["last_layer"]]
            with patch.object(muon.DistMuon, "_validate_plan_across_ranks"), patch.object(torch, "get_device_module", return_value=_CPUStreams):
                params = []
                for row in rows:
                    shape = row["shape"]
                    local_shape = list(shape)
                    dim = row["storage_shard_dim"]
                    local_shape[dim] = math.ceil(shape[dim]/32)
                    local = torch.empty(local_shape, dtype=torch.bfloat16, device="meta")
                    mesh = expert_mesh if row["expert"] else dense_mesh
                    placements = (Shard(0),Shard(0)) if row["expert"] else (Shard(dim),)
                    params.append(torch.nn.Parameter(DTensor.from_local(local, mesh, placements,
                        shape=torch.Size(shape), stride=tuple(math.prod(shape[i+1:]) for i in range(len(shape))))))
                with patch.object(runtime._BucketedRedistributionRuntime, "reserve_buffers"):
                    optimizer = muon.DistMuon([dict(params=params,param_names=[r["fqn"] for r in rows])],
                        compute_sharding_by_fqn=opt_config.compute_sharding_by_fqn,
                        bucket_configs=opt_config.bucket_configs, adjust_lr_fn="match_rms_adamw")
                # Execute the source momentum constructor for every Muon parameter.
                for item in optimizer._parameter_compute_layouts:
                    item.param.grad = torch.empty_like(item.param)
                    optimizer._momentum(item, item.param.grad)
                momentum_bytes = sum(v["momentum_buffer"].to_local().numel()*2 for v in optimizer.state.values())
                expert_momentum = sum(math.prod(r["shape"])*2//32 for r in rows if r["expert"])
                for owner in owners:
                    plans = rank_plans(optimizer._bucket_plans, owner, schedule)
                    rt = runtime._BucketedRedistributionRuntime(torch.device("cpu"))
                    local_specs = {item.fqn: optimizer._local_tensor_spec(item)
                                   for item in optimizer._parameter_compute_layouts if item.storage_is_compute_ready}
                    def cpu_spec(item):
                        shape, dtype, _ = local_specs[item.fqn]
                        return shape, dtype, torch.device("cpu")
                    with FakeTensorMode(allow_fallback_kernels=False):
                        rt.reserve_buffers(plans, local_tensor_spec=cpu_spec)
                    buffers = reserved_rows(rt)
                    for buffer in buffers:
                        buffer["allocation_id"] = f"muon-s{stage['stage']}-d{owner}-{buffer['slot']}-{buffer['buffer']}"
                    bucket_rows = []
                    redistributed_index = 0
                    for index, plan in enumerate(plans):
                        if isinstance(plan, schedule._LocalBucketPlan):
                            work = [(item,None) for item in plan.items]
                            b = dict(index=index, slot="local", storage_exchange_bytes=0, compute_exchange_bytes=0,
                                     storage_scratch_bytes=0)
                        else:
                            work = [(item,None) for item in plan.unredistributed_items]
                            work += [(item,p.compute_partition(owner)) for item,p in zip(plan.redistributed_items,plan.redistribution_plans)]
                            b = dict(index=index, slot=str(redistributed_index%2),
                                storage_exchange_bytes=2*max(plan.storage_to_compute_schedule.input_buffer_numel,plan.compute_to_storage_schedule.output_buffer_numel),
                                compute_exchange_bytes=2*max(plan.storage_to_compute_schedule.output_buffer_numel,plan.compute_to_storage_schedule.input_buffer_numel),
                                storage_scratch_bytes=2*max(math.prod(p.storage_partition(owner).tensor_shape) for p in plan.redistribution_plans),
                                storage_to_compute_input_splits=list(plan.storage_to_compute_schedule.input_split_sizes),
                                storage_to_compute_output_splits=list(plan.storage_to_compute_schedule.output_split_sizes))
                            redistributed_index += 1
                            packing_key=repr((plan.storage_to_compute_schedule.input_spans_by_parameter,
                                plan.storage_to_compute_schedule.output_spans_by_parameter,
                                tuple((p.storage_partition(owner),p.compute_partition(owner)) for p in plan.redistribution_plans)))
                            packing_key=hashlib.sha256(packing_key.encode()).hexdigest()[:16]
                            if packing_key not in packing_probes:
                                packing_probes[packing_key]=packing_probe(plan,runtime,local_specs)
                            b["packing_probe"]=packing_key
                            b["packing_transient_peak_bytes"]=packing_probes[packing_key]["transient_peak_bytes"]
                        calculations = []
                        for item, partition in work:
                            shape, views = _views(muon,item,partition)
                            if not shape.numel(): continue
                            key = repr((tuple(shape),tuple((tuple(v.shape),v.strides,v.offset) for v in views),tuple(item.lr_reference_shape)))
                            key = hashlib.sha256(key.encode()).hexdigest()[:16]
                            if key not in probes:
                                # Nested FakeTensorMode cannot mix tensors. Helpers use fresh inputs only.
                                probes[key] = kernel_probe(muon,shape,views,item.lr_reference_shape)
                            calculations.append(dict(fqn=item.fqn, compute_shape=list(shape), probe=key,
                                transient_peak_bytes=probes[key]["transient_peak_bytes"]))
                        b["computations"] = calculations
                        b["compute_scratch_bytes"] = max((2*math.prod(x["compute_shape"]) for x in calculations), default=0)
                        b["transient_peak_bytes"] = max(max((x["transient_peak_bytes"] for x in calculations),default=0),
                                                       b.get("packing_transient_peak_bytes",0))
                        bucket_rows.append(b)
                    transient = max(b["transient_peak_bytes"] for b in bucket_rows)
                    persistent = [dict(allocation_id=f"muon-s{stage['stage']}-d{owner}-momentum-{row['fqn']}",
                                       bytes=row["numel"]*2//32, pool="default", stream="compute", kind="momentum",
                                       fqn=row["fqn"], expert=row["expert"]) for row in rows]
                    persistent += [dict(buffer, kind="runtime_reservation") for buffer in buffers]
                    outputs.append(dict(stage=stage["stage"], pp_rank=stage["rank"], dp_owner=owner,
                        momentum_bytes=momentum_bytes, expert_momentum_bytes=expert_momentum,
                        dense_momentum_bytes=momentum_bytes-expert_momentum,
                        bf16_optimizer_gradient_bytes=momentum_bytes,
                        reserved_buffers=buffers, reserved_buffer_bytes=sum(b["bytes"] for b in buffers),
                        persistent_allocations=persistent, transient_peak_bytes=transient, buckets=bucket_rows))
                    del rt
                del optimizer, params
    finally:
        dist.destroy_process_group()
    ranks = []
    for pp_rank in sorted({r["pp_rank"] for r in outputs}):
        for owner in owners:
            selected = [r for r in outputs if (r["pp_rank"],r["dp_owner"])==(pp_rank,owner)]
            ranks.append(dict(pp_rank=pp_rank,dp_owner=owner,
                momentum_bytes=sum(r["momentum_bytes"] for r in selected),
                dense_momentum_bytes=sum(r["dense_momentum_bytes"] for r in selected),
                reserved_buffer_bytes=sum(r["reserved_buffer_bytes"] for r in selected),
                bf16_optimizer_gradient_bytes=sum(r["bf16_optimizer_gradient_bytes"] for r in selected),
                transient_peak_bytes=max(r["transient_peak_bytes"] for r in selected)))
    return dict(schema_version=1,source_commit=COMMIT,source_sha256=hashes,torch_runtime=str(torch.__version__),
        kind="exact source optimizer plans/reservation/kernels with synthetic CPU DTensor storage",
        assumptions=["PP8 VPP2: one independent optimizer per virtual model part, as source OptimizersContainer.",
            "Dense storage is FSDP32; stacked w13 shards matrix rows, expert storage EP32/EDP1.",
            "All Muon parameter and post-reduction optimizer gradients are BF16; momentum is source zeros_like(grad).",
            "FP32 pre-reduction accumulated gradients belong to the training/reduction timeline, not additional optimizer gradients.",
            "Each owner packed schedule is generated from actual global source redistribution plans.",
            "Both virtual-stage runtime reserves persist concurrently; optimizer kernels run sequentially."],
        substitutions=["CPU DeviceMesh and meta DTensor parameter/gradient/momentum storage; FakeStore process group; cross-rank plan hash validation omitted (no numerical collective).",
            "CPU no-op stream/event facade executes exact source reserve_buffers and all tensor allocations.",
            "Source config function executed with attribute-only config objects; matrix shape inventory transcribed from verified source.",
            "Kernel tensor allocations run eagerly in PyTorch 2.14.1 FakeTensor; target CUDA libraries/workspaces not measured."],
        excluded=["AdamW parameters/state/kernels", "CUDA/library/NCCL private workspace and allocator stream delay",
                  "Training activations, FSDP gathers/reductions and pre-reduction FP32 gradients"],
        inventory=inventory,stage_owners=outputs,ranks=ranks,kernel_probes=probes,packing_probes=packing_probes,
        coverage=dict(all_muon_parameter_momentum=True,actual_source_owner_planner=True,
            actual_source_packed_schedules=True,actual_source_runtime_reservation=True,
            actual_source_newton_schulz=True,actual_source_final_update=True,gpu_executed=False,
            actual_source_pack_unpack_tensor_operations=True,
            numerical_collectives_executed=False,full_training_peak=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--schedule",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--owners",default=",".join(map(str,range(32))))
    parser.add_argument("--stages",default=None)
    args=parser.parse_args()
    schedule=read_report(args.schedule)
    stages=schedule["stages"]
    if args.stages is not None:
        selected=set(map(int,args.stages.split(",")))
        stages=[s for s in stages if s["stage"] in selected]
    result=run(args.source,stages,owners=tuple(map(int,args.owners.split(","))))
    write_report(args.output,result)
    print(json.dumps(dict(stage_owners=len(result["stage_owners"]),kernel_probes=len(result["kernel_probes"]),
                         max_reserved_bytes=max(r["reserved_buffer_bytes"] for r in result["ranks"]),
                         max_transient_bytes=max(r["transient_peak_bytes"] for r in result["ranks"]))))


if __name__=="__main__": main()
