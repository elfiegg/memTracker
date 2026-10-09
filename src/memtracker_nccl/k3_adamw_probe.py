"""Exact-shape BF16 AdamW foreach state and first/steady-step FakeTensor probes.

Non-Muon text-path shapes are transcribed from the hash-verified K3 constructors
and reconciled against the independent layer aggregate minus the Muon inventory.
The default K3 text-only path executes dummy vision and connects it through a
zero-valued dependency. Its zero gradients still initialize AdamW moments.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.tensor import Shard

from .k3_experiment import inventory as aggregate_inventory
from .k3_muon_probe import AllocationTrace, COMMIT, load_source, muon_inventory
from .report_io import read_report, write_report


def adamw_inventory():
    rows=[]
    def add(name,shape,layer):
        rows.append(dict(fqn=name,shape=list(shape),layer=layer,numel=math.prod(shape),storage_shard_dim=0))
    for layer in range(93):
        prefix=f"layers.{layer}."
        for name in ("attention_norm","ffn_norm","ffn_res_norm"):
            add(prefix+name+".weight",(7168,),layer)
        add(prefix+"ffn_res_proj.weight",(1,7168),layer)
        if layer:
            add(prefix+"attention_res_norm.weight",(7168,),layer)
            add(prefix+"attention_res_proj.weight",(1,7168),layer)
            add(prefix+"moe.routed_norm.weight",(3584,),layer)
        if layer%4==3 or layer==92:
            add(prefix+"attention.q_norm.weight",(1536,),layer)
            add(prefix+"attention.kv_norm.weight",(512,),layer)
        else:
            for name in ("q_conv","k_conv","v_conv"):
                add(prefix+"delta_attention."+name+".weight",(12288,1,4),layer)
            add(prefix+"delta_attention.beta.weight",(96,7168),layer)
            add(prefix+"delta_attention.A_log",(96,),layer)
            add(prefix+"delta_attention.dt_bias",(96,128),layer)
            add(prefix+"delta_attention.output_norm.weight",(128,),layer)
    add("tok_embeddings.weight",(163840,7168),None)
    add("lm_head.weight",(163840,7168),None)
    add("norm.weight",(7168,),None)
    add("output_res_norm.weight",(7168,),None)
    add("output_res_proj.weight",(1,7168),None)
    add("vision_encoder.patch_embed.weight",(1024,588),None)
    add("vision_encoder.pos_embed",(64,64,1024),None)
    for layer in range(27):
        prefix=f"vision_encoder.layers.{layer}."
        for name in ("norm1","norm2"): add(prefix+name+".weight",(1024,),None)
        for name in ("wq","wk","wv"): add(prefix+"attn."+name+".weight",(1536,1024),None)
        add(prefix+"attn.proj.weight",(1024,1536),None)
        add(prefix+"mlp.linear_fc1.weight",(4096,1024),None)
        add(prefix+"mlp.linear_fc2.weight",(1024,4096),None)
    add("vision_encoder.final_norm.weight",(1024,),None)
    add("vision_encoder.projector.linear_1.weight",(4096,4096),None)
    add("vision_encoder.projector.linear_2.weight",(7168,4096),None)
    add("vision_encoder.projector.post_norm.weight",(7168,),None)
    dense_muon=defaultdict(int)
    for row in muon_inventory():
        if not row["expert"]: dense_muon[row["layer"]] += row["numel"]
    adam=defaultdict(int)
    for row in rows:
        if row["layer"] is not None: adam[row["layer"]] += row["numel"]
    for row in aggregate_inventory():
        if row["layer"] is not None and adam[row["layer"]] != row["dense_parameters"]-dense_muon[row["layer"]]:
            raise AssertionError(f"Layer {row['layer']} AdamW/Muon shape inventories do not reconcile")
        if row["name"]=="vision_encoder":
            if sum(r["numel"] for r in rows if r["fqn"].startswith("vision_encoder.")) != row["dense_parameters"]:
                raise AssertionError("Vision exact-shape inventory and aggregate disagree")
    return rows


def local_rows(rows,owner,dp_degree=32):
    result=[]
    for row in rows:
        shape=list(row["shape"])
        shape[0],offset=Shard(0)._local_shard_size_and_offset(shape[0],dp_degree,owner)
        result.append(dict(row,local_shape=shape,local_numel=math.prod(shape),shard_offset=offset))
    return result


def _trace_step(optimizer,params,phase):
    tracker=AllocationTrace()
    tracker.phase="baseline"
    tracker.track_external(*params,*[p.grad for p in params])
    for state in optimizer.state.values():
        tracker.track_external(*[value for value in state.values() if isinstance(value,torch.Tensor)])
    baseline=sum(p.numel()*p.element_size()*2 for p in params)
    if optimizer.state:
        baseline+=sum(t.numel()*t.element_size() for state in optimizer.state.values() for t in state.values() if isinstance(t,torch.Tensor))
    tracker.phase=phase
    with tracker:
        optimizer.step()
        snapshot=tracker.report()["peak"]["cpu"]
        states=[]
        for index,p in enumerate(params):
            for name,t in optimizer.state[p].items():
                if not isinstance(t,torch.Tensor): continue
                winfo,_=tracker._WINFO.get(t.untyped_storage(),(None,None))
                states.append(dict(param_index=index,name=name,bytes=t.numel()*t.element_size(),
                                   dtype=str(t.dtype),storage_id=tracker.handles.get(winfo)))
    return dict(baseline_bytes=baseline,peak_bytes=snapshot["tensor_bytes"],
                peak_extra_bytes=snapshot["tensor_bytes"]-baseline,
                states=states,storage_events=[e for e in tracker.storage_events if e["phase"]==phase])


def probe(local_shapes):
    with FakeTensorMode(allow_fallback_kernels=False):
        params=[torch.nn.Parameter(torch.empty(shape,dtype=torch.bfloat16)) for shape in local_shapes]
        for p in params: p.grad=torch.empty_like(p)
        optimizer=torch.optim.AdamW(params,lr=8e-4,betas=(.9,.95),eps=1e-8,weight_decay=0,foreach=True,fused=False)
        first=_trace_step(optimizer,params,"first_step")
        steady=_trace_step(optimizer,params,"steady_step")
        moment_bytes=sum(t.numel()*t.element_size() for state in optimizer.state.values()
                         for name,t in state.items() if name in ("exp_avg","exp_avg_sq"))
        step_bytes=sum(state["step"].numel()*state["step"].element_size() for state in optimizer.state.values())
        return dict(local_shapes=local_shapes,moment_bytes=moment_bytes,host_step_scalar_bytes=step_bytes,
                    bf16_gradient_bytes=sum(p.numel()*p.element_size() for p in params),
                    first_step=first,steady_step=steady)


def run(source: Path,stages,*,owners=range(32),include_vision=True, parameter_inventory=None,
        dp_degree=32, final_stage=15):
    _,_,_,_,hashes=load_source(source)
    rows=adamw_inventory() if parameter_inventory is None else parameter_inventory
    probes={}; outputs=[]
    for stage in stages:
        selected=[row for row in rows if row["layer"] is not None and stage["first_layer"]<=row["layer"]<=stage["last_layer"]]
        if stage["stage"]==0:
            selected += [r for r in rows if r["fqn"]=="tok_embeddings.weight" or
                         (include_vision and r["fqn"].startswith("vision_encoder."))]
        if stage["stage"]==final_stage:
            selected += [r for r in rows if r["layer"] is None and r["fqn"]!="tok_embeddings.weight"
                         and not r["fqn"].startswith("vision_encoder.")]
        for owner in owners:
            local=local_rows(selected,owner,dp_degree)
            shapes=[r["local_shape"] for r in local]
            key=hashlib.sha256(repr(shapes).encode()).hexdigest()[:16]
            if key not in probes: probes[key]=probe(shapes)
            p=probes[key]
            allocations=[dict(allocation_id=f"adam-s{stage['stage']}-d{owner}-{name}-{r['fqn']}",
                kind=name,bytes=2*r["local_numel"],fqn=r["fqn"],pool="default",stream="compute")
                for r in local for name in ("exp_avg","exp_avg_sq") if r["local_numel"]]
            outputs.append(dict(stage=stage["stage"],pp_rank=stage["rank"],dp_owner=owner,probe=key,
                moment_bytes=p["moment_bytes"],host_step_scalar_bytes=p["host_step_scalar_bytes"],
                vision_moment_bytes=sum(4*r["local_numel"] for r in local if r["fqn"].startswith("vision_encoder.")),
                vision_bf16_gradient_bytes=sum(2*r["local_numel"] for r in local if r["fqn"].startswith("vision_encoder.")),
                bf16_optimizer_gradient_bytes=p["bf16_gradient_bytes"],
                steady_step_transient_peak_bytes=p["steady_step"]["peak_extra_bytes"],
                first_step_peak_extra_bytes=p["first_step"]["peak_extra_bytes"],
                persistent_allocations=allocations,local_parameter_inventory=local))
    return dict(schema_version=1,source_commit=COMMIT,source_sha256=hashes,torch_runtime=str(torch.__version__),
        kind="Actual PyTorch AdamW foreach BF16 first/steady optimizer steps on exact local parameter shapes",
        vision_encoder_present=include_vision,
        inventory=rows,stage_owners=outputs,kernel_probes=probes,
        assumptions=["BF16 parameter/optimizer-gradient/moment buffers; scalar step states live on CPU in target noncapturable optimizer.",
            ("Vision encoder present: source text-only path executes a dummy vision forward and add_zero_vision_dependency; zero but non-None gradients initialize all vision moments."
             if include_vision else "Vision encoder explicitly removed: no vision parameters, gradients or moments are modeled."),
            "One AdamW group per model part; foreach=True,fused=False; no AMP grad scaler.",
            "Parameters use source default FSDP Shard(0); actual Shard helper determines uneven and empty shards.",
            "CPU scalar temporaries are conservatively retained in reported optimizer transient trace; GPU amount may be a few bytes smaller."],
        excluded=["CUDA foreach workspace and allocator stream delays", "Vision activations/compute kernels", "FP32 pre-reduction gradients"],
        coverage=dict(actual_torch_adamw_first_and_steady_step=True,source_shape_reconciliation=True,gpu_executed=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--schedule",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--without-vision",action="store_true",help="Only when model configuration explicitly removes the vision encoder")
    args=p.parse_args()
    result=run(args.source,read_report(args.schedule)["stages"],include_vision=not args.without_vision)
    write_report(args.output,result)
    print(json.dumps(dict(stage_owners=len(result["stage_owners"]),distinct_probes=len(result["kernel_probes"]))))


if __name__=="__main__": main()
