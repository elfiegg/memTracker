#!/usr/bin/env python3
"""Extract the recorded PP8/VPP2/64-microbatch schedule from trusted local sources.

Executes only selected AST function/class definitions, without importing the full
GPU-dependent source trees. Source hashes accompany the saved action sequence.
"""
import ast, collections, hashlib, json, logging, re, runpy, types, typing
from enum import Enum
from pathlib import Path

import argparse
parser=argparse.ArgumentParser(description='Execute selected pure schedule/layout helpers from explicit local source files; no distributed runtime or GPU.')
parser.add_argument('--pytorch-schedules',type=Path,required=True)
parser.add_argument('--torchtitan-source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
path=args.pytorch_schedules
expected_files={
    path:'61c4e103e3bcb17c11c4dc46db138d2c3400b76e3a8b2d4797cf106e51e931da',
    args.torchtitan_source/'torchtitan/distributed/pipeline_parallel.py':'7fbe51d5e194c44cba0b4939d459bfc0ed432c5a20400d894c7d992d22210755',
    args.torchtitan_source/'torchtitan/models/kimi_k3/pipeline_parallel/layout.py':'5b156f384a5fee38802dfa0f11262d4f8a7e8f58334f6f340971708378b08229',
}
for source, expected in expected_files.items():
    if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
        raise ValueError(f'Unverified source revision: {source}; review before changing the pinned experiment')
text=path.read_text(); tree=ast.parse(text)
names={'_requires_reduce_grad','_ComputationType','_Action','_get_warmup_ops','_get_1f1b_rank_ops','_add_unshard_reshard','_add_reduce_grad','_add_send_recv','_defer_recv_ops'}
ns=dict(Enum=Enum,NamedTuple=typing.NamedTuple,defaultdict=collections.defaultdict,Counter=collections.Counter,logger=logging.getLogger('extract'))
for node in tree.body:
 if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names:
  exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])),str(path),'exec'),ns)
  if node.name=='_ComputationType':
   ns.update({v.name:v for v in ns['_ComputationType']});ns.update(F=ns['FORWARD'],B=ns['FULL_BACKWARD'],I=ns['BACKWARD_INPUT'],W=ns['BACKWARD_WEIGHT'])
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ScheduleInterleaved1F1B')
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_calculate_single_rank_operations')
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])),str(path),'exec'),ns)
ctx=types.SimpleNamespace(n_local_stages=2,pp_group_size=8,microbatches_per_round=8,_n_microbatches=64)
compute={r:ns['_calculate_single_rank_operations'](ctx,r) for r in range(8)}
comms=ns['_add_send_recv']({r:ns['_add_reduce_grad'](ns['_add_unshard_reshard'](a,max_active_stages=2,unshard_lookahead=2),64) for r,a in compute.items()},lambda s:s%8,16)
pipeline=args.torchtitan_source/'torchtitan/distributed/pipeline_parallel.py'
node=next(n for n in ast.parse(pipeline.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_generate_llm_fqn_per_model_part')
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])),str(pipeline),'exec'),ns)
split=ns[node.name](16,93,1,1)
layoutpath=args.torchtitan_source/'torchtitan/models/kimi_k3/pipeline_parallel/layout.py'
layoutns=runpy.run_path(str(layoutpath))
layout=layoutns['infer_block_layout_tables'](stage_to_rank={s:s%8 for s in range(16)},n_layers=93,layers_per_block=12,layer_to_stage=layoutns['layer_to_stage_from_split'](split),cache=True)
stages=[]
for i, fqns in enumerate(split):
 layers=[int(f.split('.')[1]) for f in fqns if f.startswith('layers.')]
 stages.append(dict(stage=i,rank=i%8,first_layer=min(layers),last_layer=max(layers),incoming_delta_blocks=layout.delta_to_send(i-1) if i else [],outgoing_delta_blocks=layout.delta_to_send(i),cached_blocks_at_entry=sorted(layout.cache_at_entry(i)),committed_blocks=layout.commits_at(i)))
result=dict(schema_version=1,torch_runtime='2.15.0.dev20260928+cu130',training_source='b4d5b404bd30dec67f86873203fbd4ba5f457531',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,pipeline,layoutpath)},pp=8,virtual_stages=16,stages_per_rank=2,microbatches=64,defer_pp_recv=False,max_active_stages=2,unshard_lookahead=2,stages=stages,actions={str(r):[str(a) for a in aa] for r,aa in comms.items()},compute_actions={str(r):[str(a) if a else None for a in aa] for r,aa in compute.items()})
out=args.output;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
for r in range(8):
 live=collections.Counter();peaks=collections.Counter();total=0;firstb=None
 for a in compute[r]:
  if a is None:continue
  if a.computation_type==ns['FORWARD']:live[a.stage_index]+=1
  elif a.computation_type==ns['FULL_BACKWARD']:
   if firstb is None:firstb=sum(live.values())
   live[a.stage_index]-=1
  total=max(total,sum(live.values()))
  for k,v in live.items():peaks[k]=max(peaks[k],v)
 print(r,'layers',[(s['first_layer'],s['last_layer']) for s in stages if s['rank']==r], 'peak live',dict(peaks),'simultaneous',total,'before first B',firstb)
