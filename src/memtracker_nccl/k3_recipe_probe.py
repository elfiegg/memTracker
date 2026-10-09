"""Bounded saved-recipe adapter for source-derived K3 optimizer references.

Serialized reprs are never executed. The sole interpreted repr grammar builds
four whitelisted immutable layout descriptors from literals. Geometry comes
from saved configs, with explicit constructor-only states and vision renames.
This is not an execution of the historical base53a45 workload.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import math
import re
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

from torch.distributed.tensor import Shard

from . import k3_adamw_probe as adam_probe
from . import k3_muon_probe as muon_probe


EXTRA_SOURCE_FILES = ["torchtitan/models/" + p for p in (
    "common/moe.py", "kimi_k3/moe.py", "common/decoder.py", "common/nn_modules.py")]


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_geometry(model):
    """Validate the supported constructor tree before any inventory is emitted.

    A missing field must never turn a known trainable module into an ignored
    dictionary. Serialized configs have no class tags, so unknown structures
    fail closed instead of guessing a constructor from whichever fields remain.
    """
    metadata = {"param_init", "sharding_config"}

    def object_fields(value, path, required, optional=()):
        if not isinstance(value, dict):
            raise ValueError(f"Expected constructor config at {path}")
        missing = set(required)-value.keys()
        unknown = value.keys()-set(required)-set(optional)-metadata
        if missing or unknown:
            raise ValueError(f"Unsupported geometry at {path}: missing={sorted(missing)}, unknown={sorted(unknown)}")

    def leaf(value, path, kind):
        specs = {
            "linear": ({"in_features", "out_features", "num_linears"}, {"bias", "group_size"}),
            "norm": ({"normalized_shape", "elementwise_affine", "eps"}, {"bias"}),
            "embedding": ({"num_embeddings", "embedding_dim"}, {"padding_idx"}),
            "conv": ({"in_channels", "out_channels", "kernel_size", "groups", "bias"}, {"stride", "padding", "dilation", "padding_mode"}),
            "kda_norm": ({"dim", "eps"}, set()),
        }
        object_fields(value,path,*specs[kind])

    def composite(value, path, children, scalars=(), optional_scalars=(), nullable=()):
        object_fields(value,path,set(children)|set(scalars),optional_scalars)
        for name, kind in children.items():
            child=value[name]
            if child is None and name in nullable:
                continue
            walk(child,path+"."+name,kind)

    def walk(value,path,kind):
        if kind in ("linear","norm","embedding","conv","kda_norm"):
            return leaf(value,path,kind)
        if kind == "layer":
            children={"attention":"mla","delta_attention":"kda","feed_forward":"ffn","moe":"moe",
                      "attention_norm":"norm","ffn_norm":"norm","attention_res_norm":"norm",
                      "attention_res_proj":"linear","ffn_res_norm":"norm","ffn_res_proj":"linear"}
            composite(value,path,children,("layer_id","attn_res_block_size"),
                      nullable=("attention","delta_attention","feed_forward","moe","attention_res_norm","attention_res_proj"))
            if (value["attention"] is None)==(value["delta_attention"] is None):
                raise ValueError(f"Expected exactly one attention constructor at {path}")
            if (value["feed_forward"] is None)==(value["moe"] is None):
                raise ValueError(f"Expected exactly one feed-forward constructor at {path}")
        elif kind == "mla":
            children={name:"linear" for name in ("wq_a","wq_b","wkv_a","wkv_b","gate","wo")}
            children.update(q_norm="norm",kv_norm="norm")
            composite(value,path,children,("n_heads","dim","kv_lora_rank","qk_nope_head_dim","qk_rope_head_dim","v_head_dim","inner_attention"))
        elif kind == "kda":
            children={name:"linear" for name in ("q_proj","k_proj","v_proj","forget_a","forget_b","beta","output_gate","output_proj")}
            children.update({name:"conv" for name in ("q_conv","k_conv","v_conv")})
            children["output_norm"]="kda_norm"
            composite(value,path,children,("num_heads","head_dim","conv_kernel_size","inner_kda"))
        elif kind in ("ffn","experts"):
            children={"w13":"linear","w2":"linear"}
            scalars=("activation_fn",) if kind=="ffn" else ("activation_fn","token_dispatcher","output_postprocess")
            composite(value,path,children,scalars)
            if kind=="experts":
                if value["output_postprocess"] is not None:
                    raise ValueError(f"Unsupported expert output_postprocess at {path}")
                for name in ("w13","w2"):
                    if "group_size" not in value[name]:
                        raise ValueError(f"Missing grouped-linear group_size at {path}.{name}")
        elif kind == "moe":
            composite(value,path,{"routed_experts":"experts","router":"router","shared_experts":"ffn",
                                 "routed_down":"linear","routed_norm":"norm","routed_up":"linear"},
                      ("num_experts","load_balance_coeff"),nullable=("shared_experts",))
        elif kind == "router":
            composite(value,path,{"gate":"linear"},
                      ("num_experts","score_func","top_k","route_norm","route_norm_epsilon","route_scale","aux_loss"),
                      ("num_bins","_debug_force_load_balance"))
            if value["aux_loss"] is not None:
                raise ValueError(f"Unsupported auxiliary-loss constructor at {path}")
        elif kind == "vision":
            composite(value,path,{"patch_embed_proj":"linear","block":"vision_block","final_norm":"norm","projector":"projector"},
                      ("dim","num_layers","merge_kernel_size","init_pos_emb_height","init_pos_emb_width",
                       "interpolation_mode","rotary_pos_emb","patch_size","in_channels","max_num_frames"))
            object_fields(value["rotary_pos_emb"],path+".rotary_pos_emb",("head_dim","theta"))
        elif kind == "vision_block":
            composite(value,path,{"norm1":"norm","norm2":"norm","attn":"vision_attention","mlp":"vision_mlp"})
        elif kind == "vision_attention":
            composite(value,path,{name:"linear" for name in ("wq","wk","wv","proj")},("dim","num_heads","inner_attention"))
        elif kind == "vision_mlp":
            composite(value,path,{"fc1":"linear","fc2":"linear"},("act_fn",))
        elif kind == "projector":
            composite(value,path,{"linear_1":"linear","linear_2":"linear","post_norm":"norm"},("activation",))
        else:
            raise ValueError(f"Unsupported constructor {kind} at {path}")

    children={"tok_embeddings":"embedding","lm_head":"linear","norm":"norm",
              "output_res_norm":"norm","output_res_proj":"linear","vision_encoder":"vision"}
    composite(model,"model",children,("max_context_length","dim","vocab_size","layers","enable_weight_tying"),
              nullable=("vision_encoder",))
    if not isinstance(model["layers"],list) or not model["layers"]:
        raise ValueError("Expected nonempty model.layers list")
    for index, layer in enumerate(model["layers"]):
        walk(layer,f"model.layers.{index}","layer")


def recipe_inventory(effective_config: dict) -> dict:
    """Return per-parameter inventory, constructor buffers and legacy aggregates.

    Supported geometry is the saved K3 Linear/GroupedLinear/Conv1d/RMSNorm,
    KDA/MLA, latent MoE and optional MoonViT3d. Tied parameters are rejected.
    Initializer/sharding reprs and callable strings are ignored as data.
    """
    model = effective_config["model"]
    if model.get("enable_weight_tying"):
        raise ValueError("Tied parameters require alias-aware inventory support")
    _validate_geometry(model)
    opts = effective_config["optimizer"]["optimizers"]
    if len(opts) != 2 or "compute_sharding_by_fqn" not in opts[0]:
        raise ValueError("Expected saved DistMuon + AdamW optimizer pair")
    muon_names = set(opts[0]["compute_sharding_by_fqn"])
    try:
        patterns = [re.compile(o["pattern"]) for o in opts]
    except (KeyError,TypeError,re.error) as error:
        raise ValueError("Invalid or missing ordered optimizer pattern") from error
    rows, buffers = [], []

    def add(fqn, shape, layer, *, stacked=False):
        shape = [_positive(s, fqn) for s in shape]
        expert = ".routed_experts." in fqn
        selected = next((i for i,pattern in enumerate(patterns) if pattern.search(fqn)),None)
        if selected is None:
            raise ValueError(f"Parameter unclaimed by optimizer patterns: {fqn}")
        if (selected == 0) != (fqn in muon_names):
            raise ValueError(f"Optimizer pattern/layout conflict for {fqn}")
        rows.append(dict(fqn=fqn, layer=layer, shape=shape, numel=math.prod(shape),
                         expert=expert, storage_shard_dim=1 if stacked and not expert else 0,
                         optimizer="muon" if selected == 0 else "adamw"))

    def buffer(fqn, shape, layer, *, dtype="float32", persistent=False):
        shape = [_positive(size,fqn) for size in shape]
        buffers.append(dict(fqn=fqn, layer=layer, shape=list(shape), numel=math.prod(shape),
                            bytes=4*math.prod(shape), dtype=dtype, persistent=persistent,
                            replicated=True))

    def visit(config, prefix, layer):
        if config is None:
            return
        if not isinstance(config, dict):
            raise ValueError(f"Expected object at {prefix}")
        if "in_features" in config and "out_features" in config:
            n = _positive(config.get("num_linears", 1), prefix)
            shape = ([n] if n > 1 else []) + [config["out_features"], config["in_features"]]
            if "group_size" in config:
                shape.insert(0, config["group_size"])
            add(prefix+".weight", shape, layer, stacked=n > 1)
            if config.get("bias", False):
                add(prefix+".bias", ([n] if n > 1 else [])+[config["out_features"]], layer, stacked=n > 1)
            return
        if "num_embeddings" in config:
            add(prefix+".weight", [config["num_embeddings"], config["embedding_dim"]], layer)
            return
        if "normalized_shape" in config:
            if config.get("elementwise_affine", True):
                shape = config["normalized_shape"]
                add(prefix+".weight", [shape] if isinstance(shape, int) else shape, layer)
            if config.get("bias", False):
                raise ValueError("LayerNorm bias configuration unsupported")
            return
        if "in_channels" in config and "out_channels" in config:
            groups = _positive(config["groups"], prefix)
            if config["in_channels"] % groups:
                raise ValueError(f"Invalid grouped convolution at {prefix}")
            kernel = config["kernel_size"]
            add(prefix+".weight", [config["out_channels"], config["in_channels"]//groups]
                + ([kernel] if isinstance(kernel, int) else kernel), layer)
            if config.get("bias", False):
                add(prefix+".bias", [config["out_channels"]], layer)
            return
        if prefix.endswith("delta_attention.output_norm"):
            add(prefix+".weight", [config["dim"]], layer)
            return
        if prefix.endswith("delta_attention"):
            add(prefix+".A_log", [config["num_heads"]], layer)
            add(prefix+".dt_bias", [config["num_heads"],config["head_dim"]], layer)
        if prefix.endswith(".moe"):
            e = config["num_experts"]
            buffer(prefix+".router.tokens_per_expert_E", [e], layer)
            buffer(prefix+".expert_bias_E", [e], layer, persistent=True)
            if "num_bins" in config["router"]:
                buffer(prefix+".router.quantile_balancer.required_bias_histogram_EB",
                       [e,config["router"]["num_bins"]], layer, dtype="int32")
        for key, value in config.items():
            if key in ("param_init", "sharding_config", "inner_attention", "inner_kda",
                       "activation_fn", "token_dispatcher", "output_postprocess", "score_func",
                       "aux_loss", "act_fn", "activation", "rotary_pos_emb"):
                # Explicit source constructors identify these as nonparameter configuration.
                continue
            if isinstance(value, dict):
                # Source VisionMLP uses linear_fc1/2 for saved fc1/2 configs.
                name = {"fc1":"linear_fc1", "fc2":"linear_fc2"}.get(key,key) if prefix.endswith(".mlp") else key
                visit(value, prefix+"."+name, layer)

    for layer, config in enumerate(model["layers"]):
        if config["layer_id"] != layer:
            raise ValueError("Saved layer_ids must be consecutive from zero")
        visit(config, f"layers.{layer}", layer)
    for key in ("tok_embeddings", "lm_head", "norm", "output_res_norm", "output_res_proj"):
        visit(model.get(key), key, None)
    vision = model.get("vision_encoder")
    if vision:
        visit(vision["patch_embed_proj"], "vision_encoder.patch_embed", None)
        add("vision_encoder.pos_embed", [vision["init_pos_emb_height"],vision["init_pos_emb_width"],vision["dim"]], None)
        for i in range(vision["num_layers"]):
            visit(vision["block"], f"vision_encoder.layers.{i}", None)
        visit(vision["final_norm"], "vision_encoder.final_norm", None)
        visit(vision["projector"], "vision_encoder.projector", None)
        buffer("vision_encoder.rotary_pos_emb.inv_freq", [vision["rotary_pos_emb"]["head_dim"]//4], None)
    if {r["optimizer"] for r in rows} != {"muon","adamw"}:
        raise ValueError("Each configured optimizer must claim parameters")
    names = [r["fqn"] for r in rows]
    if len(names) != len(set(names)):
        raise AssertionError("Duplicate parameter names")
    missing = muon_names - set(names)
    if missing:
        raise ValueError(f"Saved Muon names absent from geometry: {sorted(missing)[:3]}")
    grouped = defaultdict(list)
    for row in rows:
        name = (f"layers.{row['layer']}" if row["layer"] is not None else
                "vision_encoder" if row["fqn"].startswith("vision_encoder.") else
                "tok_embeddings" if row["fqn"].startswith("tok_embeddings.") else "output")
        grouped[name].append(row)
    aggregate = []
    for name, group in grouped.items():
        layer = group[0]["layer"]
        selected_buffers = [b for b in buffers if b["layer"] == layer and
                            (layer is not None or b["fqn"].startswith(name+"."))]
        a = dict(name=name, layer=layer,
                 dense_parameters=sum(r["numel"] for r in group if not r["expert"]),
                 expert_parameters=sum(r["numel"] for r in group if r["expert"]),
                 replicated_buffer_bytes=sum(b["bytes"] for b in selected_buffers))
        if layer is not None:
            a["attention"] = "MLA" if model["layers"][layer]["attention"] else "KDA"
        aggregate.append(a)
    return dict(inventory=rows, buffers=buffers, aggregate=aggregate)


def parse_compute_layout(value: str, config):
    """Interpret literals + exact descriptor constructors; no eval/exec/attributes."""
    constructors = {"ComputeLayout": config.ComputeLayout, "BlockShard": config.BlockShard,
                    "Owned": config.Owned, "Shard": Shard}
    allowed_keywords = {"ComputeLayout": {"shardings_by_mesh_axis","shard_order_by_tensor_dim"},
                        "BlockShard": {"dim","block_sizes"}, "Owned": set(), "Shard": {"dim"}}
    if not isinstance(value,str) or len(value) > 4096:
        raise ValueError("Invalid layout descriptor")
    def build(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int,str):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(build(v) for v in node.elts)
        if isinstance(node, ast.Dict):
            return {build(k):build(v) for k,v in zip(node.keys,node.values)}
        if isinstance(node, ast.Call) and isinstance(node.func,ast.Name) and node.func.id in constructors:
            name=node.func.id
            if node.args or any(k.arg not in allowed_keywords[name] for k in node.keywords):
                raise ValueError("Unsupported layout constructor arguments")
            return constructors[name](**{k.arg:build(k.value) for k in node.keywords})
        raise ValueError(f"Unsupported layout syntax: {type(node).__name__}")
    try:
        result=build(ast.parse(value,mode="eval").body)
    except (SyntaxError, TypeError) as error:
        raise ValueError("Invalid saved layout descriptor") from error
    if not isinstance(result,config.ComputeLayout):
        raise ValueError("Expected ComputeLayout descriptor")
    return result


def _verify_extra_source(source):
    """Hash-check constructors beyond the historical probe's verified source list."""
    hashes = {}
    archive = None if (source/".git").exists() else tarfile.open(source.parent/"polyphe-torchtitan-b4d5b404.tar.gz")
    try:
        for relative in EXTRA_SOURCE_FILES:
            data = (source/relative).read_bytes()
            original = (archive.extractfile(relative).read() if archive else
                        subprocess.check_output(["git","show",f"{muon_probe.COMMIT}:{relative}"],cwd=source))
            if data != original:
                raise ValueError(f"Source differs from pinned commit: {relative}")
            hashes[relative] = hashlib.sha256(data).hexdigest()
    finally:
        if archive:
            archive.close()
    return hashes


def run_recipe_probe(effective_config: dict, stages: list[dict], source: Path, *, owners=None) -> dict:
    """Execute source Muon plans/kernels and PyTorch Adam states for saved shapes.

    Boundaries: BF16, DP=EP (EDP1), TP=CP=DPreplicate=1, default supplied
    optimizer math, untied weights. All nine supplied recipes meet these bounds.
    Stage dictionaries use stage/rank/first_layer/last_layer; callers may select
    a subset. The stage containing the final model layer receives final heads.
    """
    source = Path(source)
    p = effective_config["parallelism"]
    dp, ep = p["data_parallel_shard_degree"],p["expert_parallel_degree"]
    if dp != ep or any(p[k] != 1 for k in ("tensor_parallel_degree","context_parallel_degree","data_parallel_replicate_degree")):
        raise ValueError("Recipe adapter requires DP=EP and TP=CP=DP_REPLICATE=1")
    if effective_config["training"]["dtype"] != "bfloat16":
        raise ValueError("Recipe adapter supports BF16 parameter recipes only")
    opts = effective_config["optimizer"]["optimizers"]
    expected_muon = dict(lr=.0008,weight_decay=0.,momentum=.95,nesterov=True,
                         ns_coefficients=[3.4445,-4.775,2.0315],eps=1e-7,ns_steps=5,adjust_lr_fn="match_rms_adamw")
    expected_adam = dict(lr=.0008,betas=[.9,.95],eps=1e-8,weight_decay=0.,foreach=True,fused=False,moment_dtype="parameter")
    for actual, expected in zip(opts,(expected_muon,expected_adam)):
        for key,value in expected.items():
            if actual.get(key) != value:
                raise ValueError(f"Unsupported optimizer setting {key}={actual.get(key)!r}")
    result = recipe_inventory(effective_config)
    owners = tuple(range(dp) if owners is None else owners)
    extra_hashes = _verify_extra_source(source)
    def saved_optimizer(config):
        return SimpleNamespace(compute_sharding_by_fqn={name:parse_compute_layout(v["repr"],config)
            for name,v in opts[0]["compute_sharding_by_fqn"].items()},
            bucket_configs=[config.BucketConfig(patterns=tuple(b["patterns"]),name=b["name"])
                            for b in opts[0]["bucket_configs"]])
    muon = muon_probe.run(source,stages,owners=owners,dp_degree=dp,ep_degree=ep,
                         parameter_inventory=[r for r in result["inventory"] if r["optimizer"]=="muon"],
                         optimizer_config_factory=saved_optimizer, optimizer_kwargs=expected_muon)
    final_layer = len(effective_config["model"]["layers"])-1
    final_stage = next((s["stage"] for s in stages if s["first_layer"]<=final_layer<=s["last_layer"]),None)
    adam = adam_probe.run(source,stages,owners=owners,dp_degree=dp,final_stage=final_stage,
                         include_vision=effective_config["model"].get("vision_encoder") is not None,
                         parameter_inventory=[r for r in result["inventory"] if r["optimizer"]=="adamw"])
    muon["assumptions"][:2] = ["One independent optimizer per supplied pipeline model part.",
                              f"Dense storage FSDP{dp}; expert storage EP{ep}/EDP1; saved compute layouts/buckets."]
    muon["substitutions"][2] = "Saved geometry drives parameter shapes; whitelisted saved layouts drive source planner."
    muon["source_sha256"].update(extra_hashes)
    adam["source_sha256"].update(extra_hashes)
    return dict(schema_version=1,source_commit=muon_probe.COMMIT,kind="saved-config source-derived reference",
                **result,muon=muon,adam=adam,
                coverage=dict(saved_parameter_geometry=True,saved_optimizer_layouts_and_buckets=True,
                              source_executed_optimizer_plans_and_kernels=True,actual_adam_states=True,
                              historical_workload_equivalent=False,gpu_executed=False),
                limitations=["Reference source b4d5b404 differs from historical workload base53a45; not exact workload equivalence.",
                    "Serialized configs omit class tags. Buffer inventory uses KimiLatentMoE tokens+bias; quantile histograms require saved router.num_bins (absent in supplied recipes).",
                    "Vision parameter/state inventory included when configured; vision activations and cached positional tensors excluded.",
                    "CPU FakeTensor eager kernels do not measure CUDA graphs, private workspaces or collective numerical execution."])
