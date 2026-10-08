"""Saved full/debug geometry, safe descriptors, and actual DP2 optimizer plans."""
import copy
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from memtracker_nccl.k3_adamw_probe import adamw_inventory, local_rows
from memtracker_nccl.k3_muon_probe import muon_inventory
from memtracker_nccl.k3_recipe_probe import parse_compute_layout, recipe_inventory, run_recipe_probe

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "k3-fit/source"
with gzip.open(ROOT / "experiments/gb200_workload_recipes.json.gz", "rt") as file:
    RECIPES = {str(row["job"]):row["effective_config"] for row in json.load(file)["experiments"]}


class RecipeInventoryTests(unittest.TestCase):
    def test_full_geometry_matches_independent_legacy_inventory(self):
        result = recipe_inventory(RECIPES["3227951"])
        old_muon = muon_inventory()
        old_adam = adamw_inventory()
        legacy = {r["fqn"]:r for r in old_muon+old_adam}
        rows = result["inventory"]
        self.assertEqual(len(rows),2679)
        self.assertEqual({r["fqn"] for r in rows},set(legacy))
        for row in rows:
            for key in ("shape","numel","layer","storage_shard_dim"):
                self.assertEqual(row[key],legacy[row["fqn"]][key],row["fqn"])
        self.assertEqual({r["fqn"] for r in rows if r["optimizer"]=="muon"},
                         {r["fqn"] for r in old_muon})
        for aggregate in result["aggregate"]:
            if aggregate["layer"] is not None:
                selected=[r for r in rows if r["layer"]==aggregate["layer"]]
                self.assertEqual(sum(r["numel"] for r in selected),
                                 aggregate["dense_parameters"]+aggregate["expert_parameters"])

    def test_all_saved_recipes_use_own_shapes_and_vision_presence(self):
        for config in RECIPES.values():
            result=recipe_inventory(config)
            rows={r["fqn"]:r for r in result["inventory"]}
            model=config["model"]
            self.assertEqual(rows["tok_embeddings.weight"]["shape"],[model["vocab_size"],model["dim"]])
            self.assertEqual(any(k.startswith("vision_encoder.") for k in rows),model["vision_encoder"] is not None)
            self.assertEqual({r["layer"] for r in rows.values() if r["layer"] is not None},set(range(len(model["layers"]))))
            expert=model["layers"][1]["moe"]["routed_experts"]["w13"]
            self.assertEqual(rows["layers.1.moe.routed_experts.w13.weight"]["shape"],
                             [expert["group_size"],2,expert["out_features"],expert["in_features"]])
        debug=recipe_inventory(RECIPES["3256150"])
        rows={r["fqn"]:r for r in debug["inventory"]}
        self.assertEqual(len(rows),443)
        self.assertEqual(rows["layers.0.delta_attention.q_proj.weight"]["shape"],[512,256])
        self.assertEqual(rows["layers.3.attention.wq_b.weight"]["shape"],[384,128])
        self.assertEqual(rows["layers.1.moe.routed_experts.w13.weight"]["shape"],[8,2,128,128])

    def test_dp2_empty_and_uneven_adam_shards(self):
        rows=recipe_inventory(RECIPES["3256150"])["inventory"]
        row=next(r for r in rows if r["fqn"]=="layers.0.ffn_res_proj.weight")
        self.assertEqual(local_rows([row],0,2)[0]["local_shape"],[1,256])
        self.assertEqual(local_rows([row],1,2)[0]["local_shape"],[0,256])
        for row in rows:
            if row["optimizer"]=="adamw":
                self.assertEqual(sum(local_rows([row],owner,2)[0]["local_numel"] for owner in range(2)),row["numel"])

    def test_buffers_require_saved_num_bins_and_keep_kimi_bias(self):
        config=copy.deepcopy(RECIPES["3256150"])
        result=recipe_inventory(config)
        self.assertEqual(len(result["buffers"]),32)
        self.assertEqual(sum(b["bytes"] for b in result["buffers"]),16*8*8)
        config["model"]["layers"][1]["moe"]["router"]["num_bins"]=1000
        changed=recipe_inventory(config)
        self.assertEqual(sum(b["bytes"] for b in changed["buffers"])-sum(b["bytes"] for b in result["buffers"]),8*1000*4)

    def test_repr_parser_rejects_code_and_unlisted_calls(self):
        config=SimpleNamespace(ComputeLayout=type("ComputeLayout",(),{}),BlockShard=object,Owned=object)
        for expression in ("__import__('os').system('true')", "ComputeLayout(**{})", "ComputeLayout(x=1)",
                           "(lambda: 1)()", "[Owned() for x in (1,)]", "__import__('os')", "Owned()"):
            with self.subTest(expression=expression),self.assertRaises(ValueError):
                parse_compute_layout(expression,config)

    def test_ordered_optimizer_patterns_must_agree_with_layouts(self):
        for index,pattern,expected in ((0,".*","pattern/layout conflict"),
                                       (0,"(?!)","pattern/layout conflict"),
                                       (1,"(?!)","unclaimed"),
                                       (1,"[","Invalid")):
            config=copy.deepcopy(RECIPES["3256150"])
            config["optimizer"]["optimizers"][index]["pattern"]=pattern
            with self.subTest(index=index,pattern=pattern),self.assertRaisesRegex(ValueError,expected):
                recipe_inventory(config)

    def test_missing_parameter_geometry_and_unknown_modules_fail_closed(self):
        for module_path,field in ((("layers",0,"attention_norm"),"normalized_shape"),
                                  (("layers",0,"delta_attention","beta"),"in_features"),
                                  (("layers",1,"moe","routed_experts","w13"),"group_size"),
                                  (("tok_embeddings",),"embedding_dim"),
                                  ((),"output_res_norm")):
            config=copy.deepcopy(RECIPES["3256150"])
            target=config["model"]
            for key in module_path:
                target=target[key]
            del target[field]
            with self.subTest(path=module_path,field=field),self.assertRaisesRegex(ValueError,"[Mm]issing"):
                recipe_inventory(config)
        for module_path in ((),("layers",0),("layers",1,"moe")):
            config=copy.deepcopy(RECIPES["3256150"])
            target=config["model"]
            for key in module_path:
                target=target[key]
            target["new_trainable_module"]={"dim":256}
            with self.subTest(path=module_path),self.assertRaisesRegex(ValueError,"unknown"):
                recipe_inventory(config)

    def test_tied_parameters_are_explicitly_unsupported(self):
        config=copy.deepcopy(RECIPES["3256150"])
        config["model"]["enable_weight_tying"]=True
        with self.assertRaisesRegex(ValueError,"alias-aware"):
            recipe_inventory(config)


@unittest.skipUnless(SOURCE.exists(),"Pinned source archive required")
class RecipeSourceExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=run_recipe_probe(RECIPES["3256150"],
            [dict(stage=0,rank=0,first_layer=0,last_layer=0),
             dict(stage=7,rank=3,first_layer=15,last_layer=16)],SOURCE)

    def test_dp2_momentum_and_reservations_are_actual_source_plans(self):
        result=self.result
        for row in result["muon"]["stage_owners"]:
            momentum=[a for a in row["persistent_allocations"] if a["kind"]=="momentum"]
            self.assertEqual(sum(a["bytes"] for a in momentum),row["momentum_bytes"])
            self.assertEqual(sum(b["bytes"] for b in row["reserved_buffers"]),row["reserved_buffer_bytes"])
            self.assertGreater(row["transient_peak_bytes"],0)
        for stage in (0,7):
            owners=[r for r in result["muon"]["stage_owners"] if r["stage"]==stage]
            for left,right in zip(owners[0]["buckets"],owners[1]["buckets"]):
                if left["slot"]=="local":
                    continue
                self.assertEqual(left["storage_to_compute_input_splits"][1],right["storage_to_compute_output_splits"][0])
                self.assertEqual(right["storage_to_compute_input_splits"][0],left["storage_to_compute_output_splits"][1])
        self.assertFalse(result["coverage"]["historical_workload_equivalent"])

    def test_final_heads_follow_final_model_layer_and_adam_state_is_real(self):
        for row in self.result["adam"]["stage_owners"]:
            local=row["local_parameter_inventory"]
            names={p["fqn"] for p in local}
            self.assertEqual("lm_head.weight" in names,row["stage"]==7)
            self.assertEqual("tok_embeddings.weight" in names,row["stage"]==0)
            self.assertEqual(row["moment_bytes"],4*sum(p["local_numel"] for p in local))
            probe=self.result["adam"]["kernel_probes"][row["probe"]]
            self.assertTrue(probe["first_step"]["states"])
            self.assertEqual(probe["first_step"]["peak_extra_bytes"],
                row["moment_bytes"]+row["host_step_scalar_bytes"]+row["steady_step_transient_peak_bytes"])

    def test_layouts_are_saved_descriptors(self):
        # Debug MLA row blocks are 64+32, not the hardcoded full-model128+64.
        kernels=self.result["muon"]["kernel_probes"].values()
        view_rows={tuple(v["shape"])[-2] for p in kernels for v in p["views"]}
        self.assertIn(64,view_rows)
        self.assertIn(32,view_rows)


if __name__=="__main__":
    unittest.main()
