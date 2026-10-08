import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import torch

from memtracker_nccl.k3_adamw_probe import adamw_inventory, local_rows, probe
from memtracker_nccl.k3_muon_probe import extract_function

SOURCE=Path(__file__).resolve().parents[2]/"k3-fit/source"


class AdamWProbeTests(unittest.TestCase):
    @unittest.skipUnless(SOURCE.exists(), "Pinned source needed for dependency test")
    def test_exact_zero_vision_dependency_creates_grad_and_adam_state(self):
        helper=extract_function(SOURCE,"torchtitan/models/common/multimodal.py","add_zero_vision_dependency",
            dict(torch=torch,spmd=SimpleNamespace(is_type_checking=lambda:False,local=nullcontext)))
        vision=torch.nn.Linear(3,4,bias=False)
        text=torch.randn(2,4,requires_grad=True)
        helper(text,vision(torch.zeros(1,3))).sum().backward()
        self.assertIsNotNone(vision.weight.grad)
        self.assertEqual(vision.weight.grad.count_nonzero().item(),0)
        optimizer=torch.optim.AdamW(vision.parameters(),foreach=True,fused=False)
        optimizer.step()
        self.assertIn("exp_avg",optimizer.state[vision.weight])

    def test_inventory_reconciles_and_uneven_shards(self):
        rows=adamw_inventory()  # Reconciles every decoder layer against independent inventory.
        row=next(r for r in rows if r["fqn"]=="layers.0.ffn_res_proj.weight")
        self.assertEqual(local_rows([row],0)[0]["local_shape"],[1,7168])
        self.assertEqual(local_rows([row],1)[0]["local_shape"],[0,7168])
        self.assertEqual(len([r for r in rows if r["fqn"].startswith("vision_encoder.")]),222)

    def test_real_foreach_first_and_steady_steps(self):
        p=probe([[3,5],[0,5],[7]])
        self.assertEqual(p["moment_bytes"],2*2*(15+7))
        self.assertEqual(p["host_step_scalar_bytes"],3*4)
        self.assertEqual(p["bf16_gradient_bytes"],2*(15+7))
        self.assertEqual(p["steady_step"]["peak_extra_bytes"],p["bf16_gradient_bytes"])
        self.assertEqual(p["first_step"]["peak_extra_bytes"],
                         p["moment_bytes"]+p["host_step_scalar_bytes"]+p["steady_step"]["peak_extra_bytes"])
        for step in ("first_step","steady_step"):
            live={}; maximum=0
            for event in p[step]["storage_events"]:
                if event["op"]=="alloc": live[event["storage_id"]]=event["bytes"]
                else: live.pop(event["storage_id"])
                maximum=max(maximum,sum(live.values()))
            self.assertEqual(maximum,p[step]["peak_extra_bytes"])
            self.assertEqual(sum(live.values()),p["moment_bytes"]+p["host_step_scalar_bytes"] if step=="first_step" else 0)
        for state in p["first_step"]["states"]:
            if state["name"]!="step": self.assertEqual(state["dtype"],"torch.bfloat16")


if __name__=="__main__": unittest.main()
