import unittest
from dataclasses import replace
from memtracker_nccl.k3_experiment import K3Shape, inventory, simulate


class K3Tests(unittest.TestCase):
    def test_experts_count_all_weights_not_only_active_topk(self):
        all_experts = sum(r["expert_parameters"] for r in inventory())
        fewer_active = sum(r["expert_parameters"] for r in inventory(replace(K3Shape(), top_k=1)))
        self.assertEqual(all_experts, fewer_active)
        self.assertEqual(all_experts, 92 * 896 * 3 * 3584 * 3072)
        rows = inventory()
        self.assertEqual(sum(r.get("attention") == "KDA" for r in rows), 69)
        self.assertEqual(sum(r.get("attention") == "MLA" for r in rows), 24)

    def test_ep_overlays_dp_and_does_not_reduce_total_state_again(self):
        a = simulate(pp=1, dp=64, ep=8, prefetch=1, nccl_mib_per_communicator=0)
        b = simulate(pp=1, dp=64, ep=16, prefetch=1, nccl_mib_per_communicator=0)
        self.assertEqual(a["stages"][0]["state_bytes"], b["stages"][0]["state_bytes"])
        self.assertLess(b["stages"][0]["partial_envelope_bytes"], a["stages"][0]["partial_envelope_bytes"])
        self.assertIsNone(a["coverage"]["full_training_peak_bytes"])

    def test_external_allowance_increases_simultaneous_peak(self):
        a = simulate(pp=1, nccl_mib_per_communicator=0)
        b = simulate(pp=1, nccl_mib_per_communicator=16)
        self.assertEqual(b["stages"][0]["partial_envelope_bytes"] - a["stages"][0]["partial_envelope_bytes"], 3*16*2**20)

    def test_invalid_mesh_rejected(self):
        with self.assertRaises(ValueError):
            simulate(dp=63, ep=8)


if __name__ == "__main__":
    unittest.main()
