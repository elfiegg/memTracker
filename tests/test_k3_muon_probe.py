"""Optimizer coverage, packed conservation, lifetimes and source reservation tests."""
import math
from pathlib import Path
import unittest

from memtracker_nccl.k3_muon_probe import muon_inventory, run


SOURCE = Path(__file__).resolve().parents[2] / "k3-fit/source"


class InventoryTests(unittest.TestCase):
    def test_all_layers_and_stacked_projections(self):
        rows = muon_inventory()
        self.assertEqual({r["layer"] for r in rows}, set(range(93)))
        self.assertEqual(len({r["fqn"] for r in rows}), len(rows))
        experts = [r for r in rows if r["expert"]]
        self.assertEqual(len(experts), 92*2)
        self.assertEqual(sum(r["numel"] for r in experts), 92*896*3*3072*3584)
        dense_w13 = next(r for r in rows if r["fqn"] == "layers.0.feed_forward.w13.weight")
        self.assertEqual(dense_w13["shape"], [2,33792,7168])
        self.assertEqual(dense_w13["storage_shard_dim"], 1)


@unittest.skipUnless(SOURCE.exists(), "Pinned TorchTitan archive required for source execution")
class SourceExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run(SOURCE, [dict(stage=0,rank=0,first_layer=0,last_layer=4)], owners=range(32))

    def test_momentum_covers_dense_and_experts(self):
        for row in self.result["stage_owners"]:
            self.assertGreater(row["dense_momentum_bytes"], 0)
            self.assertEqual(row["momentum_bytes"], row["dense_momentum_bytes"]+row["expert_momentum_bytes"])
            momentum = [a for a in row["persistent_allocations"] if a["kind"]=="momentum"]
            self.assertEqual(sum(a["bytes"] for a in momentum),row["momentum_bytes"])
            self.assertEqual(row["bf16_optimizer_gradient_bytes"],row["momentum_bytes"])

    def test_all_to_all_bytes_conserved_across_owners(self):
        owners = self.result["stage_owners"]
        for i in range(len(owners[0]["buckets"])):
            buckets = [r["buckets"][i] for r in owners]
            if buckets[0]["slot"] == "local": continue
            self.assertEqual(sum(b["storage_exchange_bytes"] for b in buckets),
                             sum(b["compute_exchange_bytes"] for b in buckets))
            for source in range(32):
                for target in range(32):
                    self.assertEqual(buckets[source]["storage_to_compute_input_splits"][target],
                                     buckets[target]["storage_to_compute_output_splits"][source])

    def test_source_slot_reservation_matches_requirements(self):
        for row in self.result["stage_owners"]:
            for buffer in row["reserved_buffers"]:
                slot = buffer["slot"]
                expected = max(b[buffer["buffer"]+"_bytes"] for b in row["buckets"] if b["slot"]==slot)
                self.assertEqual(buffer["bytes"],expected)
            self.assertEqual(sum(a["bytes"] for a in row["persistent_allocations"]),
                             row["momentum_bytes"]+row["reserved_buffer_bytes"])
        # Source planner puts full dense gate/up matrices on the first owners.
        self.assertGreater(self.result["stage_owners"][0]["reserved_buffer_bytes"],
                           self.result["stage_owners"][31]["reserved_buffer_bytes"])

    def test_kernel_events_drain_and_peak_matches_storage_trace(self):
        for probe in self.result["kernel_probes"].values():
            live, maximum, handles = 0, 0, {}
            for event in probe["storage_events"]:
                self.assertEqual(event["phase"], "newton_schulz")  # prepare/update in-place
                if event["op"]=="alloc":
                    self.assertNotIn(event["storage_id"], handles)
                    handles[event["storage_id"]]=event["bytes"]
                    live += event["bytes"]
                else:
                    live -= handles.pop(event["storage_id"])
                maximum=max(maximum,live)
            self.assertFalse(handles)
            self.assertEqual(live,0)
            self.assertEqual(maximum,probe["transient_peak_bytes"])
        expert = next(p for p in self.result["kernel_probes"].values() if p["shape"]==[28,2,3072,3584])
        # Source batched path retains original BF16 result + old/new matrices,
        # and two gram matrices at the largest allocation event.
        self.assertEqual(expert["transient_peak_bytes"], 2*(3*28*2*3072*3584+2*28*2*3072**2))

    def test_source_packing_creates_no_unmodeled_reshape_copies(self):
        self.assertTrue(self.result["coverage"]["actual_source_pack_unpack_tensor_operations"])
        self.assertTrue(self.result["packing_probes"])
        for probe in self.result["packing_probes"].values():
            self.assertEqual(probe["transient_peak_bytes"],0)
            self.assertEqual(probe["storage_events"],[])


if __name__ == "__main__": unittest.main()
