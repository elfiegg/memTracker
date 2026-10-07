"""Model behavior tests; these do not require PyTorch or CUDA."""
import unittest

from memtracker_nccl.nccl_model import (
    NcclMemoryModel, calibrated_profile, p2p_profile, shared_net_profile,
)


class Ledger:
    def __init__(self):
        self.allocations = {}
        self.events = []
        self.peak = 0

    def external_alloc(self, allocation_id, size_bytes, **kwargs):
        if allocation_id in self.allocations:
            raise ValueError("duplicate allocation")
        self.allocations[allocation_id] = size_bytes
        self.events.append(("alloc", allocation_id, size_bytes))
        self.peak = max(self.peak, sum(self.allocations.values()))

    def external_free(self, allocation_id):
        self.events.append(("free", allocation_id))
        del self.allocations[allocation_id]


class NcclModelTests(unittest.TestCase):
    def setUp(self):
        self.ledger = Ledger()
        self.model = NcclMemoryModel(self.ledger)
        self.model.register_communicator(
            "dp", calibrated_profile(1024, provenance="test scenario, not a measurement")
        )

    def test_payload_is_not_added_to_nccl_overhead_and_pool_is_reused(self):
        self.assertEqual(self.ledger.allocations, {})
        for payload in (4096, 2**40):
            token = self.model.begin_collective("dp", payload_bytes=payload)
            self.model.complete_collective(token)
        self.assertEqual(sum(self.ledger.allocations.values()), 1024)
        self.assertEqual(len(self.ledger.events), 1)
        self.model.destroy_communicator("dp")
        self.assertEqual(self.ledger.allocations, {})

    def test_overlapping_temporaries_live_until_completion(self):
        first = self.model.begin_collective("dp", temporary_bytes=300)
        second = self.model.begin_collective("dp", temporary_bytes=700)
        self.assertEqual(self.ledger.peak, 2024)
        with self.assertRaisesRegex(RuntimeError, "pending"):
            self.model.destroy_communicator("dp")
        self.model.complete_collective(second)
        self.assertEqual(sum(self.ledger.allocations.values()), 1324)
        self.model.complete_collective(first)
        self.assertEqual(sum(self.ledger.allocations.values()), 1024)
        with self.assertRaises(KeyError):
            self.model.complete_collective(first)

    def test_distinct_communicators_and_model_instances_do_not_alias(self):
        self.model.register_communicator("ep", calibrated_profile(2048, "test"))
        a = self.model.begin_collective("dp")
        b = self.model.begin_collective("ep")
        second = NcclMemoryModel(self.ledger)
        second.register_communicator("dp", calibrated_profile(4096, "test"))
        c = second.begin_collective("dp")
        self.assertEqual(sum(self.ledger.allocations.values()), 7168)
        self.model.complete_collective(a)
        self.model.complete_collective(b)
        second.complete_collective(c)

    def test_p2p_source_profile_counts_local_storage_only(self):
        profile = p2p_profile(channels=4)
        # NCCL 2.28.9: local send=2 MiB, local recv=10 MiB per channel.
        self.assertEqual(profile.persistent_bytes, 48 * 2**20)
        self.assertEqual(p2p_profile(channels=4, read=True).persistent_bytes,
                         profile.persistent_bytes)
        self.assertEqual(p2p_profile(channels=4, recv_connections_per_channel=0).persistent_bytes,
                         8 * 2**20)
        self.assertIn("v2.28.9-1", str(profile.to_dict()))
        self.assertTrue(profile.excluded)

    def test_shared_network_pool_does_not_scale_by_peer_or_payload(self):
        profile = shared_net_profile(channels=8)
        self.assertEqual(profile.persistent_bytes, 32 * 2**20)
        send = shared_net_profile(channels=8, recv_pool=False)
        self.assertEqual(send.persistent_bytes, 16 * 2**20)
        self.assertEqual(shared_net_profile(channels=8, chunk_bytes=64*1024).persistent_bytes,
                         16 * 2**20)

    def test_invalid_inputs_fail_before_allocating(self):
        for amount in (-1, 1.5, True):
            with self.assertRaises((ValueError, TypeError)):
                self.model.begin_collective("dp", temporary_bytes=amount)
        with self.assertRaises(ValueError):
            self.model.register_communicator("dp", calibrated_profile(1, "test"))
        with self.assertRaises(ValueError):
            calibrated_profile(1024, "")
        with self.assertRaises(ValueError):
            p2p_profile(channels=0)
        self.assertEqual(self.ledger.allocations, {})


if __name__ == "__main__":
    unittest.main()
