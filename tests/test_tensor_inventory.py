import unittest
import torch
from memtracker_nccl import tensor_inventory as inventory


class InventoryTests(unittest.TestCase):
    def test_inventory_exists(self):
        self.assertTrue(callable(getattr(inventory, 'tensor_inventory', None)))

    def test_parameters_gradients_buffers_and_lazy_optimizer_states(self):
        model = torch.nn.Linear(4, 2)
        model.register_buffer('lookup', torch.ones(3))
        optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
        before = inventory.tensor_inventory([model], [optimizer])
        self.assertNotIn('optimizer_state', before['bytes_by_category'])
        model(torch.ones(1,4)).sum().backward()
        optimizer.step()
        after = inventory.tensor_inventory([model], [optimizer])
        self.assertEqual(after['bytes_by_category']['parameters'], 40)
        self.assertEqual(after['bytes_by_category']['gradients'], 40)
        self.assertEqual(after['bytes_by_category']['buffers'], 12)
        self.assertEqual(after['bytes_by_category']['optimizer_state'], 88)
        self.assertEqual(sum(r['storage_bytes'] for r in after['storages']), after['total_storage_bytes'])
        names = [alias['name'] for row in after['storages'] for alias in row['aliases']]
        self.assertTrue(any('exp_avg' in name for name in names))

    def test_aliases_and_views_are_counted_once(self):
        model = torch.nn.Module()
        p = torch.nn.Parameter(torch.ones(16))
        model.register_parameter('weight', p)
        model.register_buffer('view', p.detach()[:2])
        r = inventory.tensor_inventory([model], [])
        self.assertEqual(r['total_storage_bytes'], 64)
        self.assertEqual(len(r['storages']), 1)
        self.assertEqual(r['storages'][0]['category'], 'shared')
        self.assertEqual(len(r['storages'][0]['aliases']), 2)


if __name__ == '__main__': unittest.main()
