"""Read-only storage ownership inventory for ordinary PyTorch modules/optimizers."""
from collections import Counter


def tensor_inventory(model_parts, optimizers=()):
    """Count aliased storage once, preserving generic optimizer state key names.

    Never calls state_dict(), gathers DTensors, or reads tensor values. Private
    optimizer buffers outside .state and kernel-private tensors remain unknown.
    """
    import torch
    storages, errors = {}, []
    def add(tensor, category, name):
        if not isinstance(tensor, torch.Tensor):
            return
        # DTensor's local payload: no collective and no redistribution.
        tensor = getattr(tensor, '_local_tensor', tensor)
        try:
            if tensor.device.type == 'meta':
                raise ValueError('Meta tensors have no resident allocation')
            storage = tensor.untyped_storage()
            size = storage.nbytes()
            if size == 0:
                return
            device, address = str(tensor.device), storage.data_ptr()
            key = (device, address)
            row = storages.setdefault(key, dict(device=device, address=address,
                storage_bytes=size, aliases=[]))
            if row['storage_bytes'] != size:
                raise ValueError('Storage changed during inventory')
            row['aliases'].append(dict(category=category, name=name, shape=list(tensor.shape), dtype=str(tensor.dtype)))
        except Exception as error:
            errors.append(dict(name=name, error=f'{type(error).__name__}: {error}'))
    for index, model in enumerate(model_parts):
        for name, tensor in model.named_parameters(remove_duplicate=False):
            add(tensor, 'parameters', f'model.{index}.{name}')
            add(tensor.grad, 'gradients', f'model.{index}.{name}.grad')
        for name, tensor in model.named_buffers(remove_duplicate=False):
            add(tensor, 'buffers', f'model.{index}.{name}')
    def walk(value, name, visited):
        if isinstance(value, torch.Tensor):
            add(value, 'optimizer_state', name)
        elif isinstance(value, (dict, list, tuple)) and id(value) not in visited:
            visited.add(id(value))
            items = value.items() if isinstance(value, dict) else enumerate(value)
            for key, item in items:
                walk(item, f'{name}.{key}', visited)
    for index, optimizer in enumerate(optimizers):
        for param_index, state in enumerate(optimizer.state.values()):
            walk(state, f'optimizer.{index}.{type(optimizer).__name__}.state.{param_index}', set())
    by_category, by_device = Counter(), Counter()
    for row in storages.values():
        categories = {a['category'] for a in row['aliases']}
        row['category'] = next(iter(categories)) if len(categories) == 1 else 'shared'
        by_category[row['category']] += row['storage_bytes']
        by_device[row['device']] += row['storage_bytes']
    return dict(storages=list(storages.values()), bytes_by_category=dict(by_category),
        bytes_by_device=dict(by_device), total_storage_bytes=sum(by_device.values()), errors=errors,
        limitations=['Storage aliases are counted once; shared-category storage is not assigned twice.',
            'Private optimizer buffers outside .state are not inventoried.',
            'Inventory is a boundary sample, not tensor ownership at every transient peak.'])
