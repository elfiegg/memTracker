"""Model-independent attribution of captured allocator history."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json


METRICS = ('allocated_bytes', 'active_bytes', 'pending_free_bytes', 'reserved_bytes', 'cached_bytes',
           'requested_allocated_bytes', 'requested_active_bytes', 'requested_pending_free_bytes')
PHYSICAL = ('allocated_bytes', 'active_bytes', 'pending_free_bytes', 'cached_bytes')


def _integer(value):
    if type(value) is not int or value < 0:
        raise ValueError(f'Expected nonnegative byte count/address, got {value!r}')
    return value


def _pool(value):
    return ':'.join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)


class _History:
    def __init__(self, segments, device):
        self.blocks = {}
        self.mappings = []
        self.totals = Counter(dict.fromkeys(METRICS, 0))
        self.origins, self.sites, self.pools = Counter(), Counter(), Counter()
        self.stacks = {}
        self.oom = []
        self.unknown_sizes = 0
        for segment in segments:
            if segment.get('device', device) != device:
                continue
            self.map(segment['address'], segment['total_size'], _pool(segment.get('segment_pool_id', [0, 0])))
            for block in segment['blocks']:
                state = block['state']
                if state == 'inactive':
                    continue
                if state not in ('active_allocated', 'active_awaiting_free'):
                    raise ValueError(f'Unsupported block state: {state}')
                self.allocate(block['address'], block['size'], 'pre_capture', block.get('frames', []),
                              requested=block.get('requested_size',block['size']))
                if state == 'active_awaiting_free':
                    self.free_requested(block['address'])

    def map(self, address, size, pool):
        address, size = _integer(address), _integer(size)
        end = address + size
        if any(address < b and a < end for a, b, _ in self.mappings):
            raise ValueError('Overlapping allocator mappings')
        self.mappings.append((address, end, pool))
        self.totals['reserved_bytes'] += size
        self.pools[pool] += size

    def unmap(self, address, size):
        address, size = _integer(address), _integer(size)
        end, removed, remaining = address + size, 0, []
        if any(address < addr+b['size'] and addr < end for addr, b in self.blocks.items()):
            raise ValueError('Unmapping live or pending storage')
        for a, b, pool in self.mappings:
            lo, hi = max(a, address), min(b, end)
            if hi <= lo:
                remaining.append((a, b, pool))
                continue
            removed += hi-lo
            self.pools[pool] -= hi-lo
            if a < lo: remaining.append((a, lo, pool))
            if hi < b: remaining.append((hi, b, pool))
        if removed != size:
            raise ValueError('Unmapping unknown storage; incomplete history')
        self.mappings = remaining
        self.totals['reserved_bytes'] -= size

    def allocate(self, address, size, origin, frames, *, requested=None, exact=True):
        address, size = _integer(address), _integer(size)
        if address in self.blocks:
            raise ValueError('Reusing live or pending address')
        covered = sum(max(0, min(b, address+size)-max(a, address)) for a,b,_ in self.mappings)
        if covered != size:
            raise ValueError('Allocation outside known mappings; incomplete history')
        if any(address < a+b['size'] and a < address+size for a,b in self.blocks.items()):
            raise ValueError('Overlapping live or pending allocations')
        stack = hashlib.sha256(json.dumps(frames, sort_keys=True).encode()).hexdigest()[:16]
        self.stacks[stack] = frames
        requested = size if requested is None else _integer(requested)
        if requested > size:
            raise ValueError('Requested size exceeds known physical block')
        self.blocks[address] = dict(size=size, origin=origin, stack=stack, pending=False, requested=requested, exact=exact)
        self.unknown_sizes += int(not exact)
        self.totals['allocated_bytes'] += size
        self.totals['active_bytes'] += size
        self.totals['requested_allocated_bytes'] += requested
        self.totals['requested_active_bytes'] += requested
        self.origins[origin] += size
        self.sites[stack] += size

    def free_requested(self, address):
        block = self.blocks.get(address)
        if block is None or block['pending']:
            raise ValueError('Free requested for unknown/already freed allocation')
        block['pending'] = True
        self.totals['allocated_bytes'] -= block['size']
        self.totals['pending_free_bytes'] += block['size']
        self.totals['requested_allocated_bytes'] -= block['requested']
        self.totals['requested_pending_free_bytes'] += block['requested']

    def free_completed(self, address):
        block = self.blocks.get(address)
        if block is None or not block['pending']:
            raise ValueError('Free completed without matching request')
        self.blocks.pop(address)
        self.unknown_sizes -= int(not block['exact'])
        self.totals['active_bytes'] -= block['size']
        self.totals['pending_free_bytes'] -= block['size']
        self.totals['requested_active_bytes'] -= block['requested']
        self.totals['requested_pending_free_bytes'] -= block['requested']
        self.origins[block['origin']] -= block['size']
        self.sites[block['stack']] -= block['size']

    def event(self, event, origin, index, *, requested_sizes=False, size_hint=None):
        action = event['action']
        if action == 'alloc':
            size = event['size'] if size_hint is None else size_hint
            self.allocate(event['addr'], size, origin, event.get('frames', []),
                requested=event['size'], exact=not requested_sizes or size_hint is not None)
        elif action == 'free_requested': self.free_requested(event['addr'])
        elif action == 'free_completed': self.free_completed(event['addr'])
        elif action in ('segment_alloc', 'segment_map'):
            self.map(event['addr'], event['size'], _pool(event.get('pool_id', [0, 0])))
        elif action in ('segment_free', 'segment_unmap'):
            self.unmap(event['addr'], event['size'])
        elif action == 'oom':
            self.oom.append(dict(event_index=index, requested_bytes=_integer(event['size']),
                                 device_free_bytes=event.get('device_free'), phase=origin))
        elif action not in ('snapshot', 'annotate'):
            raise ValueError(f'Unsupported allocator action: {action}')

    def snapshot(self):
        counts = dict(self.totals)
        counts['cached_bytes'] = counts['reserved_bytes']-counts['active_bytes']
        if counts['cached_bytes'] < 0:
            raise ValueError('Active memory exceeds mapped storage')
        return dict(**counts,
            physical_sizes_complete=self.unknown_sizes == 0,
            active_by_origin_phase={k:v for k,v in self.origins.items() if v},
            active_by_stack={k:v for k,v in self.sites.items() if v},
            reserved_by_pool={k:v for k,v in self.pools.items() if v})


def _size_hints(events, checkpoints):
    """Bind observed block sizes to allocation generations, not reused addresses."""
    by_index = defaultdict(list)
    for cp in checkpoints:
        by_index[cp['trace_index']].extend(cp.get('block_sizes', []))
    live, hints = {}, {}
    for index in range(len(events)+1):
        for block in by_index[index]:
            generation = live.get(block['address'])
            if generation is not None:
                size = _integer(block['size'])
                if generation in hints and hints[generation] != size:
                    raise ValueError('Physical allocation changed size without a new allocation event')
                hints[generation] = size
        if index < len(events):
            event = events[index]
            if event['action'] == 'alloc': live[event['addr']] = index
            elif event['action'] == 'free_completed': live.pop(event['addr'], None)
    return hints


def _public_state(state):
    state = deepcopy(state)
    if not state['physical_sizes_complete']:
        for key in PHYSICAL:
            suffix = 'upper_bound' if key == 'cached_bytes' else 'lower_bound'
            state[f'{key}_{suffix}'] = state[key]
            state[key] = None
        state['active_attribution_is_lower_bound'] = True
    return state


def analyze_capture(capture):
    """Exact accounting of recorded events, not a prediction of unseen execution.

    Native PyTorch event sizes are requested bytes. Physical sizes are resolved
    from boundary block observations when possible. Unobserved transient block
    sizes stay unknown: requested-byte peaks are exact, physical peaks are lower
    bounds. Synthetic normalized traces may explicitly supply block sizes.
    """
    if capture.get('schema_version') != 1 or not capture.get('history_complete'):
        raise ValueError('Unsupported or incomplete allocator history')
    events = capture['events']
    semantics = capture.get('trace_size_semantics', 'block')
    if semantics not in ('requested', 'block'):
        raise ValueError('Unsupported trace size semantics')
    hints = _size_hints(events, capture.get('checkpoints', [])) if semantics == 'requested' else {}
    spans = deepcopy(capture['spans'])
    starts, ends, checkpoints = defaultdict(list), defaultdict(list), defaultdict(list)
    identifiers = set()
    for s in spans:
        if not s.get('name') or s['id'] in identifiers or not 0 <= s['start'] <= s['end'] <= len(events):
            raise ValueError('Invalid or duplicate phase span')
        identifiers.add(s['id'])
        s['peaks'] = {}
        starts[s['start']].append(s)
        ends[s['end']].append(s)
    for checkpoint_id, checkpoint in enumerate(capture.get('checkpoints', [])):
        if not 0 <= checkpoint['trace_index'] <= len(events):
            raise ValueError('Invalid checkpoint trace index')
        checkpoints[checkpoint['trace_index']].append(dict(deepcopy(checkpoint), checkpoint_id=checkpoint_id))
    history = _History(capture['baseline_segments'], capture['device'])
    active, compared = {}, []
    def sample(s, state, index):
        s['_has_unknown_sizes'] = s.get('_has_unknown_sizes',False) or not state['physical_sizes_complete']
        for metric in METRICS:
            if metric not in s['peaks'] or state[metric] > s['peaks'][metric]['bytes']:
                s['peaks'][metric] = dict(bytes=state[metric], trace_index=index, **state)
    for index in range(len(events)+1):
        state = history.snapshot()
        for s in active.values(): sample(s, state, index)
        for s in starts[index]:
            active[s['id']] = s
            s['begin'] = state
            sample(s, state, index)
        for s in ends[index]:
            s['end_state'] = state
            active.pop(s['id'])
        for cp in checkpoints[index]:
            cp['reconstructed'] = _public_state(state)
            cp['counter_differences'] = {k: cp[k]-state[k] for k in ('allocated_bytes','active_bytes','reserved_bytes') if cp.get(k) is not None}
            cp['physical_sizes_complete'] = state['physical_sizes_complete']
            owned, padding, unmatched, seen = Counter(), 0, [], set()
            for tensor in cp.get('tensor_owners', {}).get('storages', []):
                if tensor['device'] != f"cuda:{capture['device']}":
                    continue
                address = tensor['address']
                if address in seen:
                    raise ValueError('Duplicate storage in ownership inventory')
                seen.add(address)
                block = history.blocks.get(address)
                if block is None or block['pending'] or block['size'] < tensor['storage_bytes']:
                    unmatched.append(address)
                    continue
                owned[tensor['category']] += tensor['storage_bytes']
                padding += block['size']-tensor['storage_bytes']
            cp['ownership'] = dict(storage_bytes_by_category=dict(owned),
                allocator_padding_bytes=padding, unmatched_addresses=unmatched,
                unattributed_allocated_bytes=state['allocated_bytes']-sum(owned.values())-padding)
            reserved = cp.get('reserved_bytes')
            for scope in ('device', 'process'):
                used = cp.get(f'{scope}_used_bytes')
                cp[f'{scope}_outside_allocator_bytes'] = None if used is None or reserved is None else used-reserved
            compared.append(cp)
        if index < len(events):
            origin = max(active.values(), key=lambda s:s['id'])['name'] if active else 'unscoped'
            history.event(events[index], origin, index, requested_sizes=semantics == 'requested', size_hint=hints.get(index))
    for s in spans:
        s['trace_end'] = s.pop('end')
        s['end'] = s.pop('end_state')
        incomplete = s.pop('_has_unknown_sizes',False)
        s['physical_peak_sizes_complete'] = not incomplete
        s['begin'], s['end'] = _public_state(s['begin']), _public_state(s['end'])
        for key, peak in s['peaks'].items():
            s['peaks'][key] = _public_state(peak)
            if incomplete and key in PHYSICAL:
                bound = 'upper_bound_bytes' if key == 'cached_bytes' else 'lower_bound_bytes'
                s['peaks'][key][bound] = peak['bytes']
                s['peaks'][key]['bytes'] = None
        s['external_boundary_delta_bytes'] = {}
        if 'start_checkpoint' in s and 'end_checkpoint' in s:
            by_id = {c['checkpoint_id']:c for c in compared}
            before, after = by_id[s['start_checkpoint']], by_id[s['end_checkpoint']]
            for scope in ('device', 'process'):
                key = f'{scope}_outside_allocator_bytes'
                a, b = before[key], after[key]
                s['external_boundary_delta_bytes'][scope] = None if a is None or b is None else b-a
            if capture.get('isolated_peak_counters'):
                intervals = [by_id[i] for i in range(s['start_checkpoint']+1, s['end_checkpoint']+1)]
                for key in ('allocated_bytes','active_bytes','reserved_bytes'):
                    values = [cp.get('interval_peaks',{}).get(key) for cp in intervals]
                    if values and all(v is not None for v in values):
                        # Physical counters can observe unsampled transient block
                        # sizes. Their peak instant/stack attribution is unknown.
                        s['peaks'][key] = dict(bytes=max(values), source='isolated_allocator_peak_counters',
                            trace_index=None, active_by_origin_phase=None, active_by_stack=None)
    return dict(schema_version=1, kind='measured_allocator_history_attribution',
        trace_size_semantics=semantics,
        metadata=deepcopy(capture.get('metadata', {})),
        completed_successfully=not history.oom and all(s.get('status', 'ok') == 'ok' for s in spans),
        counters_reconciled=(all(cp['physical_sizes_complete'] and not any(cp['counter_differences'].values()) for cp in compared)
                             if any(cp['counter_differences'] for cp in compared) else None),
        phases=spans, final=_public_state(history.snapshot()), checkpoints=compared, stacks=history.stacks,
        oom_events=history.oom,
        limitations=['Origin phases and call stacks do not prove semantic tensor ownership.',
            'External memory is sampled only at boundaries; transient external peaks are unknown.',
            'Device-wide residual includes other processes; it is not an NCCL allocation counter.',
            'Recorded history replay is measured evidence, not independent estimator validation.'])


def compare_phase_estimate(measured, estimated, *, estimated_device='cpu'):
    """Diagnostic differences only; caller must establish matching source/windows.

    Matching uses phase name, step/accumulation labels, and occurrence number.
    OOM/error phases stay censored. Tensor-request bytes are never compared with
    allocator-active bytes. Missing predictions stay unknown rather than zero.
    """
    if measured['counters_reconciled'] is False:
        return dict(status='counter_mismatch', accuracy_verified=False, comparisons=[])
    def indexed(rows):
        seen, result = Counter(), {}
        for row in rows:
            meta = row.get('metadata', {})
            base = (row['name'], meta.get('step'), meta.get('accumulation_index'))
            key = (*base, seen[base])
            seen[base] += 1
            result[key] = row
        return result
    predictions = indexed(estimated['phases'])
    comparisons = []
    for key, row in indexed(measured['phases']).items():
        prediction = predictions.pop(key, None)
        peaks = prediction.get('devices', {}).get(estimated_device, {}) if prediction else {}
        metrics = {'allocated_bytes':'allocated_peak_bytes', 'active_bytes':'active_peak_bytes',
                   'reserved_bytes':'reserved_peak_bytes', 'pending_free_bytes':'pending_free_peak_bytes',
                   'requested_allocated_bytes':'requested_allocated_peak_bytes'}
        observed = {k:row['peaks'][k]['bytes'] for k in metrics}
        predicted = {k:peaks.get(v) for k,v in metrics.items()}
        censored = row.get('status', 'ok') != 'ok' or (prediction and prediction.get('status', 'ok') != 'ok')
        comparisons.append(dict(name=row['name'], metadata=row.get('metadata', {}), occurrence=key[-1],
            measured_peak_bytes=observed, modeled_peak_bytes=predicted,
            gaps_bytes=None if censored else {k:None if predicted[k] is None or observed[k] is None else observed[k]-predicted[k] for k in metrics},
            censored=bool(censored), missing_prediction=prediction is None,
            measured_active_peak_origins=row['peaks']['active_bytes']['active_by_origin_phase'],
            measured_active_peak_stacks=row['peaks']['active_bytes']['active_by_stack'],
            external_boundary_delta_bytes=row['external_boundary_delta_bytes']))
    return dict(status='diagnostic_only', accuracy_verified=False,
        counters_reconciled=measured['counters_reconciled'], comparisons=comparisons,
        unmatched_modeled_phases=[list(k) for k in predictions],
        limitations=['Matching phase labels alone does not establish matching code, config or runtime.',
            'Differences diagnose missing lifetimes; no fitted adjustment or default multiplier is applied.',
            'External boundary deltas are not external phase peaks.'])


def main():
    import argparse
    from pathlib import Path
    from .report_io import read_report, write_report
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--estimate', type=Path)
    parser.add_argument('--estimated-device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze_capture(read_report(args.capture))
    if args.estimate:
        report['estimate_comparison'] = compare_phase_estimate(report, read_report(args.estimate), estimated_device=args.estimated_device)
    write_report(args.output, report)


if __name__ == '__main__': main()
