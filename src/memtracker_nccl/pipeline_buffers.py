"""Exercise real PipelineStage receive allocation helpers with supplied metadata.

This does not construct a distributed pipeline or execute model forward/backward.
The caller supplies shapes; only the receive-buffer allocation is executed.
Private PyTorch API, verified on 2.14.1 and the recorded 2.15 GPU runtime.
"""
from __future__ import annotations


def allocate_receive_buffers(*, stage_index: int, tokens: int, incoming_blocks: int,
                             outgoing_blocks: int, microbatches: int,
                             device: str = 'cpu', dim: int = 7168):
    import torch
    from torch.distributed.pipelining.stage import PipelineStage
    from torch.distributed.pipelining._utils import _StageMeta, _TensorMeta
    for key, value in dict(stage_index=stage_index, tokens=tokens, incoming_blocks=incoming_blocks,
                           outgoing_blocks=outgoing_blocks, microbatches=microbatches, dim=dim).items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if stage_index > 7 or min(tokens, microbatches, dim) < 1:
        raise ValueError('Require PP8 stage 0..7 and positive tokens/microbatches/dim')

    def meta(shape, requires_grad=True):
        stride, count = [], 1
        for size in reversed(shape):
            stride.insert(0, count)
            count *= size
        return _TensorMeta(torch.Size(shape), tuple(stride), torch.bfloat16, requires_grad)

    # Deliberately omit process-group/model setup. These three unmodified methods
    # need only the fields below and allocate the real per-microbatch buffers.
    stage = object.__new__(PipelineStage)
    stage.stage_index, stage.num_stages, stage.device = stage_index, 8, torch.device(device)
    stage.args_recv_info, stage.grad_recv_info = {}, {}
    inputs = (meta((tokens, dim)), meta((tokens, incoming_blocks, dim)))
    outputs = (meta((tokens, dim)), meta((tokens, outgoing_blocks, dim)))
    grads = (meta((tokens, dim), False), meta((tokens, outgoing_blocks, dim), False))
    stage._stage_meta = _StageMeta(inputs=inputs, outputs=outputs, output_grads=grads)
    stage._setup_forward_recv_info(microbatches, True)
    stage._setup_forward_send_info()
    stage._setup_backward_recv_info(microbatches)
    return stage


def buffer_bytes(stage) -> dict[str, int]:
    def count(infos):
        return sum(item.buffer.numel() * item.buffer.element_size()
                   for entries in infos.values() for item in entries if item.buffer is not None)
    forward, backward = count(stage.args_recv_info), count(stage.grad_recv_info)
    return {'forward_bytes': forward, 'backward_bytes': backward, 'total_bytes': forward + backward}
