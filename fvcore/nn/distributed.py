from typing import List, Tuple

import torch
import torch.distributed as dist
from torch.autograd.function import Function


# pyre-ignore-all-errors[2,14,16]


class _AllReduce(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


def differentiable_all_reduce(input: torch.Tensor) -> torch.Tensor:
    """
    Differentiable counterpart of `dist.all_reduce`.
    """
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() == 1
    ):
        return input
    return _AllReduce.apply(input)


class _AllGather(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def differentiable_all_gather(input: torch.Tensor) -> List[torch.Tensor]:
    """
    Differentiable counterpart of `dist.all_gather`.
    """
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() == 1
    ):
        return [input]
    return list(_AllGather.apply(input))
