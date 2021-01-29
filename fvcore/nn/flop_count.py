import typing

import torch.nn as nn
from fvcore.nn.jit_analysis import JitModelAnalysis
from fvcore.nn.jit_handles import (
    addmm_flop_jit,
    bmm_flop_jit,
    conv_flop_jit,
    einsum_flop_jit,
    matmul_flop_jit,
)


_DEFAULT_SUPPORTED_OPS: typing.Dict[str, typing.Callable] = {
    "aten::addmm": addmm_flop_jit,
    "aten::bmm": bmm_flop_jit,
    "aten::_convolution": conv_flop_jit,
    "aten::einsum": einsum_flop_jit,
    "aten::matmul": matmul_flop_jit,
}


def default_flop_counter(
    model: nn.Module,
    inputs: typing.Tuple[object, ...],
    additional_ops: typing.Dict[str, typing.Callable] = {},
) -> JitModelAnalysis:
    """
    Constructs a JitModelAnalysis class configured for flop counting.
    By default, counts the flops for convolutions and matrix
    matrix multiplications, and reports results in Gflops.
    Args:
        model (nn.Module): The model to compute flop counts.
        inputs (tuple): Inputs that are passed to `model` to count flops.
            Inputs need to be in a tuple.
        additional_ops (dict(str,Callable) or None) : provide additional
            handlers for extra ops, or overwrite the existing handlers for
            convolution and matmul and einsum. The key is operator name and the value
            is a function that takes (inputs, outputs) of the op. We count
            one Multiply-Add as one FLOP.
    Returns:
        JitModelAnalysis : computes and stores flop counts, organized
            by module and by operator type.
    """
    ops_handles = {**_DEFAULT_SUPPORTED_OPS, **(additional_ops or {})}
    flop_counter = JitModelAnalysis(model, inputs, ops_handles)
    flop_counter.set_output_scale("giga")
    return flop_counter


def flop_count(
    model: nn.Module,
    inputs: typing.Tuple[object, ...],
    # pyre-fixme[24]: Generic type `typing.Callable` expects 2 type parameters.
    supported_ops: typing.Union[typing.Dict[str, typing.Callable], None] = None,
) -> typing.Tuple[typing.DefaultDict[str, float], typing.Counter[str]]:
    """
    Given a model and an input to the model, compute the Gflops of the given
    model.
    Args:
        model (nn.Module): The model to compute flop counts.
        inputs (tuple): Inputs that are passed to `model` to count flops.
            Inputs need to be in a tuple.
        supported_ops (dict(str,Callable) or None) : provide additional
            handlers for extra ops, or overwrite the existing handlers for
            convolution and matmul and einsum. The key is operator name and the value
            is a function that takes (inputs, outputs) of the op. We count
            one Multiply-Add as one FLOP.
    Returns:
        tuple[defaultdict, Counter]: A dictionary that records the number of
            gflops for each operation and a Counter that records the number of
            skipped operations.
    """
    flop_counter = default_flop_counter(model, inputs, supported_ops)
    return flop_counter.by_operator(), flop_counter.skipped_ops()
