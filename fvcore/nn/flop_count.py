# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import typing
from collections import defaultdict

import torch.nn as nn

from .jit_handles import (
    addmm_flop_jit,
    bmm_flop_jit,
    conv_flop_jit,
    einsum_flop_jit,
    get_jit_model_analysis,
    matmul_flop_jit,
)


# A dictionary that maps supported operations to their flop count jit handles.
# pyre-fixme[24]: Generic type `typing.Callable` expects 2 type parameters.
_DEFAULT_SUPPORTED_OPS: typing.Dict[str, typing.Callable] = {
    "aten::addmm": addmm_flop_jit,
    "aten::bmm": bmm_flop_jit,
    "aten::_convolution": conv_flop_jit,
    "aten::einsum": einsum_flop_jit,
    "aten::matmul": matmul_flop_jit,
}


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
    assert isinstance(inputs, tuple), "Inputs need to be in a tuple."
    supported_ops = {**_DEFAULT_SUPPORTED_OPS, **(supported_ops or {})}

    # Run flop count.
    total_flop_counter, skipped_ops = get_jit_model_analysis(
        model, inputs, supported_ops
    )

    # Log for skipped operations.
    logger = logging.getLogger(__name__)
    if len(skipped_ops) > 0:
        for op, freq in skipped_ops.items():
            logger.warning("Skipped operation {} {} time(s)".format(op, freq))

    # Convert flop count to gigaflops.
    final_count = defaultdict(float)
    for op in total_flop_counter:
        final_count[op] = total_flop_counter[op] / 1e9

    return final_count, skipped_ops
