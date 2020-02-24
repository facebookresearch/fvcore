# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import typing
from collections import defaultdict
import torch.nn as nn

from .jit_handles import (
    addmm_flop_jit,
    conv_flop_jit,
    einsum_flop_jit,
    get_jit_model_analysis,
    matmul_flop_jit,
)

# A dictionary that maps supported operations to their flop count jit handles.
_SUPPORTED_OPS: typing.Dict[str, typing.Callable] = {
    "aten::addmm": addmm_flop_jit,
    "aten::_convolution": conv_flop_jit,
    "aten::einsum": einsum_flop_jit,
    "aten::matmul": matmul_flop_jit,
}


def flop_count(
    model: nn.Module,
    inputs: typing.Tuple[object, ...],
    supported_ops: typing.Union[typing.Dict[str, typing.Callable], None] = None,
) -> typing.Tuple[typing.DefaultDict[str, float], typing.Counter[str]]:
    """
    Given a model and an input to the model, compute the Gflops of the given
    model. Note the input should have a batch size of 1.

    Args:
        model (nn.Module): The model to compute flop counts.
        inputs (tuple): Inputs that are passed to `model` to count flops.
            Inputs need to be in a tuple.
        supported_ops (dict(str,Callable) or None) : By default, we count flops
            for convolution layers, fully connected layers, torch.matmul and
            torch.einsum operations. We define a FLOP as a single atomic
            Multiply-Add. Users can provide customized supported_ops for
            counting flops if desired.

    Returns:
        tuple[defaultdict, Counter]: A dictionary that records the number of
            gflops for each operation and a Counter that records the number of
            skipped operations.
    """
    assert isinstance(inputs, tuple), "Inputs need to be in a tuple."
    if not supported_ops:
        supported_ops = _SUPPORTED_OPS.copy()

    # Run flop count.
    total_flop_counter, skipped_ops = get_jit_model_analysis(
        model, inputs, supported_ops
    )

    # Log for skipped operations.
    if len(skipped_ops) > 0:
        for op, freq in skipped_ops.items():
            logging.warning("Skipped operation {} {} time(s)".format(op, freq))

    # Convert flop count to gigaflops.
    final_count = defaultdict(float)
    for op in total_flop_counter:
        final_count[op] = total_flop_counter[op] / 1e9

    return final_count, skipped_ops
