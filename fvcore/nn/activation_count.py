# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import typing
from collections import defaultdict

import torch.nn as nn

from .jit_handles import generic_activation_jit, get_jit_model_analysis


# A dictionary that maps supported operations to their activation count handles.
_DEFAULT_SUPPORTED_OPS: typing.Dict[str, typing.Callable] = {
    "aten::_convolution": generic_activation_jit("conv"),
    "aten::addmm": generic_activation_jit("addmm"),
}


def activation_count(
    model: nn.Module,
    inputs: typing.Tuple[object, ...],
    supported_ops: typing.Union[typing.Dict[str, typing.Callable], None] = None,
) -> typing.Tuple[typing.DefaultDict[str, float], typing.Counter[str]]:
    """
    Given a model and an input to the model, compute the total number of
    activations of the model.

    Args:
        model (nn.Module): The model to compute activation counts.
        inputs (tuple): Inputs that are passed to `model` to count activations.
            Inputs need to be in a tuple.
        supported_ops (dict(str,Callable) or None) : provide additional
            handlers for extra ops, or overwrite the existing handlers for
            convolution and matmul. The key is operator name and the value
            is a function that takes (inputs, outputs) of the op.

    Returns:
        tuple[defaultdict, Counter]: A dictionary that records the number of
            activation (mega) for each operation and a Counter that records the
            number of skipped operations.
    """
    assert isinstance(inputs, tuple), "Inputs need to be in a tuple."
    supported_ops = {**_DEFAULT_SUPPORTED_OPS, **(supported_ops or {})}

    # Run activation count.
    total_activation_count, skipped_ops = get_jit_model_analysis(
        model, inputs, supported_ops
    )

    # Log for skipped operations.
    logger = logging.getLogger(__name__)
    if len(skipped_ops) > 0:
        for op, freq in skipped_ops.items():
            logger.warning("Skipped operation {} {} time(s)".format(op, freq))

    # Convert activation count to mega count.
    final_count = defaultdict(float)
    for op in total_activation_count:
        final_count[op] = total_activation_count[op] / 1e6

    return final_count, skipped_ops
