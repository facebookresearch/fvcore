# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# pyre-ignore-all-errors[2,33]

import torch.nn as nn
from collections import defaultdict
from typing import (
    Any, 
    Callable, 
    Counter, 
    DefaultDict, 
    Dict, 
    List, 
    Tuple, 
    Union,
    Optional
)

from .jit_analysis import JitModelAnalysis
from .jit_handles import generic_activation_jit, Handle


# A dictionary that maps supported operations to their activation count handles.
_DEFAULT_SUPPORTED_OPS: Dict[str, Handle] = {
    "aten::_convolution": generic_activation_jit("conv"),
    "aten::addmm": generic_activation_jit("addmm"),
}

class ActivationCount(JitModelAnalysis):
    """
    Provides access to per-submodule model activation count obtained by
    tracing a model with pytorch's jit tracing functionality. By default,
    comes with standard activation counters for two operators:

    aten::addmm, used by linear layers
    aten::_convolution, used by convolutional layers

    Handles for additional may be added, or the default ones overwritten,
    using the 'additional_ops' input at initialization or the 
    .set_op_handle(name, func) method.

    Activation counts can be obtained as:

    .total(module="") : total activation count for the module
    .by_operator(module="") : activation counts for the module, as a
        dictionary over different operator types
    .by_module() : dictionary of activation counts for all descendants 
        of the model,
    .by_module_and_operator() : dictionary indexed by descendant of dictionaries
        over different operator types

    Modules may be referenced using their name as a string (where the
    input model is "") or using the reference to the module itelf.
        
    """
    def __init__(
        self,
        model: nn.Module,
        inputs: Tuple[Any, ...],
        additional_ops: Optional[Dict[str, Handle]] = None
    ) -> None:
        """
        Args:
            model (nn.Module) : The model to analyze
            inputs (tuple) : The inputs to the model for analysis.
            additional_ops (dict(str,Callable) : Map an operator name in the
                trace graph to a function used to calculate the activation
                The function must take the inputs and outputs of the op, 
                each as a list(torch._C.Value), and returns a counter of 
                the form {op_name : number}. This adds to or overwrites
                the default activation handles for aten::addmm and 
                aten::_convolution.
        """
        ops_handles = {**_DEFAULT_SUPPORTED_OPS, **(additional_ops or {})}
        super(ActivationCount, self).__init__(
            model=model,
            inputs=inputs,
            ops_handles=ops_handles
        )


def activation_count(
    model: nn.Module,
    inputs: Tuple[Any, ...],
    supported_ops: Union[Dict[str, Handle], None] = None,
) -> Tuple[DefaultDict[str, float], Counter[str]]:
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

    act_counter = ActivationCount(model, inputs, supported_ops)
    mega_acts = defaultdict(float)
    for op, act in act_counter.by_operator().items():
        mega_acts[op] = act / 1e6
    return mega_acts, act_counter.skipped_ops()
