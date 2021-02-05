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
from .jit_handles import (
    Handle,
    addmm_flop_jit,
    bmm_flop_jit,
    conv_flop_jit,
    einsum_flop_jit,
    matmul_flop_jit,
)


# A dictionary that maps supported operations to their flop count jit handles.
_DEFAULT_SUPPORTED_OPS: Dict[str, Handle] = {
    "aten::addmm": addmm_flop_jit,
    "aten::bmm": bmm_flop_jit,
    "aten::_convolution": conv_flop_jit,
    "aten::einsum": einsum_flop_jit,
    "aten::matmul": matmul_flop_jit,
}

class FlopCount(JitModelAnalysis):
    """
    Provides access to per-submodule model flop count obtained by
    tracing a model with pytorch's jit tracing functionality. By default,
    comes with standard flop counters for five operators:

    aten::addmm, used by linear layers
    aten::_convolution, used by convolutional layers
    aten::bmm, batch matrix multiply
    aten::matmul, matrix multiplication
    aten::einsum, Einstein notation summation

    Handles for additional operators may be added, or the default ones 
    overwritten, using the 'additional_ops' input at initialization or the 
    .set_op_handle(name, func) method.

    Flop counts can be obtained as:

    .total(module="") : total flop count for the module
    .by_operator(module="") : flpo counts for the module, as a dictionary 
        over different operator types
    .by_module() : dictionary of flop counts for all descendants 
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
                trace graph to a function used to calculate the flop.
                The function must take the inputs and outputs of the op, 
                each as a list(torch._C.Value), and returns a counter of 
                the form {op_name : number}. This adds to or overwrites
                the default flop  handles. 
        """
        ops_handles = {**_DEFAULT_SUPPORTED_OPS, **(additional_ops or {})}
        super(FlopCount, self).__init__(
            model=model,
            inputs=inputs,
            ops_handles=ops_handles
        )


def flop_count(
    model: nn.Module,
    inputs: Tuple[Any, ...],
    supported_ops: Union[Dict[str, Handle], None] = None,
) -> Tuple[DefaultDict[str, float], Counter[str]]:
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

    flop_counter = FlopCount(model, inputs, supported_ops)
    giga_flops = defaultdict(float)
    for op, flop in flop_counter.by_operator().items():
        giga_flops[op] = flop / 1e9
    return giga_flops, flop_counter.skipped_ops()
