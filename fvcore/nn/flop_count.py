# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# pyre-ignore-all-errors[2,33]

from collections import defaultdict
from typing import Any, Counter, DefaultDict, Dict, Optional, Tuple

import torch.nn as nn

from .jit_analysis import JitModelAnalysis
from .jit_handles import (
    Handle,
    addmm_flop_jit,
    bmm_flop_jit,
    conv_flop_jit,
    einsum_flop_jit,
    linear_flop_jit,
    matmul_flop_jit,
)


# A dictionary that maps supported operations to their flop count jit handles.
_DEFAULT_SUPPORTED_OPS: Dict[str, Handle] = {
    "aten::addmm": addmm_flop_jit,
    "aten::bmm": bmm_flop_jit,
    "aten::_convolution": conv_flop_jit,
    "aten::einsum": einsum_flop_jit,
    "aten::matmul": matmul_flop_jit,
    "aten::linear": linear_flop_jit,
}


class FlopCountAnalysis(JitModelAnalysis):
    """
    Provides access to per-submodule model flop count obtained by
    tracing a model with pytorch's jit tracing functionality. By default,
    comes with standard flop counters for convolutional and dot-product operators.
    Note that we count one Multiply-Add as one FLOP.

    Handles for additional operators may be added, or the default ones
    overwritten, using the ``.set_op_handle(name, func)`` method.
    See the method documentation for details.

    Flop counts can be obtained as:

    * ``.total(module_name="")``: total flop count for the module
    * ``.by_operator(module_name="")``: flop counts for the module, as a Counter
      over different operator types
    * ``.by_module()``: Counter of flop counts for all submodules
    * ``.by_module_and_operator()``: dictionary indexed by descendant of Counters
      over different operator types

    An operator is treated as within a module if it is executed inside the
    module's ``__call__`` method. Note that this does not include calls to
    other methods of the module or explicit calls to ``module.forward(...)``.

    Example usage:

    >>> import torch.nn as nn
    >>> import torch
    >>> class TestModel(nn.Module):
    ...    def __init__(self):
    ...        super().__init__()
    ...        self.fc = nn.Linear(in_features=1000, out_features=10)
    ...        self.conv = nn.Conv2d(
    ...            in_channels=3, out_channels=10, kernel_size=1
    ...        )
    ...        self.act = nn.ReLU()
    ...    def forward(self, x):
    ...        return self.fc(self.act(self.conv(x)).flatten(1))

    >>> model = TestModel()
    >>> inputs = (torch.randn((1,3,10,10)),)
    >>> flops = FlopCountAnalysis(model, inputs)
    >>> flops.total()
    13000
    >>> flops.total("fc")
    10000
    >>> flops.by_operator()
    Counter({"addmm" : 10000, "conv" : 3000})
    >>> flops.by_module()
    Counter({"" : 13000, "fc" : 10000, "conv" : 3000, "act" : 0})
    >>> flops.by_module_and_operator()
    {"" : Counter({"addmm" : 10000, "conv" : 3000}),
     "fc" : Counter({"addmm" : 10000}),
     "conv" : Counter({"conv" : 3000}),
     "act" : Counter()
    }
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Tuple[Any, ...],
        op_handles: Optional[Dict[str, Handle]] = None,
    ) -> None:
        op_handles = {**_DEFAULT_SUPPORTED_OPS, **(op_handles or {})}
        super(FlopCountAnalysis, self).__init__(
            model=model, inputs=inputs, op_handles=op_handles
        )

    __init__.__doc__ = JitModelAnalysis.__init__.__doc__


def flop_count(
    model: nn.Module,
    inputs: Tuple[Any, ...],
    supported_ops: Optional[Dict[str, Handle]] = None,
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
            unsupported operations.
    """

    flop_counter = FlopCountAnalysis(model, inputs, supported_ops)
    giga_flops = defaultdict(float)
    for op, flop in flop_counter.by_operator().items():
        giga_flops[op] = flop / 1e9
    return giga_flops, flop_counter.unsupported_ops()
