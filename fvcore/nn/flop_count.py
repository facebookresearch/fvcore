# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# pyre-ignore-all-errors[2,33]

from collections import defaultdict
from typing import Any, Counter, DefaultDict, Dict, Optional, Tuple, Union

import torch.nn as nn

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


class FlopCountAnalysis(JitModelAnalysis):
    """
    Provides access to per-submodule model flop count obtained by
    tracing a model with pytorch's jit tracing functionality. By default,
    comes with standard flop counters for five operators, which count one
    Multiply-Add as one FLOP:

    aten::addmm, used by linear layers
    aten::_convolution, used by convolutional layers
    aten::bmm, batch matrix multiply
    aten::matmul, matrix multiplication
    aten::einsum, Einstein notation summation

    Handles for additional operators may be added, or the default ones
    overwritten, using the 'additional_ops' input at initialization or the
    .set_op_handle(name, func) method. The function must take the inputs
    and outputs of the op, each as a list(torch._C.Value), and returns a
    counter of the form {op_name : number}.All handles may be cleared with
    .clear_op_handles().

    Flop counts can be obtained as:

    .total(module_name="") : total flop count for the module
    .by_operator(module_name="") : flop counts for the module, as a Counter
        over different operator types
    .by_module() : Counter of flop counts for all descendants
        of the model,
    .by_module_and_operator() : dictionary indexed by descendant of Counters
        over different operator types

    Submodules may be referred to using the module's name. The input model has
    name "", while its descendants have names of the form
    "child.grandchild.grandgrandchild...". For submodules that may have multiple
    names, any may be used to access the module.

    An operator is treated as within the scope of a module if calling that
    module directly resulted in that operator being run. In particular,
    this means that calls to other functions owned by a module or explicit
    calls to module.forward(...) will not register resulting operators as
    contributing flops to that module.

    Additional methods include:

    .skipped_ops(module_name="") : get the number of times each non-trivial
        op that didn't have a handle was skipped for the specified module.
        Returns a Counter over operator names. Operators considered trivial
        and not reported here are listed in .ignored_ops.

    .canonical_module_name(name) : gets the name that will be used for the
        module in .by_module() and .by_module_and_operator()

    .copy(new_model=None, new_inputs=None) : copies the analyzer for use on
        a new model and/or new inputs, keeping all operator handles and
        warning settings the same.

    .skipped_ops_warnings(enabled) : enables or disables warnings for
        skipped operators when analysis is first run. The number of
        skipped operators can be obtained from .skipped_ops() in either
        case. Defaults to enabled.

    .tracer_warnings(mode) : enables or disables warnings raised by the
        jit tracer when performing the analysis. 'mode' may be 'none',
        'no_tracer_warning', or 'all' to show no warnings, only warnings
        that aren't the TracerWarning type, or all warnings. Defaults to
        'no_tracer_warning'.


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
        additional_ops: Optional[Dict[str, Handle]] = None,
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
        op_handles = {**_DEFAULT_SUPPORTED_OPS, **(additional_ops or {})}
        super(FlopCountAnalysis, self).__init__(
            model=model, inputs=inputs, op_handles=op_handles
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

    flop_counter = FlopCountAnalysis(model, inputs, supported_ops)
    giga_flops = defaultdict(float)
    for op, flop in flop_counter.by_operator().items():
        giga_flops[op] = flop / 1e9
    return giga_flops, flop_counter.skipped_ops()
