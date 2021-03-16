# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# pyre-ignore-all-errors[2,33]

from collections import defaultdict
from typing import Any, Counter, DefaultDict, Dict, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from .jit_analysis import JitModelAnalysis
from .jit_handles import Handle, generic_activation_jit


# A dictionary that maps supported operations to their activation count handles.
_DEFAULT_SUPPORTED_OPS: Dict[str, Handle] = {
    "aten::_convolution": generic_activation_jit("conv"),
    "aten::addmm": generic_activation_jit(),
    "aten::bmm": generic_activation_jit(),
    "aten::einsum": generic_activation_jit(),
    "aten::matmul": generic_activation_jit(),
    "aten::linear": generic_activation_jit(),
}


class ActivationCountAnalysis(JitModelAnalysis):
    """
    Provides access to per-submodule model activation count obtained by
    tracing a model with pytorch's jit tracing functionality. By default,
    comes with standard activation counters for convolutional and dot-product
    operators.

    Handles for additional operators may be added, or the default ones
    overwritten, using the ``.set_op_handle(name, func)`` method.
    See the method documentation for details.

    Activation counts can be obtained as:

    * ``.total(module_name="")``: total activation count for a module
    * ``.by_operator(module_name="")``: activation counts for the module, as a
      Counter over different operator types
    * ``.by_module()``: Counter of activation counts for all submodules
    * ``.by_module_and_operator()``: dictionary indexed by descendant of Counters
      over different operator types

    An operator is treated as within a module if it is executed inside the
    module's ``__call__`` method. Note that this does not include calls to
    other methods of the module or explicit calls to ``module.forward(...)``.

    Example usage:

    >>> import torch.nn as nn
    >>> import torch
    >>> class TestModel(nn.Module):
    ...     def __init__(self):
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
    >>> acts = ActivationCountAnalysis(model, inputs)
    >>> acts.total()
    1010
    >>> acts.total("fc")
    10
    >>> acts.by_operator()
    Counter({"conv" : 1000, "addmm" : 10})
    >>> acts.by_module()
    Counter({"" : 1010, "fc" : 10, "conv" : 1000, "act" : 0})
    >>> acts.by_module_and_operator()
    {"" : Counter({"conv" : 1000, "addmm" : 10}),
     "fc" : Counter({"addmm" : 10}),
     "conv" : Counter({"conv" : 1000}),
     "act" : Counter()
    }
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        super().__init__(model=model, inputs=inputs)
        self.set_op_handle(**_DEFAULT_SUPPORTED_OPS)

    __init__.__doc__ = JitModelAnalysis.__init__.__doc__


def activation_count(
    model: nn.Module,
    inputs: Tuple[Any, ...],
    supported_ops: Optional[Dict[str, Handle]] = None,
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
            number of unsupported operations.
    """
    if supported_ops is None:
        supported_ops = {}
    act_counter = ActivationCountAnalysis(model, inputs).set_op_handle(**supported_ops)
    mega_acts = defaultdict(float)
    for op, act in act_counter.by_operator().items():
        mega_acts[op] = act / 1e6
    return mega_acts, act_counter.unsupported_ops()
