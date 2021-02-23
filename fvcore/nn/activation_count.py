# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# pyre-ignore-all-errors[2,33]

from collections import defaultdict
from typing import Any, Counter, DefaultDict, Dict, Optional, Tuple, Union

import torch.nn as nn

from .jit_analysis import JitModelAnalysis
from .jit_handles import Handle, generic_activation_jit


# A dictionary that maps supported operations to their activation count handles.
_DEFAULT_SUPPORTED_OPS: Dict[str, Handle] = {
    "aten::_convolution": generic_activation_jit("conv"),
    "aten::addmm": generic_activation_jit("addmm"),
    "aten::linear": generic_activation_jit("linear"),
}


class ActivationCountAnalysis(JitModelAnalysis):
    """
    Provides access to per-submodule model activation count obtained by
    tracing a model with pytorch's jit tracing functionality. By default,
    comes with standard activation counters for two operators:

    aten::addmm, used by linear layers
    aten::_convolution, used by convolutional layers

    Handles for additional operators may be added, or the default ones
    overwritten, using the 'additional_ops' input at initialization or the
    .set_op_handle(name, func) method. The function must take the inputs
    and outputs of the op, each as a list(torch._C.Value), and returns a
    counter of the form {op_name : number}.All handles may be cleared with
    .clear_ops_handles().

    Activation counts can be obtained as:

    .total(module_name="") : total activation count for the module
    .by_operator(module_name="") : activation counts for the module, as a
        Counter over different operator types
    .by_module() : Counter of activation counts for all descendants
        of the model,
    .by_module_and_operator() : dictionary indexed by descendant of Counters
        over different operator types

    Submodules may be referred to using the module's name. The input model has
    name "", while its descendants have names of the form
    "child.grandchild.grandgrandchild...".

    An operator is treated as within the scope of a module if calling that
    module directly resulted in that operator being run. In particular,
    this means that calls to other functions owned by a module or explicit
    calls to module.forward(...) will not register resulting operators as
    contributing activations to that module.

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
        inputs: Tuple[Any, ...],
        additional_ops: Optional[Dict[str, Handle]] = None,
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
        op_handles = {**_DEFAULT_SUPPORTED_OPS, **(additional_ops or {})}
        super(ActivationCountAnalysis, self).__init__(
            model=model, inputs=inputs, op_handles=op_handles
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

    act_counter = ActivationCountAnalysis(model, inputs, supported_ops)
    mega_acts = defaultdict(float)
    for op, act in act_counter.by_operator().items():
        mega_acts[op] = act / 1e6
    return mega_acts, act_counter.skipped_ops()
