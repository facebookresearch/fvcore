# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# pyre-ignore-all-errors[2,33]

from collections import defaultdict
from copy import deepcopy
from typing import Any, Counter, DefaultDict, Dict, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from .jit_analysis import JitModelAnalysis, Statistics
from .jit_handles import (
    Handle,
    addmm_flop_jit,
    batchnorm_flop_jit,
    bmm_flop_jit,
    conv_flop_jit,
    einsum_flop_jit,
    elementwise_flop_counter,
    linear_flop_jit,
    matmul_flop_jit,
    norm_flop_counter,
)


# A dictionary that maps supported operations to their flop count jit handles.
_DEFAULT_SUPPORTED_OPS: Dict[str, Handle] = {
    "aten::addmm": addmm_flop_jit,
    "aten::bmm": bmm_flop_jit,
    "aten::_convolution": conv_flop_jit,
    "aten::einsum": einsum_flop_jit,
    "aten::matmul": matmul_flop_jit,
    "aten::mm": matmul_flop_jit,
    "aten::linear": linear_flop_jit,
    # Flops for the following ops are just estimates as they are not very well
    # defined and don't correlate with wall time very much. They shouldn't take
    # a big portion of any model anyway.
    # You might want to ignore BN flops due to inference-time fusion.
    # Use `set_op_handle("aten::batch_norm", None)
    "aten::batch_norm": batchnorm_flop_jit,
    "aten::group_norm": norm_flop_counter(2),
    "aten::layer_norm": norm_flop_counter(2),
    "aten::instance_norm": norm_flop_counter(1),
    "aten::upsample_nearest2d": elementwise_flop_counter(0, 1),
    "aten::upsample_bilinear2d": elementwise_flop_counter(0, 8),
    "aten::adaptive_avg_pool2d": elementwise_flop_counter(2, 0),
    "aten::grid_sampler": elementwise_flop_counter(0, 8),  # assume bilinear
}


class FlopCountAnalysis(JitModelAnalysis):
    """
    Provides access to per-submodule flop count obtained by tracing a model
    with pytorch's jit tracing functionality. By default, comes with standard
    flop counters for a few common operators.

    Flop represents floating point operations. Another common metric is MAC
    (multiply-add count), which represents a multiply and an add operations.
    We count MAC (multiply-add counts) by default, but this can be changed
    by `set_use_mac(False)`. We just assume MAC is half of flops, which
    is true for most expensive operators we care.

    Note that flop/MAC is not a well-defined concept for many ops. We just produce
    our best estimate.

    Handles for additional operators may be added, or the default ones
    overwritten, using the ``.set_op_handle(name, func)`` method.
    See the method documentation for details.
    The handler for each op should always calculate flops instead of MAC.

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
        inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        super().__init__(model=model, inputs=inputs)
        self.set_op_handle(**_DEFAULT_SUPPORTED_OPS)
        self._use_mac = True  # NOTE: maybe we'll want to change the default to False

    def set_use_mac(self, enabled: bool) -> "FlopCountAnalysis":
        """
        Decide whether to count MAC (multiply-add counts) rather than flops.
        Default to True because this is the convention in many computer vision papers.
        Unfortunately this concept is typically misused as flops.

        To implement counting of MAC, we simply assume MAC is half of flops.
        Although we note that this is not true for all ops.
        """
        self._use_mac = enabled
        return self

    def _analyze(self) -> Statistics:
        stats = super()._analyze()
        if self._use_mac:
            stats = deepcopy(stats)
            for v in stats.counts.values():
                for k in list(v.keys()):
                    v[k] = v[k] // 2
        return stats

    __init__.__doc__ = JitModelAnalysis.__init__.__doc__


def flop_count(
    model: nn.Module,
    inputs: Tuple[Any, ...],
    supported_ops: Optional[Dict[str, Handle]] = None,
) -> Tuple[DefaultDict[str, float], Counter[str]]:
    """
    Given a model and an input to the model, compute the per-operator GMACs
    (10^9 multiply-adds) of the given model.

    For more features and customized counting, please use :class:`FlopCountAnalysis`.

    Args:
        model (nn.Module): The model to compute flop counts.
        inputs (tuple): Inputs that are passed to `model` to count flops.
            Inputs need to be in a tuple.
        supported_ops (dict(str,Callable) or None) : provide additional
            handlers for extra ops, or overwrite the existing handlers for
            convolution and matmul and einsum. The key is operator name and the value
            is a function that takes (inputs, outputs) of the op.

    Returns:
        tuple[defaultdict, Counter]: A dictionary that records the number of
            GMACs for each operation and a Counter that records the number of
            unsupported operations.
    """
    if supported_ops is None:
        supported_ops = {}
    mac_counter = (
        FlopCountAnalysis(model, inputs)  # pyre-ignore
        .set_op_handle(**supported_ops)
        .set_use_mac(True)
    )
    giga_macs = defaultdict(float)
    for op, mac in mac_counter.by_operator().items():
        giga_macs[op] = mac / 1e9
    return giga_macs, mac_counter.unsupported_ops()
