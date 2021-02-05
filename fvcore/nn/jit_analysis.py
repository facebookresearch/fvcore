import logging
import typing
import warnings
from collections import Counter
from copy import copy
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from fvcore.common.checkpoint import _named_modules_with_dup
from torch.jit import _get_trace_graph


_IGNORED_OPS: Set[str] = {
    "aten::Int",
    "aten::ScalarImplicit",
    "aten::__and__",
    "aten::arange",
    "aten::cat",
    "aten::chunk",
    "aten::clamp",
    "aten::clamp_",
    "aten::constant_pad_nd",
    "aten::contiguous",
    "aten::copy_",
    "aten::detach",
    "aten::dropout",
    "aten::empty",
    "aten::eq",
    "aten::expand",
    "aten::flatten",
    "aten::floor",
    "aten::floor_divide",
    "aten::full",
    "aten::ge",
    "aten::gt",
    "aten::index",
    "aten::index_put_",
    "aten::max",
    "aten::nonzero",
    "aten::permute",
    "aten::relu",
    "aten::relu_",
    "aten::remainder",
    "aten::reshape",
    "aten::select",
    "aten::size",
    "aten::slice",
    "aten::split",
    "aten::split_with_sizes",
    "aten::squeeze",
    "aten::stack",
    "aten::t",
    "aten::to",
    "aten::transpose",
    "aten::unsqueeze",
    "aten::unsqueeze_",
    "aten::view",
    "aten::zeros",
    "aten::zeros_like",
    "prim::Constant",
    "prim::ImplicitTensorToNum",
    "prim::Int",
    "prim::ListConstruct",
    "prim::ListUnpack",
    "prim::NumToTensor",
    "prim::TupleConstruct",
}


def _get_scoped_trace_graph(
    module: nn.Module,
    inputs: Tuple[object, ...],
    aliases: Dict[Union[str, nn.Module], str],
) -> typing.Tuple[torch._C.Graph, Dict[Union[str, nn.Module], str]]:
    """
    Traces the provided module using torch.jit._get_trace_graph, but adds
    submodule scope information to each graph node. The resulting graph
    is in-lined and has all model parameters treated as inputs. The input
    model has the scope name '', while its descendants have names of the
    form 'child.grandchild.grandgrandchild...'.

    Args:
        model (nn.Module) : The module to trace
        inputs (tuple) : Inputs used during the trace of the model
        aliases (dict(str or nn.Module, str) : maps modules and module
            names to the canonical name to be used as the scope for
            that module.

    Returns:
        graph (torch._C.Graph) : The pytorch JIT trace of the model
        aliases (dict) : A dictionary of alternative names for each
            submodule. This maps both the nn.module object itself to
            its scope name in the graph, but also includes mappings
            of alternative scope names if a module appears as a chile
            of multiple different submodules.
    """

    class ScopePushHook(object):
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(
            self, module: nn.Module, inputs: Tuple[object, ...]
        ) -> Tuple[object, ...]:
            tracing_state = torch._C._get_tracing_state()
            if tracing_state:
                name = self.name
                tracing_state.push_scope(name)
            return inputs

    class ScopePopHook(object):
        def __call__(
            self,
            module: nn.Module,
            inputs: Tuple[object, ...],
            outputs: Tuple[object, ...],
        ) -> Tuple[object, ...]:
            tracing_state = torch._C._get_tracing_state()
            if tracing_state:
                tracing_state.pop_scope()
            return outputs

    seen = set()
    hook_handles = []

    def register_hooks(mod: nn.Module, name: str) -> None:
        prehook = mod.register_forward_pre_hook(ScopePushHook(name))
        posthook = mod.register_forward_hook(ScopePopHook())
        hook_handles.append(prehook)
        hook_handles.append(posthook)

    # Torch script does not support parallel torch models, but we still
    # want the scope names to be correct for the complete module.
    if isinstance(
        module, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)
    ):

        # Since DataParallel just wraps the model, add an extra set of hooks
        # to the model it wraps to account for the wrapper. Then trace it.
        root_name = aliases[module]
        module = module.module
        register_hooks(module, root_name)

    for name, mod in module.named_modules():
        if mod not in seen:
            name = aliases[mod]
            register_hooks(mod, name)
            seen.add(mod)

    if hasattr(torch.jit, "get_trace_graph"):
        trace, _ = torch.jit.get_trace_graph(module, inputs)
        graph = trace.graph()
    else:
        graph, _ = _get_trace_graph(module, inputs)

    for handle in hook_handles:
        handle.remove()

    return graph


class JitModelAnalysis(object):
    """
    Provides access to per-submodule model statistics obtained by
    tracing a model with pytorch's jit tracing functionality. Calculates
    a statistic on a per-operator basis using the provided set of functions
    that acts on the inputs and outputs to the operator, then aggregates
    this over modules in the model. Can return the aggregate statistic for
    any submodule in the model. Is lazily evaluated, and will perform the
    trace when a statistic is first requested. Changing the model, inputs,
    or operator handles will cause the trace to be rerun on the next
    request.

    Submodules may be referred to using the nn.Module object itself, or
    a string associated with the module's name. The input model has name
    '', while its descendants have names of the form
    'child.grandchild.grandgrandchild...'.
    """

    ignored_ops = _IGNORED_OPS

    def __init__(
        self,
        model: nn.Module,
        inputs: Tuple[object, ...],
        ops_handles: Optional[Dict[str, typing.Callable]] = None,
    ) -> None:
        """
        Args:
            model (nn.Module) : The model to analyze
            inputs (tuple) : The inputs to the model for analysis.
            ops_handles (dict(str,Callable) : Map an operator name in the
                trace graph to a function used to calculate the desire
                statistic. The function must take the inputs and outputs
                of the op, each as a list(torch._C.Value), and returns
                a counter of the form {op_name : number}.
        """
        self._model = model
        self._inputs = inputs
        self._ops_handles = ops_handles
        self.aliases = self._get_aliases(model)
        self.counts = None
        self._skipped_ops = None
        self.warn_skipped = True
        self.warn_trace = False

    def total(self, module: Union[str, nn.Module] = "") -> float:
        """
        Returns the total aggregated statistic across all operators
        for the requested module.

        Args:
            module (nn.Module or str) : The submodule to get data for.
                Defaults to the entire model.
        Returns:
            int or float : The aggregated statistic
        """
        self._analyze()
        module = self._canonical_module_name(module)
        self._warn_skipped_ops(module)
        total_count = sum(self.counts[module].values())
        return total_count

    def by_operator(self, module: Union[str, nn.Module] = "") -> typing.Counter[str]:
        """
        Returns the statistics for a requested module, separated out by
        operator type. The operator handle determines the name associated
        with each operator type.

        Args:
            module (nn.Module or str) : The submodule to get data for.
                Defaults to the entire model.
        Returns:
            Counter(str) : The statistics for each operator.
        """
        self._analyze()
        module = self._canonical_module_name(module)
        self._warn_skipped_ops(module)
        return self.counts[module]

    def by_module_and_operator(self) -> Dict[str, typing.Counter[str]]:
        """
        Returns the statistics for all submodules, separated out by
        operator type for each submodule. The operator handle determines
        the name associated with each operator type.

        Returns:
            dict(str, Counter(str)) : The statistics for each submodule
                and each operator. Organized per-module, labeled
                by the submodule's name, then by operator name.
        """
        self._analyze()
        self._warn_skipped_ops("")
        return self.counts

    def by_module(self) -> typing.Counter[str]:
        """
        Returns the statistics for all submodules, aggregated over
            all operators.

        Returns:
            Counter(str) : The statistics for each submodule
                and each operator. Organized per-module, labeled
                by the submodule's name.
        """
        self._analyze()
        self._warn_skipped_ops("")
        summed_counts = Counter()
        for mod, results in self.counts.items():
            summed_counts[mod] = sum(results.values())
        return summed_counts

    def skipped_ops(self, module: Union[str, nn.Module] = "") -> typing.Counter[str]:
        """
        Lists the number of operators that were skipped because no
        operator handle existed for them. Does not include operators
        listed in _IGNORED_OPS.

        Args:
            module (nn.Module or str) : The submodule to skipped ops for.
                Defaults to the entire model.
        Returns:
            Counter(str) : The number of each type of operator skipped.
        """
        self._analyze()
        module = self._canonical_module_name(module)
        return self._skipped_ops[module]

    def set_ops_handle(self, name: str, func: Callable) -> None:
        """
        Sets an additional operator handle, or replacing an existing one.

        Args:
            name (str) : The operator's name
            func (Callable) : Function that calculates the desirable
                statistic from an operator. Must take two arguments,
                which are the inputs and outputs of the operator, in the
                form of list(torch._C.Value).
        """
        self._ops_handles[name] = func
        self.counts = None

    def clear_ops_handles(self) -> None:
        """
        Clears all set operator handles.
        """
        self._ops_handles = {}
        self.counts = None

    def copy(
        self,
        new_model: Optional[nn.Module] = None,
        new_inputs: Tuple[object, ...] or None = None,
    ) -> "JitModelAnalysis":
        """
        Returns a copy of the JitModelAnalysis object, keeping all
        settings, but potentially on a new model or new inputs.

        Args:
            new_model (nn.Module or None) : a new model for the new
                JitModelAnalysis. If None, uses the original model.
            new_inputs (typing.Tuple[object, ...] or None) : new inputs
                for the new JitModelAnalysis. If None, uses the original
                inputs.
        Returns:
            JitModelAnalysis : the new model analysis object
        """
        model = self._model if new_model is None else new_model
        inputs = self._inputs if new_inputs is None else new_inputs
        analyzer = JitModelAnalysis(
            model=model, inputs=inputs, ops_handles=self._ops_handles
        )
        analyzer.warn_skipped = self.warn_skipped
        analyzer.warn_trace = self.warn_trace

        return analyzer

    def tracer_warnings(self, enabled: bool) -> None:
        """
        Sets if warnings from the jit tracing process are shown. Defaults
        to True.

        Args:
            enabled (bool) : Set to 'True' to show tracer warnings
        """
        self.warn_trace = enabled

    def skipped_ops_warnings(self, enabled: bool) -> None:
        """
        Sets if warnings from skipped operators are shown. Defaults
        to True. Counts of skipped operators may be obtained from
        .skipped_ops(module) regardless of this setting.

        Args:
            enabled (bool) : Set to 'True' to show skipped operator
                warnings.
        """
        self.warn_skipped = enabled

    def _warn_skipped_ops(self, module: Union[str, nn.Module]) -> None:
        if not self.warn_skipped:
            return
        logger = logging.getLogger(__name__)
        skipped_ops = self._skipped_ops[module]
        if len(skipped_ops) > 0:
            for op, freq in skipped_ops.items():
                logger.warning("Skipped operation {} {} time(s)".format(op, freq))

    def _canonical_module_name(self, module: Union[str, nn.Module]) -> str:
        if module in self.aliases:
            return self.aliases[module]
        else:
            raise KeyError(
                "Requested module or module name is not among "
                "the descendants of the analyzed model."
            )

    def _get_aliases(self, model: nn.Module) -> Dict[Union[str, nn.Module], str]:
        aliases = {}
        for name, module in _named_modules_with_dup(model):
            if module not in aliases:
                aliases[module] = name
            aliases[name] = aliases[module]
        return aliases

    def _analyze(self) -> None:
        # Don't calculate if results are already stored.
        if self.counts is not None:
            return

        with warnings.catch_warnings():
            if not self.warn_trace:
                warnings.simplefilter("ignore")
            graph = _get_scoped_trace_graph(self._model, self._inputs, self.aliases)

        # Assures even modules not in the trace graph are initialized to zero count
        counts = {}
        skipped_ops = {}
        for _, mod in self._model.named_modules():
            name = self.aliases[mod]
            counts[name] = Counter()
            skipped_ops[name] = Counter()

        for node in graph.nodes():
            kind = node.kind()
            scope_names = node.scopeName().split("/")
            if kind not in self._ops_handles:
                if kind in self.ignored_ops:
                    continue

                for name in scope_names:
                    skipped_ops[name][kind] += 1
            else:
                inputs, outputs = list(node.inputs()), list(node.outputs())
                op_counts = self._ops_handles[kind](inputs, outputs)

                for name in scope_names:
                    counts[name] += op_counts

        self.counts = counts
        self._skipped_ops = skipped_ops
