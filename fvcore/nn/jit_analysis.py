# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pyre-ignore-all-errors[2,33]

import logging
import typing
import warnings
from collections import Counter
from copy import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from fvcore.common.checkpoint import _named_modules_with_dup
from torch import Tensor
from torch.jit import TracerWarning, _get_trace_graph

from .jit_handles import Handle


# Only ignore ops that are technically truly 0 flops:
# shape-manipulation ops, integer ops, memory copy ops
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
    "aten::narrow",
    "aten::unbind",
    "aten::full_like",
    "aten::stack",
    "aten::t",
    "aten::to",
    "aten::transpose",
    "aten::unsqueeze",
    "aten::unsqueeze_",
    "aten::view",
    "aten::zeros",
    "aten::zeros_like",
}


@dataclass
class Statistics:
    """
    For keeping track of the various model statistics recorded during
    analysis.
    """

    counts: "Dict[str, Counter[str]]"
    unsupported_ops: "Dict[str, Counter[str]]"
    uncalled_mods: "Set[str]"


def _get_scoped_trace_graph(
    module: nn.Module,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    aliases: Dict[Union[str, nn.Module], str],
) -> torch._C.Graph:  # pyre-ignore[11]
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
    """

    class ScopePushHook:
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, module: nn.Module, inputs: Any) -> Any:
            tracing_state = torch._C._get_tracing_state()
            if tracing_state:
                tracing_state.push_scope(self.name)
            return inputs

    class ScopePopHook:
        def __call__(self, module: nn.Module, inputs: Any, outputs: Any) -> Any:
            tracing_state = torch._C._get_tracing_state()
            if tracing_state:
                tracing_state.pop_scope()
            return outputs

    seen = set()
    hook_handles: List[Any] = []

    def register_hooks(mod: nn.Module, name: str) -> None:
        prehook = mod.register_forward_pre_hook(ScopePushHook(name))  # pyre-ignore[16]
        posthook = mod.register_forward_hook(ScopePopHook())  # pyre-ignore[16]
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

    # We don't need the duplication here, but self._model.named_modules()
    # gives slightly different results for some wrapped models.
    for name, mod in _named_modules_with_dup(module):
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


class JitModelAnalysis:
    """
    Provides access to per-submodule model statistics obtained by
    tracing a model with pytorch's jit tracing functionality. Calculates
    a statistic on a per-operator basis using the provided set of functions
    that acts on the inputs and outputs to the operator, then aggregates
    this over modules in the model. Can return the aggregate statistic for
    any submodule in the model. Is lazily evaluated, and will perform the
    trace when a statistic is first requested. Changing the operator handles
    will cause the trace to be rerun on the next request.

    Submodules may be referred to using the module's name. The input model has
    name "", while its descendants have names of the form
    "child.grandchild.grandgrandchild...".

    An operator is treated as within the scope of a module if calling that
    module directly resulted in that operator being run. In particular,
    this means that calls to other functions owned by a module or explicit
    calls to module.forward(...) will not register resulting operators as
    contributing statistics to that module.
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        """
        Args:
            model: The model to analyze
            inputs: The inputs to the model for analysis.
        """
        self._model = model
        self._inputs = inputs
        self._op_handles: Dict[str, Handle] = {}
        # Mapping from names to submodules
        self._named_modules: Dict[str, nn.Module] = dict(_named_modules_with_dup(model))
        # Mapping from submodules and their aliases to the canonical name of each submodule
        self._aliases: Dict[Union[nn.Module, str], str] = self._get_aliases(model)
        self._stats: Optional[Statistics] = None
        self._enable_warn_unsupported_ops = True
        self._enable_warn_uncalled_mods = True
        self._warn_trace = "no_tracer_warning"
        self._ignored_ops: Set[str] = copy(_IGNORED_OPS)

    def total(self, module_name: str = "") -> int:
        """
        Returns the total aggregated statistic across all operators
        for the requested module.

        Args:
            module_name (str) : The submodule to get data for. Defaults to
                the entire model.
        Returns:
            int : The aggregated statistic.
        """
        stats = self._analyze()
        module_name = self.canonical_module_name(module_name)
        total_count = sum(stats.counts[module_name].values())
        return total_count

    def by_operator(self, module_name: str = "") -> typing.Counter[str]:
        """
        Returns the statistics for a requested module, grouped by operator
        type. The operator handle determines the name associated with each
        operator type.

        Args:
            module_name (str) : The submodule to get data for. Defaults
                to the entire model.
        Returns:
            Counter(str) : The statistics for each operator.
        """
        stats = self._analyze()
        module_name = self.canonical_module_name(module_name)
        return stats.counts[module_name]

    def by_module_and_operator(self) -> Dict[str, typing.Counter[str]]:
        """
        Returns the statistics for all submodules, separated out by
        operator type for each submodule. The operator handle determines
        the name associated with each operator type.

        Returns:
            dict(str, Counter(str)):
                The statistics for each submodule and each operator.
                Grouped by submodule names, then by operator name.
        """
        stats = self._analyze()
        return stats.counts

    def by_module(self) -> typing.Counter[str]:
        """
        Returns the statistics for all submodules, aggregated over
        all operators.

        Returns:
            Counter(str): statistics counter grouped by submodule names
        """
        stats = self._analyze()
        summed_counts = Counter()
        for mod, results in stats.counts.items():
            summed_counts[mod] = sum(results.values())
        return summed_counts

    def unsupported_ops(self, module_name: str = "") -> typing.Counter[str]:
        """
        Lists the number of operators that were encountered but unsupported
        because no operator handle is available for them. Does not include
        operators that are explicitly ignored.

        Args:
            module_name (str) : The submodule to list unsupported ops.
                Defaults to the entire model.

        Returns:
            Counter(str) : The number of occurences each unsupported operator.
        """
        if self._stats is None:
            raise RuntimeError(
                "Analysis results should be computed "
                "before calling unsupported_ops()"
            )
        module_name = self.canonical_module_name(module_name)
        return self._stats.unsupported_ops[module_name]  # pyre-fixme

    def uncalled_modules(self) -> Set[str]:
        """
        Returns a set of submodules that were never called during the
        trace of the graph. This may be because they were unused, or
        because they were accessed via direct calls .forward() or with
        other python methods. In the latter case, statistics will not be
        attributed to the submodule, though the statistics will be included
        in the parent module.

        Returns:
            set(str) : The set of submodule names that were never called
                during the trace of the model.
        """
        stats = self._analyze()
        return stats.uncalled_mods

    def set_op_handle(self, *args, **kwargs: Optional[Handle]) -> "JitModelAnalysis":
        """
        Sets additional operator handles, or replaces existing ones.

        Args:
            args: (str, Handle) pairs of operator names and handles.
            kwargs: mapping from operator names to handles.

        If a handle is ``None``, the op will be explicitly ignored. Otherwise,
        handle should be a function that calculates the desirable statistic
        from an operator. The function must take two arguments, which are the
        inputs and outputs of the operator, in the form of ``list(torch._C.Value)``.
        The function should return a counter object with per-operator statistics.

        Examples
        ::
            handlers = {"aten::linear": my_handler}
            counter.set_op_handle("aten::matmul", None, "aten::bmm", my_handler2)
                   .set_op_handle(**handlers)
        """
        self._stats = None
        if len(args) % 2 != 0:
            raise TypeError(
                "set_op_handle should be called with pairs of names and handles!"
            )
        for name, handle in zip(args[::2], args[1::2]):
            kwargs[name] = handle
        for name, handle in kwargs.items():
            if handle is None:
                self._ignored_ops.add(name)
            else:
                self._op_handles[name] = handle
        return self

    def clear_op_handles(self) -> "JitModelAnalysis":
        """
        Clears all operator handles currently set.
        """
        self._op_handles = {}
        self._ignored_ops = copy(_IGNORED_OPS)
        self._stats = None
        return self

    def canonical_module_name(self, name: str) -> str:
        """
        Returns the canonical module name of the given ``name``, which might be
        different from the given ``name`` if the module is shared.
        This is the name that will be used as a key when statistics are
        output using .by_module() and .by_module_and_operator().

        Args:
            name (str) : The name of the module to find the canonical name for.
        Returns:
            str : The canonical name of the module.
        """
        # Blocks access by a direct module reference
        assert isinstance(name, str), "Module name must be a string."
        if name in self._aliases:
            return self._aliases[name]
        else:
            raise KeyError(
                "Requested module name is not among "
                "the descendants of the analyzed model."
            )

    def copy(
        self,
        new_model: Optional[nn.Module] = None,
        new_inputs: Union[None, Tensor, Tuple[Tensor, ...]] = None,
    ) -> "JitModelAnalysis":
        """
        Returns a copy of the :class:`JitModelAnalysis` object, keeping all
        settings, but on a new model or new inputs.

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
        return (
            JitModelAnalysis(model=model, inputs=inputs)
            .set_op_handle(**self._op_handles)
            .unsupported_ops_warnings(self._enable_warn_unsupported_ops)
            .uncalled_modules_warnings(self._enable_warn_uncalled_mods)
            .tracer_warnings(self._warn_trace)
        )

    def tracer_warnings(self, mode: str) -> "JitModelAnalysis":
        """
        Sets which warnings to print when tracing the graph to calculate
        statistics. There are three modes. Defaults to 'no_tracer_warning'.
        Allowed values are:

        * 'all' : keeps all warnings raised while tracing
        * 'no_tracer_warning' : suppress torch.jit.TracerWarning only
        * 'none' : suppress all warnings raised while tracing

        Args:
            mode (str) : warning mode in one of the above values.
        """
        assert mode in [
            "all",
            "no_tracer_warning",
            "none",
        ], "Unrecognized trace warning mode."
        self._warn_trace = mode
        return self

    def unsupported_ops_warnings(self, enabled: bool) -> "JitModelAnalysis":
        """
        Sets if warnings for unsupported operators are shown. Defaults
        to True. Counts of unsupported operators may be obtained from
        :meth:`unsupported_ops` regardless of this setting.

        Args:
            enabled (bool) : Set to 'True' to show unsupported operator
                warnings.
        """
        self._enable_warn_unsupported_ops = enabled
        return self

    def uncalled_modules_warnings(self, enabled: bool) -> "JitModelAnalysis":
        """
        Sets if warnings from uncalled submodules are shown. Defaults to true.
        A submodule is considered "uncalled" if it is never called during
        tracing. This may be because it is actually unused, or because it is
        accessed via calls to ``.forward()`` or other methods of the module.
        The set of uncalled modules may be obtained from
        :meth:`uncalled_modules` regardless of this setting.

        Args:
            enabled (bool) : Set to 'True' to show warnings.
        """
        self._enable_warn_uncalled_mods = enabled
        return self

    def _warn_unsupported_ops(self, ops: typing.Counter[str]) -> None:
        if not self._enable_warn_unsupported_ops:
            return
        logger = logging.getLogger(__name__)
        for op, freq in ops.items():
            logger.warning(
                "Unsupported operator {} encountered {} time(s)".format(op, freq)
            )

    def _warn_uncalled_mods(self, uncalled_mods: Set[str]) -> None:
        if not self._enable_warn_uncalled_mods or not uncalled_mods:
            return

        logger = logging.getLogger(__name__)
        logger.warning(
            "The following submodules of the model were never "
            "called during the trace of the graph. They may be "
            "unused, or they were accessed by direct calls to "
            ".forward() or via other python methods. In the latter "
            "case they will have zeros for statistics, though their "
            "statistics will still contribute to their parent calling "
            "module."
        )
        for mod in uncalled_mods:
            logger.warning("Module never called: {}".format(mod))

    def _get_aliases(self, model: nn.Module) -> Dict[Union[str, nn.Module], str]:
        aliases = {}
        for name, module in _named_modules_with_dup(model):
            if module not in aliases:
                aliases[module] = name
            aliases[name] = aliases[module]
        return aliases

    def _analyze(self) -> "Statistics":
        # Don't calculate if results are already stored.
        stats = self._stats
        if stats is not None:
            return stats

        with warnings.catch_warnings():
            if self._warn_trace == "none":
                warnings.simplefilter("ignore")
            elif self._warn_trace == "no_tracer_warning":
                warnings.filterwarnings("ignore", category=TracerWarning)
            graph = _get_scoped_trace_graph(self._model, self._inputs, self._aliases)

        # Assures even modules not in the trace graph are initialized to zero count
        counts = {}
        unsupported_ops = {}
        # We don't need the duplication here, but self._model.named_modules()
        # gives slightly different results for some wrapped models.
        for _, mod in _named_modules_with_dup(self._model):
            name = self._aliases[mod]
            counts[name] = Counter()
            unsupported_ops[name] = Counter()

        all_seen = set()
        for node in graph.nodes():
            kind = node.kind()
            scope_names = node.scopeName().split("/")
            all_seen.update(scope_names)
            if kind not in self._op_handles:
                # ignore all prim:: operators
                if kind in self._ignored_ops or kind.startswith("prim::"):
                    continue

                for name in set(scope_names):
                    unsupported_ops[name][kind] += 1
            else:
                inputs, outputs = list(node.inputs()), list(node.outputs())
                op_counts = self._op_handles[kind](inputs, outputs)

                # Assures an op contributes at most once to a module
                for name in set(scope_names):
                    counts[name] += op_counts

        uncalled_mods = set(self._aliases.values()) - all_seen

        def has_forward(module_type) -> bool:
            # Containers are not meant to be called anyway (they don't have forward)
            no_forward_mods = {nn.ModuleList, nn.ModuleDict, nn.Module}
            for mod in no_forward_mods:
                if module_type.forward is mod.forward:
                    return False
            return True

        uncalled_mods = {
            m for m in uncalled_mods if has_forward(type(self._named_modules.get(m)))
        }

        stats = Statistics(
            counts=counts, unsupported_ops=unsupported_ops, uncalled_mods=uncalled_mods
        )
        self._stats = stats
        self._warn_unsupported_ops(unsupported_ops[""])
        self._warn_uncalled_mods(uncalled_mods)
        return stats
