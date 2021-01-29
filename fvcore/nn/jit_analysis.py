import logging
import typing
import warnings
from collections import Counter
from copy import copy

import torch
import torch.nn as nn
from torch.jit import _get_trace_graph


_IGNORED_OPS: typing.Set[str] = {
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
    module: nn.Module, inputs: typing.Tuple[object, ...]
) -> typing.Tuple[torch._C.Graph, typing.Dict[str or nn.Module, str]]:
    """
    Traces the provided module using torch.jit._get_trace_graph, but adds
    submodule scope information to each graph node. The resulting graph
    is in-lined and has all model parameters treated as inputs. The input
    model has the scope name '', while its descendants have names of the
    form 'child.grandchild.grandgrandchild...'.

    Args:
        model (nn.Module) : The module to trace
        inputs (tuple) : Inputs used during the trace of the model

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
            self, module: nn.Module, inputs: typing.Tuple[object, ...]
        ) -> typing.Tuple[object, ...]:
            tracing_state = torch._C._get_tracing_state()
            if tracing_state:
                name = self.name
                tracing_state.push_scope(name)
            return inputs

    class ScopePopHook(object):
        def __call__(
            self,
            module: nn.Module,
            inputs: typing.Tuple[object, ...],
            outputs: typing.Tuple[object, ...],
        ) -> typing.Tuple[object, ...]:
            tracing_state = torch._C._get_tracing_state()
            if tracing_state:
                tracing_state.pop_scope()
            return outputs

    aliases = {}
    hook_handles = []

    def register_hooks(mod: nn.Module, name: str) -> None:
        prehook = mod.register_forward_pre_hook(ScopePushHook(name))
        posthook = mod.register_forward_hook(ScopePopHook())
        hook_handles.append(prehook)
        hook_handles.append(posthook)

    # This naming scheme for scopes matches the structure of
    # fvcore.nn.parameter_count. Names may not be completely identical
    # since the module structure may not be walked in the same order.
    # It does not match the naming scheme used by torch.jit.trace(...).
    def recurse_hooks(mod: nn.Module, name: str) -> None:
        if mod not in aliases:
            register_hooks(mod, name)
            aliases[mod] = name
        else:
            # We've seen this submodule before. Add the new name to aliases
            aliases[name] = aliases[mod]
        for subname, child in mod.named_children():
            if name != "":
                subname = name + "." + subname
            recurse_hooks(child, subname)

    # Torch script does not support parallel torch models, but we still
    # want the scope names to be correct for the complete module.
    if isinstance(
        module, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)
    ):
        aliases[module] = ""
        module = module.module
        register_hooks(module, "")  # Push scope for the removed DataParallel wrapper
        recurse_hooks(module, "module")
    else:
        recurse_hooks(module, "")

    if hasattr(torch.jit, "get_trace_graph"):
        trace, _ = torch.jit.get_trace_graph(module, inputs)
        graph = trace.graph()
    else:
        graph, _ = _get_trace_graph(module, inputs)

    for handle in hook_handles:
        handle.remove()

    return graph, aliases


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
        inputs: typing.Tuple[object, ...],
        ops_handles: typing.Dict[str, typing.Callable] = {},
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
        self.counts = None
        self._skipped_ops = None
        self.warn_skipped = True
        self.warn_trace = True
        self.scale = 1

    def total(self, module: str or nn.Module = "") -> int or float:
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
        total_count = sum([c for c in self.counts[module].values()])
        return self._rescale_count(total_count)

    def by_operator(self, module: str or nn.Module = "") -> typing.Counter[str]:
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
        return self._rescale_dict(self.counts[module])

    def by_module_and_operator(self) -> typing.Dict[str, typing.Counter[str]]:
        """
        Returns the statistics for all submodules, separated out by
        operator type for each submodule. The operator handle determines
        the name associated with each operator type.

        Returns:
            dict(str, Counter(str)) : The statistics for each submodule
                and each operator. Organized per-module, labelled
                by the submodule's name, then by operator name.
        """
        self._analyze()
        self._warn_skipped_ops("")
        counts = {}
        for mod in self.counts:
            counts[mod] = self._rescale_dict(self.counts[mod])
        return counts

    def by_module(self) -> typing.Counter[str]:
        """
        Returns the statistics for all submodules, aggregated over
            all operators.

        Returns:
            Counter(str) : The statistics for each submodule
                and each operator. Organized per-module, labelled
                by the submodule's name.
        """
        self._analyze()
        self._warn_skipped_ops("")
        summed_counts = Counter()
        for mod, results in self.counts.items():
            summed_counts[mod] = sum([c for c in results.values()])
        return self._rescale_dict(summed_counts)

    def skipped_ops(self, module: str or nn.Module = "") -> typing.Counter[str]:
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

    def set_ops_handle(self, name: str, func: typing.Callable) -> None:
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

    def set_output_scale(self, scale: str or float) -> None:
        """
        Sets the scale of the output statistics.

        Args:
            scale (str or float) : The scale to output statistics with.
                Can be a string in ['unity', 'kilo', 'mega', 'giga',
                'tera', 'peta'] or a number to divide results by.
        """
        named_scales = {
            "unity": 1,
            "kilo": 1e3,
            "mega": 1e6,
            "giga": 1e9,
            "tera": 1e12,
            "peta": 1e15,
        }
        if scale in named_scales:
            scale = named_scales[scale]
        assert not isinstance(
            scale, str
        ), "Unrecognized scale name. Must be in {}.".format(list(named_scales.keys()))
        self.scale = scale

    def get_output_scale(self) -> float:
        """
        Gets the output scale of the statistics.

        Returns:
            float : the output scale divided by
        """
        return self.scale

    def copy(
        self,
        new_model: nn.Module or None = None,
        new_inputs: typing.Tuple[object, ...] or None = None,
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
        analyzer = copy(self)
        analyzer.model = model
        analyzer.inputs = inputs
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

    @property
    def model(self) -> nn.Module:
        return self._model

    @model.setter
    def model(self, new_model: nn.Module) -> None:
        self._model = new_model
        self.counts = None

    @property
    def inputs(self) -> typing.Tuple[object, ...]:
        return self._inputs

    @inputs.setter
    def inputs(self, new_inputs: typing.Tuple[object, ...]) -> None:
        self._inputs = new_inputs
        self.counts = None

    @property
    def ops_handles(self) -> typing.Dict[str, typing.Callable]:
        return self.__ops_handles

    @ops_handles.setter
    def ops_handles(self, new_ops_handles: typing.Dict[str, typing.Callable]) -> None:
        self._ops_handles = new_ops_handles
        self.counts = None

    def _rescale_count(self, count: int) -> float:
        if self.scale == 1:
            return count  # Maintain integer-valued counters
        return count / self.scale

    def _rescale_dict(self, count_dict: typing.Dict) -> typing.Dict:
        scaled_dict = {k: self._rescale_count(count) for k, count in count_dict.items()}
        if isinstance(count_dict, Counter):
            scaled_dict = Counter(scaled_dict)
        return scaled_dict

    def _warn_skipped_ops(self, module: str or nn.Module) -> None:
        if not self.warn_skipped:
            return
        logger = logging.getLogger(__name__)
        skipped_ops = self._skipped_ops[module]
        if len(skipped_ops) > 0:
            for op, freq in skipped_ops.items():
                logger.warning("Skipped operation {} {} time(s)".format(op, freq))

    def _canonical_module_name(self, module: str or nn.Module) -> str:
        if module in self.aliases:
            return self.aliases[module]
        return module

    def _analyze(self) -> None:
        # Don't calculate if results are already stored.
        if self.counts is not None:
            return

        with warnings.catch_warnings():
            if not self.warn_trace:
                warnings.simplefilter("ignore")
            graph, self.aliases = _get_scoped_trace_graph(self._model, self._inputs)

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
