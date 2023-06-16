# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pyre-ignore-all-errors[2,56]

import logging
import typing
import unittest
import warnings
from collections import Counter
from typing import Any, Dict, List

import torch
import torch.nn as nn

from fvcore.nn.flop_count import FlopCountAnalysis
from fvcore.nn.jit_analysis import JitModelAnalysis
from fvcore.nn.jit_handles import addmm_flop_jit, conv_flop_jit, Handle, linear_flop_jit


class NestedNetInnerModule(nn.Module):
    """
    A submodule for the nested net test module below.
    """

    def __init__(self, lin_op: str = "addmm") -> None:
        super().__init__()
        conv_input_size = (2, 5)
        conv_in = 2
        conv_out = 2
        kernel_size = 1
        padding = 0
        fc_in = 10
        fc_out = 10

        self.conv = nn.Conv1d(
            in_channels=conv_in,
            out_channels=conv_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.fc = nn.Linear(in_features=fc_in, out_features=fc_out)

        fc_flops = fc_in * fc_out
        fc_flops = Counter({lin_op: fc_flops})
        spatial_pos = (conv_input_size[1] + 2 * padding) - 2 * (kernel_size // 2)
        conv_flops = spatial_pos * kernel_size * conv_in * conv_out
        conv_flops = Counter({"conv": conv_flops})
        model_flops = conv_flops + fc_flops
        self.flops: "Dict[str, typing.Counter[str]]" = {
            "": model_flops,
            "fc": fc_flops,
            "conv": conv_flops,
        }

        self.name_to_module: "Dict[str, nn.Module]" = {
            "": self,
            "fc": self.fc,
            "conv": self.conv,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 2, 5)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = 3 * self.fc(x) + 1
        return x


class NestedNet(nn.Module):
    """
    A network with nested submodules for testing the ability to correctly
    capture scope information.
    """

    def __init__(self, lin_op: str = "addmm") -> None:
        super().__init__()
        self.input_size = (4, 5)

        conv_in = 4
        conv_out = 4
        kernel_size = 3
        padding = 1
        fc_in = 20
        fc_out = 10

        self.submod = NestedNetInnerModule(lin_op)
        self.fc = nn.Linear(in_features=fc_in, out_features=fc_out)
        self.conv = nn.Conv1d(
            in_channels=conv_in,
            out_channels=conv_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        fc_flops = fc_in * fc_out
        fc_flops = Counter({lin_op: fc_flops})
        spatial_pos = (self.input_size[1] + 2 * padding) - 2 * (kernel_size // 2)
        conv_flops = spatial_pos * kernel_size * conv_in * conv_out
        conv_flops = Counter({"conv": conv_flops})

        model_flops = conv_flops + fc_flops + self.submod.flops[""]
        self.flops: "Dict[str, Counter[str]]" = {
            "": model_flops,
            "fc": fc_flops,
            "conv": conv_flops,
            "submod": self.submod.flops[""],
            "submod.fc": self.submod.flops["fc"],
            "submod.conv": self.submod.flops["conv"],
        }

        self.name_to_module: "Dict[str, nn.Module]" = {
            "": self,
            "fc": self.fc,
            "conv": self.conv,
            "submod": self.submod,
            "submod.fc": self.submod.name_to_module["fc"],
            "submod.conv": self.submod.name_to_module["conv"],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.submod(x) ** 2
        return x


class UnusedNet(nn.Module):
    """
    Has a submodule that is never called in the forward function.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_size = (10,)
        fc1_in, fc1_out = 10, 10
        fc2_in, fc2_out = 10, 1
        unused_in, unused_out = 20, 20

        self.fc1 = nn.Linear(in_features=fc1_in, out_features=fc1_out)
        self.fc2 = nn.Linear(in_features=fc2_in, out_features=fc2_out)
        self.unused = nn.Linear(in_features=unused_in, out_features=unused_out)
        self.act: "nn.Module" = nn.ReLU()

        self.fc1_flops: int = fc1_in * fc1_out
        self.fc2_flops: int = fc2_in * fc2_out
        self.unused_flops: int = unused_in * unused_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class RepeatedNet(nn.Module):
    """
    Makes repeated calls to the same submodule.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_size = (10,)
        fc1_in, fc1_out = 10, 10
        fc2_in, fc2_out = 10, 10
        self.fc1_num = 3
        self.fc2_num = 2

        self.fc1 = nn.Linear(in_features=fc1_in, out_features=fc1_out)
        self.fc2 = nn.Linear(in_features=fc2_in, out_features=fc2_out)

        self.fc1_flops: int = fc1_in * fc1_out
        self.fc2_flops: int = fc2_in * fc2_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _i in range(self.fc1_num):
            x = self.fc1(x)
        for _i in range(self.fc2_num):
            x = self.fc2(x)
        return x


class NonForwardInnerModule(nn.Module):
    """
    Has a function separate from the forward function.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_size = (10,)
        fc_in, fc_out = 10, 1

        self.fc = nn.Linear(in_features=fc_in, out_features=fc_out)

        self.fc_flops: int = fc_in * fc_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def other_func(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class NonForwardNet(nn.Module):
    """
    The submodule has a non-forward function called by the parent module.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_size = (10,)
        fc_in, fc_out = 10, 10

        self.submod = NonForwardInnerModule()
        self.fc = nn.Linear(in_features=fc_in, out_features=fc_out)

        self.fc_flops: int = fc_in * fc_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.submod.other_func(self.fc(x))


class SharedInnerModule(nn.Module):
    """
    Is initialized with a module that it may share with other modules.
    """

    def __init__(self, submod: nn.Module) -> None:
        super().__init__()
        self.submod = submod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.submod(x)


class SharedModuleNet(nn.Module):
    """
    A subsubmodule is shared by multiple submodules. Also calls a module
    using multiple names.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_size = (10,)
        fc1_in, fc1_out = 10, 10
        fc2_in, fc2_out = 10, 1

        inner = nn.Linear(in_features=fc1_in, out_features=fc1_out)
        self.submod1 = SharedInnerModule(inner)
        self.submod2 = SharedInnerModule(inner)
        multiname = nn.Linear(in_features=fc2_in, out_features=fc2_out)
        self.multiname1: "nn.Module" = multiname
        self.multiname2: "nn.Module" = multiname

        self.multiname_flops: int = fc2_in * fc2_out
        self.shared_flops: int = fc1_in * fc1_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.submod1(x) + self.submod2(x)
        x = self.multiname1(x) + self.multiname2(x)
        return x


class RecursiveScopeNet(nn.Module):
    """
    An op is in the same module's scope multiple times.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_size = (10,)
        fc_in, fc_out = 10, 1

        self.fc = nn.Linear(in_features=fc_in, out_features=fc_out)

        self.flops: int = fc_in * fc_out

    def forward(self, x: torch.Tensor, count: int = 3) -> torch.Tensor:
        if count > 0:
            return self(x, count - 1)
        return self.fc(x)


class TraceWarningNet(nn.Module):
    """
    Will raise a warning on trace due to python comparison of tensor data,
    and explicitly raises a runtime warning. Also has an aten::add op that
    will be skipped and raise a warning.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_size = (10,)
        fc1_in, fc1_out = 10, 1
        fc2_in, fc2_out = 10, 10

        self.fc1 = nn.Linear(in_features=fc1_in, out_features=fc1_out)
        self.fc2 = nn.Linear(in_features=fc2_in, out_features=fc2_out)

        self.fc1_flops: int = fc1_in * fc1_out
        self.fc2_flops: int = fc2_in * fc2_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x).item()
        warnings.warn("Dummy RuntimeWarning.", RuntimeWarning)
        if y < 0.0:
            x = self.fc2(x)
        return x + 2


class TestJitModelAnalysis(unittest.TestCase):
    """
    Unittest for JitModelAnalysis. Tests for specific jit_handles are
    covered in test_flop_count.py and test_activation_count.py.
    """

    def setUp(self) -> None:
        # nn.Linear uses a different operator based on version, so make sure
        # we are testing the right thing.
        lin = nn.Linear(10, 10)
        lin_x: torch.Tensor = torch.randn(10, 10)
        trace = torch.jit.trace(lin, (lin_x,))
        node_kinds = [node.kind() for node in trace.graph.nodes()]
        assert "aten::addmm" in node_kinds or "aten::linear" in node_kinds
        if "aten::addmm" in node_kinds:
            self.lin_op = "addmm"
        else:
            self.lin_op = "linear"

    def test_total(self) -> None:
        """
        Tests that JitModelAnalysis.total(module) returns the correct
        counts for string and module inputs.
        """

        model = NestedNet(lin_op=self.lin_op)
        inputs = (torch.randn((1, *model.input_size)),)

        analyzer = FlopCountAnalysis(model=model, inputs=inputs)
        analyzer.unsupported_ops_warnings(enabled=False)

        # Using a string input
        for name in model.flops:
            with self.subTest(name=name):
                gt_flops = sum(model.flops[name].values())
                self.assertEqual(analyzer.total(name), gt_flops)

    def test_by_module(self) -> None:
        """
        Tests that JitModelAnalysis.by_module() returns the correct
        counts in the correctly structured dictionary.
        """

        model = NestedNet(lin_op=self.lin_op)
        inputs = (torch.randn((1, *model.input_size)),)

        analyzer = FlopCountAnalysis(model=model, inputs=inputs)
        analyzer.unsupported_ops_warnings(enabled=False)

        flops = {name: sum(counts.values()) for name, counts in model.flops.items()}

        self.assertEqual(analyzer.by_module(), flops)

    def test_by_operator(self) -> None:
        """
        Tests that JitModelAnalysis.by_operator(module) returns the correct
        counts for string and module inputs.
        """

        model = NestedNet(lin_op=self.lin_op)
        inputs = (torch.randn((1, *model.input_size)),)

        analyzer = FlopCountAnalysis(model=model, inputs=inputs)
        analyzer.unsupported_ops_warnings(enabled=False)

        # Using a string input
        for name in model.flops:
            with self.subTest(name=name):
                self.assertEqual(analyzer.by_operator(name), model.flops[name])

    def test_by_module_and_operator(self) -> None:
        """
        Tests that JitModelAnalysis.by_module_and_operator() returns
        the correct counts in the correct structure.
        """

        model = NestedNet(lin_op=self.lin_op)
        inputs = (torch.randn((1, *model.input_size)),)

        analyzer = FlopCountAnalysis(model=model, inputs=inputs)
        analyzer.unsupported_ops_warnings(enabled=False)

        self.assertEqual(analyzer.by_module_and_operator(), model.flops)

    def test_unused_module(self) -> None:
        """
        Tests that unused modules return 0 count for operator sums and
        and empty Counter() for per-operator results. Also tests that
        unused modules are reported by .uncalled_modules(), but that
        modules that simply have zero flops (like ReLU) are not.
        """

        model = UnusedNet()
        inputs = (torch.randn((1, *model.input_size)),)
        analyzer = FlopCountAnalysis(model=model, inputs=inputs)

        unused_count = 0
        unused_per_operator = Counter()
        model_count = model.fc1_flops + model.fc2_flops

        self.assertEqual(analyzer.total("unused"), unused_count)
        self.assertEqual(analyzer.by_operator("unused"), unused_per_operator)
        self.assertEqual(analyzer.total(""), model_count)

        # The unused mod is recognized as never called
        self.assertEqual(analyzer.uncalled_modules(), {"unused"})

    def test_repeated_module(self) -> None:
        """
        Tests that repeated calls to the same submodule correct aggregates
        results to that submodule.
        """

        model = RepeatedNet()
        inputs = (torch.randn((1, *model.input_size)),)

        analyzer = FlopCountAnalysis(model=model, inputs=inputs)
        fc1_count = model.fc1_num * model.fc1_flops
        fc2_count = model.fc2_num * model.fc2_flops
        total_count = fc1_count + fc2_count
        fc1_per_operator = Counter({self.lin_op: fc1_count})

        self.assertEqual(analyzer.total("fc1"), fc1_count)
        self.assertEqual(analyzer.total("fc2"), fc2_count)
        self.assertEqual(analyzer.total(""), total_count)
        self.assertEqual(analyzer.by_operator("fc1"), fc1_per_operator)

        # Tests no uncalled mods
        self.assertEqual(analyzer.uncalled_modules(), set())

    def test_non_forward_func_call(self) -> None:
        """
        Tests calls to a submodule's non-forward function.
        Also tests that the intermediate module is correctly identified as a skipped module.
        """

        model = NonForwardNet()
        inputs = (torch.randn((1, 10)),)
        analyzer = FlopCountAnalysis(model=model, inputs=inputs).ancestor_mode("caller")

        inner_fc_count = model.submod.fc_flops
        total_count = model.fc_flops + inner_fc_count

        self.assertEqual(analyzer.total("submod"), 0)
        self.assertEqual(analyzer.total("submod.fc"), inner_fc_count)
        self.assertEqual(analyzer.total(""), total_count)

        # The mod not directly called is registered as such
        self.assertEqual(analyzer.uncalled_modules(), {"submod"})

        analyzer = FlopCountAnalysis(model=model, inputs=inputs).ancestor_mode("owner")
        self.assertEqual(analyzer.total("submod"), inner_fc_count)
        self.assertEqual(analyzer.total("submod.fc"), inner_fc_count)
        self.assertEqual(analyzer.total(""), total_count)
        self.assertEqual(analyzer.uncalled_modules(), set())

    def test_shared_module(self) -> None:
        """
        Tests the behavior of shared submodules that may have multiple
        names.
        """

        model = SharedModuleNet()
        inputs = (torch.randn((1, *model.input_size)),)

        analyzer = (
            FlopCountAnalysis(model=model, inputs=inputs)
            .unsupported_ops_warnings(enabled=False)
            .ancestor_mode("caller")
        )

        # The names `submod2.submod` and `multiname2` are not included,
        # since only the first name of a module is made the canonical one.
        # The counts associated with these cases are included under
        # `submod1.submod` and `multiname1` respectively.
        multiname_flops = 2 * model.multiname_flops  # Called under 2 names
        shared_flops = 2 * model.shared_flops  # Shared under 2 submodules
        total_flops = multiname_flops + shared_flops
        flops = {
            "": total_flops,
            "submod1": model.shared_flops,
            "submod1.submod": shared_flops,
            "submod2": model.shared_flops,
            "multiname1": multiname_flops,
        }

        self.assertEqual(analyzer.by_module(), flops)

        # Test access by alternative name
        self.assertEqual(
            analyzer.total("submod2.submod"),
            flops["submod1.submod"],
        )
        self.assertEqual(
            analyzer.total("multiname2"),
            flops["multiname1"],
        )

        # Test getting canonical name
        self.assertEqual(analyzer.canonical_module_name("multiname2"), "multiname1")
        self.assertEqual(analyzer.canonical_module_name("multiname1"), "multiname1")
        self.assertEqual(analyzer.canonical_module_name("submod2.submod"), "submod1.submod")
        self.assertEqual(analyzer.canonical_module_name("submod1.submod"), "submod1.submod")

        # Tests no uncalled modules
        self.assertEqual(analyzer.uncalled_modules(), set())

    def test_recursive_scope(self) -> None:
        """
        Tests that an op is only counted once per module, even if it is
        in the scope of that module multiple times.
        """
        model = RecursiveScopeNet()
        inputs = (torch.randn((1, *model.input_size)),)

        analyzer = FlopCountAnalysis(model, inputs)

        self.assertEqual(analyzer.total(), model.flops)
        self.assertEqual(analyzer.total("fc"), model.flops)

        # Tests no uncalled modules
        self.assertEqual(analyzer.uncalled_modules(), set())

    def test_data_parallel(self) -> None:
        """
        Tests that a model wrapped in DataParallel still returns results
        labeled by the correct scopes.
        """
        model = NestedNet(lin_op=self.lin_op)
        inputs = (torch.randn((1, *model.input_size)),)

        # Find flops for wrapper
        flops = {
            "module" + ("." if name else "") + name: flop for name, flop in model.flops.items()
        }
        flops[""] = model.flops[""]
        name_to_module = {
            "module" + ("." if name else "") + name: mod
            for name, mod in model.name_to_module.items()
        }
        name_to_module[""] = model.name_to_module[""]

        model = torch.nn.DataParallel(model).cpu()
        analyzer = FlopCountAnalysis(model=model, inputs=inputs)
        analyzer.unsupported_ops_warnings(enabled=False)

        # Using a string input
        for name in flops:
            with self.subTest(name=name):
                gt_flops = sum(flops[name].values())
                self.assertEqual(analyzer.total(name), gt_flops)

        # Output as dictionary
        self.assertEqual(analyzer.by_module_and_operator(), flops)

        # Test no uncalled modules
        self.assertEqual(analyzer.uncalled_modules(), set())

    def test_data_parallel_root_scope(self) -> None:
        # A test case discussed in D32227000
        model = nn.DataParallel(nn.Linear(10, 10)).cpu()
        for mode in ["caller", "owner"]:
            flop = FlopCountAnalysis(model, (torch.randn(10, 10),))
            flop.ancestor_mode(mode)
            self.assertEqual(flop.total(), 1000)

    def test_unsupported_ops(self) -> None:
        """
        Tests per-module recording of unsupported operations.
        """

        model = NestedNet(lin_op=self.lin_op)
        inputs = (torch.randn((1, *model.input_size)),)

        analyzer = JitModelAnalysis(model=model, inputs=inputs).set_op_handle(
            "aten::addmm",
            addmm_flop_jit,
            "aten::linear",
            linear_flop_jit,
        )
        analyzer.total()

        skipped_inner_conv = Counter({"aten::_convolution": 1})
        skipped_inner_fc = Counter()
        skipped_inner = Counter({"aten::add": 1, "aten::mul": 1})
        skipped_inner += skipped_inner_fc
        skipped_inner += skipped_inner_conv

        skipped_outer_conv = Counter({"aten::_convolution": 1})
        skipped_outer_fc = Counter()
        skipped_outer = Counter({"aten::pow": 1})
        skipped_outer += skipped_outer_conv
        skipped_outer += skipped_outer_fc
        skipped_outer += skipped_inner

        skipped = {
            "": skipped_outer,
            "conv": skipped_outer_conv,
            "fc": skipped_outer_fc,
            "submod": skipped_inner,
            "submod.conv": skipped_inner_conv,
            "submod.fc": skipped_inner_fc,
        }

        # Access by string
        for name in skipped:
            with self.subTest(name=name):
                self.assertEqual(analyzer.unsupported_ops(name), skipped[name])

    def test_changing_handles(self) -> None:
        """
        Tests .set_op_handle(), .clear_op_handles()
        """
        model = NestedNet(lin_op=self.lin_op)
        inputs = (torch.randn((1, *model.input_size)),)
        op_handles: "Dict[str, Handle]" = {
            "aten::addmm": addmm_flop_jit,
            "aten::linear": linear_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs).set_op_handle(**op_handles)
        analyzer.unsupported_ops_warnings(enabled=False)

        # Request a result once to cache flop counts
        _ = analyzer.total("")

        # Add an op handle
        analyzer.set_op_handle("aten::_convolution", conv_flop_jit)

        self.assertEqual(analyzer.by_module_and_operator(), model.flops)

        # Overwrite an op handle
        def make_dummy_op(name: str, output: int) -> Handle:
            def dummy_ops_handle(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
                return Counter({name: output})

            return dummy_ops_handle

        dummy_name = "dummy_op"
        dummy_out = 1000
        analyzer.set_op_handle("aten::{}".format(self.lin_op), make_dummy_op(dummy_name, dummy_out))

        dummy_flops = {}
        for name, counts in model.flops.items():
            dummy_flops[name] = Counter(
                {op: flop for op, flop in counts.items() if op != self.lin_op}
            )
        dummy_flops[""][dummy_name] = 2 * dummy_out
        dummy_flops["fc"][dummy_name] = dummy_out
        dummy_flops["submod"][dummy_name] = dummy_out
        dummy_flops["submod.fc"][dummy_name] = dummy_out

        self.assertEqual(analyzer.by_module_and_operator(), dummy_flops)

        # Clear ops handles
        analyzer.clear_op_handles()

        empty_flops = {name: Counter() for name in model.flops}

        self.assertEqual(analyzer.by_module_and_operator(), empty_flops)

    def test_copy(self) -> None:
        """
        Tests .copy(...)
        """

        model = RepeatedNet()
        inputs = (torch.randn((1, *model.input_size)),)

        analyzer = (
            JitModelAnalysis(model=model, inputs=inputs)
            .set_op_handle(
                "aten::addmm",
                addmm_flop_jit,
                "aten::linear",
                linear_flop_jit,
            )
            .unsupported_ops_warnings(enabled=False)
            .tracer_warnings(mode="none")
        )

        repeated_net_flops = model.fc1_num * model.fc1_flops
        repeated_net_flops += model.fc2_num * model.fc2_flops

        analyzer_copy = analyzer.copy()

        # Outputs are the same
        self.assertEqual(
            analyzer.by_module_and_operator(),
            analyzer_copy.by_module_and_operator(),
        )

        # Settings match
        self.assertEqual(
            analyzer._enable_warn_unsupported_ops,
            analyzer_copy._enable_warn_unsupported_ops,
        )
        self.assertEqual(
            analyzer._enable_warn_uncalled_mods,
            analyzer_copy._enable_warn_uncalled_mods,
        )
        self.assertEqual(analyzer._warn_trace, analyzer_copy._warn_trace)

        # Changing copy does not change original
        analyzer_copy.unsupported_ops_warnings(enabled=True)
        self.assertNotEqual(
            analyzer._enable_warn_unsupported_ops,
            analyzer_copy._enable_warn_unsupported_ops,
        )

        # Copy with new model and inputs
        new_model = NonForwardNet()
        bs = 5
        new_inputs = (torch.randn((bs, *new_model.input_size)),)
        analyzer_new = analyzer.copy(new_model=new_model, new_inputs=new_inputs)

        non_forward_flops = new_model.fc_flops + new_model.submod.fc_flops

        # Total is correct for new model and inputs
        self.assertEqual(analyzer_new.total(), non_forward_flops * bs)

        # Original is unaffected
        self.assertEqual(analyzer.total(), repeated_net_flops)

        # Settings match
        self.assertEqual(
            analyzer._enable_warn_unsupported_ops,
            analyzer_new._enable_warn_unsupported_ops,
        )
        self.assertEqual(analyzer._warn_trace, analyzer_new._warn_trace)

    def test_disable_warnings(self) -> None:
        """
        Tests .unsupported_ops_warnings(...) and .tracer_warnings(...)
        """
        model = TraceWarningNet()
        inputs = (torch.randn((1, *model.input_size)),)
        analyzer = FlopCountAnalysis(model=model, inputs=inputs)

        # Tracer warnings
        analyzer.tracer_warnings(mode="all")
        analyzer._stats = None  # Manually clear cache so trace is rerun
        self.assertWarns(torch.jit._trace.TracerWarning, analyzer.total)
        analyzer._stats = None  # Manually clear cache so trace is rerun
        self.assertWarns(RuntimeWarning, analyzer.total)

        analyzer.tracer_warnings(mode="none")
        analyzer._stats = None  # Manually clear cache so trace is rerun
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = analyzer.total()
            if w:
                warning_types = [s.category for s in w]
                self.assertFalse(torch.jit._trace.TracerWarning in warning_types)
                self.assertFalse(RuntimeWarning in warning_types)

        analyzer.tracer_warnings(mode="no_tracer_warning")
        analyzer._stats = None  # Manually clear cache so trace is rerun
        self.assertWarns(RuntimeWarning, analyzer.total)
        analyzer._stats = None  # Manually clear cache so trace is rerun
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = analyzer.total()
            if w:
                warning_types = [s.category for s in w]
                self.assertFalse(torch.jit._trace.TracerWarning in warning_types)

        # Unsupported ops and uncalled modules warnings

        logger = logging.getLogger()
        skipeed_msg = "Unsupported operator aten::add encountered 1 time(s)"
        uncalled_msg = "never called"
        uncalled_modules = "fc1"  # fc2 is called by chance

        analyzer.uncalled_modules_warnings(enabled=False)
        analyzer.unsupported_ops_warnings(enabled=False)
        analyzer._stats = None  # Manually clear cache so trace is rerun
        with self.assertLogs(logger, logging.WARN) as cm:
            logger.warning("Dummy warning.")
            _ = analyzer.total()
        self.assertFalse(any(skipeed_msg in s for s in cm.output))
        self.assertFalse(any(uncalled_msg in s for s in cm.output))

        analyzer.unsupported_ops_warnings(enabled=True)
        analyzer.uncalled_modules_warnings(enabled=True)
        analyzer._stats = None  # Manually clear cache so trace is rerun
        with self.assertLogs(logger, logging.WARN) as cm:
            _ = analyzer.total()
        self.assertTrue(any(skipeed_msg in s for s in cm.output))
        self.assertTrue(any(uncalled_msg in s for s in cm.output))
        self.assertTrue(any(uncalled_modules in s for s in cm.output))

    def test_skip_uncalled_containers_warnings(self) -> None:
        # uncalled containers should not warn

        class A(nn.Module):
            def forward(self, x):
                return self.submod[0](x) + 1

        mod = A()
        mod.submod = nn.ModuleList([nn.Linear(3, 3)])  # pyre-ignore
        analyzer = FlopCountAnalysis(model=mod, inputs=torch.rand(1, 3))
        analyzer.unsupported_ops_warnings(enabled=False)

        logger = logging.getLogger()
        with self.assertLogs(logger, logging.WARN) as cm:
            logger.warning("Dummy warning.")
            _ = analyzer.total()
        uncalled_string = "Module never called: submod"
        self.assertFalse(any(uncalled_string in s for s in cm.output))
