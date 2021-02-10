import logging
import typing
import unittest
import warnings
from collections import Counter
from typing import Any, List

import torch
import torch.nn as nn
from fvcore.nn.flop_count import FlopCount
from fvcore.nn.jit_analysis import JitModelAnalysis
from fvcore.nn.jit_handles import Handle, addmm_flop_jit, conv_flop_jit


class NestedNetInnerModule(nn.Module):
    """
    A submodule for the nested net test module below.
    """

    def __init__(self) -> None:
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
        fc_flops = Counter({"addmm": fc_flops})
        spatial_pos = (conv_input_size[1] + 2 * padding) - 2 * (kernel_size // 2)
        conv_flops = spatial_pos * kernel_size * conv_in * conv_out
        conv_flops = Counter({"conv": conv_flops})
        model_flops = conv_flops + fc_flops
        self.flops = {
            "": model_flops,
            "fc": fc_flops,
            "conv": conv_flops,
        }

        self.name_to_module = {
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

    def __init__(self) -> None:
        super().__init__()
        self.input_size = (4, 5)

        conv_in = 4
        conv_out = 4
        kernel_size = 3
        padding = 1
        fc_in = 20
        fc_out = 10

        self.submod = NestedNetInnerModule()
        self.fc = nn.Linear(in_features=fc_in, out_features=fc_out)
        self.conv = nn.Conv1d(
            in_channels=conv_in,
            out_channels=conv_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        fc_flops = fc_in * fc_out
        fc_flops = Counter({"addmm": fc_flops})
        spatial_pos = (self.input_size[1] + 2 * padding) - 2 * (kernel_size // 2)
        conv_flops = spatial_pos * kernel_size * conv_in * conv_out
        conv_flops = Counter({"conv": conv_flops})

        model_flops = conv_flops + fc_flops + self.submod.flops[""]
        self.flops = {
            "": model_flops,
            "fc": fc_flops,
            "conv": conv_flops,
            "submod": self.submod.flops[""],
            "submod.fc": self.submod.flops["fc"],
            "submod.conv": self.submod.flops["conv"],
        }

        self.name_to_module = {
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

        self.fc1_flops = fc1_in * fc1_out
        self.fc2_flops = fc2_in * fc2_out
        self.unused_flops = unused_in * unused_out  # If it were applied

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


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

        self.fc1_flops = fc1_in * fc1_out
        self.fc2_flops = fc2_in * fc2_out

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

        self.fc_flops = fc_in * fc_out

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

        self.fc_flops = fc_in * fc_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.submod.other_func(self.fc(x))


class SharingInnerModule(nn.Module):
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
        self.submod1 = SharingInnerModule(inner)
        self.submod2 = SharingInnerModule(inner)
        multiname = nn.Linear(in_features=fc2_in, out_features=fc2_out)
        self.multiname1 = multiname
        self.multiname2 = multiname

        self.multiname_flops = fc2_in * fc2_out
        self.shared_flops = fc1_in * fc1_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.submod1(x) + self.submod2(x)
        x = self.multiname1(x) + self.multiname2(x)
        return x


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

        self.fc1_flops = fc1_in, fc1_out
        self.fc2_flops = fc2_in, fc2_out

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

    def test_total(self) -> None:
        """
        Tests that JitModelAnalysis.total(module) returns the correct
        counts for string and module inputs.
        """

        model = NestedNet()
        inputs = (torch.randn((1, *model.input_size)),)

        analyzer = FlopCount(model=model, inputs=inputs)
        analyzer.skipped_ops_warnings(enabled=False)

        # Using a string input
        for name in model.flops:
            with self.subTest(name=name):
                gt_flops = sum(model.flops[name].values())
                self.assertEqual(analyzer.total(name), gt_flops)

        # Using a module input
        for name in model.flops:
            with self.subTest(name=name):
                mod = model.name_to_module[name]
                gt_flops = sum(model.flops[name].values())
                self.assertEqual(analyzer.total(mod), gt_flops)

    def test_by_module(self) -> None:
        """
        Tests that JitModelAnalysis.by_module() returns the correct
        counts in the correctly structured dictionary.
        """

        model = NestedNet()
        inputs = (torch.randn((1, *model.input_size)),)

        analyzer = FlopCount(model=model, inputs=inputs)
        analyzer.skipped_ops_warnings(enabled=False)

        flops = {name: sum(counts.values()) for name, counts in model.flops.items()}

        self.assertEqual(analyzer.by_module(), flops)

    def test_by_operator(self) -> None:
        """
        Tests that JitModelAnalysis.by_operator(module) returns the correct
        counts for string and module inputs.
        """

        model = NestedNet()
        inputs = (torch.randn((1, *model.input_size)),)

        analyzer = FlopCount(model=model, inputs=inputs)
        analyzer.skipped_ops_warnings(enabled=False)

        # Using a string input
        for name in model.flops:
            with self.subTest(name=name):
                self.assertEqual(analyzer.by_operator(name), model.flops[name])

        # Using a module input
        for name in model.flops:
            with self.subTest(name=name):
                mod = model.name_to_module[name]
                self.assertEqual(analyzer.by_operator(mod), model.flops[name])

    def test_by_module_and_operator(self) -> None:
        """
        Tests that JitModelAnalysis.by_module_and_operator() returns
        the correct counts in the correct structure.
        """

        model = NestedNet()
        inputs = (torch.randn((1, *model.input_size)),)

        analyzer = FlopCount(model=model, inputs=inputs)
        analyzer.skipped_ops_warnings(enabled=False)

        self.assertEqual(analyzer.by_module_and_operator(), model.flops)

    def test_unused_module(self) -> None:
        """
        Tests that unused modules return 0 count for operator sums and
        and empty Counter() for per-operator results.
        """

        model = UnusedNet()
        inputs = (torch.randn((1, *model.input_size)),)
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)

        unused_count = 0
        unused_per_operator = Counter()
        model_count = model.fc1_flops + model.fc2_flops

        self.assertEqual(analyzer.total("unused"), unused_count)
        self.assertEqual(analyzer.by_operator("unused"), unused_per_operator)
        self.assertEqual(analyzer.total(""), model_count)

    def test_repeated_module(self) -> None:
        """
        Tests that repeated calls to the same submodule correct aggregates
        results to that submodule.
        """

        model = RepeatedNet()
        inputs = (torch.randn((1, *model.input_size)),)
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)

        fc1_count = model.fc1_num * model.fc1_flops
        fc2_count = model.fc2_num * model.fc2_flops
        total_count = fc1_count + fc2_count
        fc1_per_operator = Counter({"addmm": fc1_count})

        self.assertEqual(analyzer.total("fc1"), fc1_count)
        self.assertEqual(analyzer.total("fc2"), fc2_count)
        self.assertEqual(analyzer.total(""), total_count)
        self.assertEqual(analyzer.by_operator("fc1"), fc1_per_operator)

    def test_non_forward_func_call(self) -> None:
        """
        Tests that calls to a submodule's non-forward function attribute
        resulting counts to the calling module.
        """

        model = NonForwardNet()
        inputs = (torch.randn((1, 10)),)
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)

        submod_count = 0
        inner_fc_count = model.submod.fc_flops
        total_count = model.fc_flops + inner_fc_count

        self.assertEqual(analyzer.total("submod"), submod_count)
        self.assertEqual(analyzer.total("submod.fc"), inner_fc_count)
        self.assertEqual(analyzer.total(""), total_count)

    def test_shared_module(self) -> None:
        """
        Tests the behavior of shared submodules that may have multiple
        names.
        """

        model = SharedModuleNet()
        inputs = (torch.randn((1, *model.input_size)),)
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)
        analyzer.skipped_ops_warnings(enabled=False)

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
            analyzer.total(model.submod2.submod),
            flops["submod1.submod"],
        )
        self.assertEqual(
            analyzer.total("multiname2"),
            flops["multiname1"],
        )
        self.assertEqual(
            analyzer.total(model.multiname2),
            flops["multiname1"],
        )

        # Test getting canonical name
        self.assertEqual(analyzer.canonical_module_name("multiname2"), "multiname1")
        self.assertEqual(analyzer.canonical_module_name("multiname1"), "multiname1")
        self.assertEqual(analyzer.canonical_module_name(model.multiname2), "multiname1")
        self.assertEqual(analyzer.canonical_module_name(model.multiname1), "multiname1")
        self.assertEqual(
            analyzer.canonical_module_name("submod2.submod"), "submod1.submod"
        )
        self.assertEqual(
            analyzer.canonical_module_name("submod1.submod"), "submod1.submod"
        )
        self.assertEqual(
            analyzer.canonical_module_name(model.submod2.submod), "submod1.submod"
        )
        self.assertEqual(
            analyzer.canonical_module_name(model.submod1.submod), "submod1.submod"
        )

    def test_data_parallel(self) -> None:
        """
        Tests that a model wrapped in DataParallel still returns results
        labelled by the correct scopes.
        """
        model = NestedNet()
        inputs = (torch.randn((1, *model.input_size)),)

        # Find flops for wrapper
        flops = {
            "module" + ("." if name else "") + name: flop
            for name, flop in model.flops.items()
        }
        flops[""] = model.flops[""]
        name_to_module = {
            "module" + ("." if name else "") + name: mod
            for name, mod in model.name_to_module.items()
        }
        name_to_module[""] = model.name_to_module[""]

        model = torch.nn.DataParallel(model)
        analyzer = FlopCount(model=model, inputs=inputs)
        analyzer.skipped_ops_warnings(enabled=False)

        # Using a string input
        for name in flops:
            with self.subTest(name=name):
                gt_flops = sum(flops[name].values())
                self.assertEqual(analyzer.total(name), gt_flops)

        # Using a module input
        for name in flops:
            with self.subTest(name=name):
                mod = name_to_module[name]
                gt_flops = sum(flops[name].values())
                self.assertEqual(analyzer.total(mod), gt_flops)

        # Output as dictionary
        self.assertEqual(analyzer.by_module_and_operator(), flops)

    def test_skipped_ops(self) -> None:
        """
        Tests per-module recording of skipped operations.
        """

        model = NestedNet()
        inputs = (torch.randn((1, *model.input_size)),)
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)

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
                self.assertEqual(analyzer.skipped_ops(name), skipped[name])

        # Access by module
        for name in skipped:
            with self.subTest(name=name):
                mod = model.name_to_module[name]
                self.assertEqual(analyzer.skipped_ops(mod), skipped[name])

    def test_changing_handles(self) -> None:
        """
        Tests .set_ops_handle(), .clear_ops_handles()
        """
        model = NestedNet()
        inputs = (torch.randn((1, *model.input_size)),)
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)
        analyzer.skipped_ops_warnings(enabled=False)

        # Request a result once to cache flop counts
        _ = analyzer.total("")

        # Add an op handle
        analyzer.set_ops_handle("aten::_convolution", conv_flop_jit)

        self.assertEqual(analyzer.by_module_and_operator(), model.flops)

        # Overwrite an op handle
        def make_dummy_op(name: str, output: int) -> Handle:
            def dummy_ops_handle(
                inputs: List[Any], outputs: List[Any]
            ) -> typing.Counter:
                return Counter({name: output})

            return dummy_ops_handle

        dummy_name = "dummy_op"
        dummy_out = 1000
        analyzer.set_ops_handle("aten::addmm", make_dummy_op(dummy_name, dummy_out))

        dummy_flops = {}
        for name, counts in model.flops.items():
            dummy_flops[name] = Counter(
                {op: flop for op, flop in counts.items() if op != "addmm"}
            )
        dummy_flops[""][dummy_name] = 2 * dummy_out
        dummy_flops["fc"][dummy_name] = dummy_out
        dummy_flops["submod"][dummy_name] = dummy_out
        dummy_flops["submod.fc"][dummy_name] = dummy_out

        self.assertEqual(analyzer.by_module_and_operator(), dummy_flops)

        # Clear ops handles
        analyzer.clear_ops_handles()

        empty_flops = {name: Counter() for name in model.flops}

        self.assertEqual(analyzer.by_module_and_operator(), empty_flops)

    def test_copy(self) -> None:
        """
        Tests .copy(...)
        """

        model = RepeatedNet()
        inputs = (torch.randn((1, *model.input_size)),)
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)
        analyzer.skipped_ops_warnings(enabled=False)
        analyzer.tracer_warnings(mode="none")

        repeated_net_flops = model.fc1_num * model.fc1_flops
        repeated_net_flops += model.fc2_num * model.fc2_flops

        analyzer_copy = analyzer.copy()

        # Outputs are the same
        self.assertEqual(
            analyzer.by_module_and_operator(),
            analyzer_copy.by_module_and_operator(),
        )

        # Settings match
        self.assertEqual(analyzer._warn_skipped, analyzer_copy._warn_skipped)
        self.assertEqual(analyzer._warn_trace, analyzer_copy._warn_trace)

        # Changing copy does not change original
        analyzer_copy.skipped_ops_warnings(enabled=True)
        self.assertNotEqual(analyzer._warn_skipped, analyzer_copy._warn_skipped)

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
        self.assertEqual(analyzer._warn_skipped, analyzer_new._warn_skipped)
        self.assertEqual(analyzer._warn_trace, analyzer_new._warn_trace)

    def test_disable_warnings(self) -> None:
        """
        Tests .skipped_ops_warnings(...) and .tracer_warnings(...)
        """
        model = TraceWarningNet()
        inputs = (torch.randn((1, *model.input_size)),)
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)

        # Tracer warnings
        analyzer.tracer_warnings(mode="all")
        analyzer._counts = None  # Manually clear cache so trace is rerun
        self.assertWarns(torch.jit._trace.TracerWarning, analyzer.total)
        analyzer._counts = None  # Manually clear cache so trace is rerun
        self.assertWarns(RuntimeWarning, analyzer.total)

        analyzer.tracer_warnings(mode="none")
        analyzer._counts = None  # Manually clear cache so trace is rerun
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = analyzer.total()
            warning_types = [s.category for s in w]
            self.assertFalse(torch.jit._trace.TracerWarning in warning_types)
            self.assertFalse(RuntimeWarning in warning_types)

        analyzer.tracer_warnings(mode="no_tracer_warning")
        analyzer._counts = None  # Manually clear cache so trace is rerun
        self.assertWarns(RuntimeWarning, analyzer.total)
        analyzer._counts = None  # Manually clear cache so trace is rerun
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = analyzer.total()
            warning_types = [s.category for s in w]
            self.assertFalse(torch.jit._trace.TracerWarning in warning_types)

        # Skipped ops warnings

        logger = logging.getLogger()
        skipped_string = "Skipped operation aten::add 1 time(s)"

        analyzer.skipped_ops_warnings(enabled=False)
        analyzer._counts = None  # Manually clear cache so trace is rerun
        with self.assertLogs(logger, logging.WARN) as cm:
            logger.warning("Dummy warning.")
            _ = analyzer.total()
        self.assertTrue(cm.output == ["WARNING:root:Dummy warning."])

        analyzer.skipped_ops_warnings(enabled=True)
        analyzer._counts = None  # Manually clear cache so trace is rerun
        with self.assertLogs(logger, logging.WARN) as cm:
            _ = analyzer.total()
        self.assertTrue(skipped_string in cm.output[0])
