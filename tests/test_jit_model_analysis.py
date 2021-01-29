import logging
import typing
import unittest
import warnings
from collections import Counter
from io import StringIO
from typing import Any, Dict, List

import torch
import torch.nn as nn
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

        self.fc_flops = fc_in * fc_out
        spatial_pos = (conv_input_size[1] + 2 * padding) - 2 * (kernel_size // 2)
        self.conv_flops = spatial_pos * kernel_size * conv_in * conv_out

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

        self.fc_flops = fc_in * fc_out
        spatial_pos = (self.input_size[1] + 2 * padding) - 2 * (kernel_size // 2)
        self.conv_flops = spatial_pos * kernel_size * conv_in * conv_out

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
        for i in range(self.fc1_num):
            x = self.fc1(x)
        for i in range(self.fc2_num):
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
        # multiname2 will not appear in module.named_children()
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
    Will raise a warning on trace due to python comparison of tensor data.
    Also has an aten::add op that will be skipped and raise a warning.
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
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
            "aten::_convolution": conv_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)
        analyzer.skipped_ops_warnings(enabled=False)

        submod_flops = model.submod.fc_flops + model.submod.conv_flops
        model_flops = model.conv_flops + model.fc_flops + submod_flops
        flops = {
            "": model_flops,
            "fc": model.fc_flops,
            "conv": model.conv_flops,
            "submod": submod_flops,
            "submod.fc": model.submod.fc_flops,
            "submod.conv": model.submod.conv_flops,
        }

        name_to_module = {
            "": model,
            "fc": model.fc,
            "conv": model.conv,
            "submod": model.submod,
            "submod.fc": model.submod.fc,
            "submod.conv": model.submod.conv,
        }

        # Using a string input
        for name in flops:
            with self.subTest(name=name):
                self.assertEqual(
                    analyzer.total(name),
                    flops[name],
                    ".total(...) failed count test for input string.",
                )

        # Using a module input
        for name in flops:
            with self.subTest(name=name):
                mod = name_to_module[name]
                self.assertEqual(
                    analyzer.total(mod),
                    flops[name],
                    ".total(...) failed count test for input module.",
                )

    def test_by_module(self) -> None:
        """
        Tests that JitModelAnalysis.by_module() returns the correct
        counts in the correctly structured dictionary.
        """

        model = NestedNet()
        inputs = (torch.randn((1, *model.input_size)),)
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
            "aten::_convolution": conv_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)
        analyzer.skipped_ops_warnings(enabled=False)

        submod_flops = model.submod.fc_flops + model.submod.conv_flops
        model_flops = model.conv_flops + model.fc_flops + submod_flops
        flops = {
            "": model_flops,
            "fc": model.fc_flops,
            "conv": model.conv_flops,
            "submod": submod_flops,
            "submod.fc": model.submod.fc_flops,
            "submod.conv": model.submod.conv_flops,
        }

        self.assertEqual(analyzer.by_module(), flops, ".by_module() failed count test.")

    def test_by_operator(self) -> None:
        """
        Tests that JitModelAnalysis.by_operator(module) returns the correct
        counts for string and module inputs.
        """

        model = NestedNet()
        inputs = (torch.randn((1, *model.input_size)),)
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
            "aten::_convolution": conv_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)
        analyzer.skipped_ops_warnings(enabled=False)

        submod_conv_flops = Counter({"conv": model.submod.conv_flops})
        submod_fc_flops = Counter({"addmm": model.submod.fc_flops})
        submod_flops = submod_conv_flops + submod_fc_flops
        conv_flops = Counter({"conv": model.conv_flops})
        fc_flops = Counter({"addmm": model.fc_flops})
        model_flops = submod_flops + conv_flops + fc_flops
        flops = {
            "": model_flops,
            "fc": fc_flops,
            "conv": conv_flops,
            "submod": submod_flops,
            "submod.fc": submod_fc_flops,
            "submod.conv": submod_conv_flops,
        }

        name_to_module = {
            "": model,
            "fc": model.fc,
            "conv": model.conv,
            "submod": model.submod,
            "submod.fc": model.submod.fc,
            "submod.conv": model.submod.conv,
        }

        # Using a string input
        for name in flops:
            with self.subTest(name=name):
                self.assertEqual(
                    analyzer.by_operator(name),
                    flops[name],
                    ".total(...) failed count test for input string.",
                )

        # Using a module input
        for name in flops:
            with self.subTest(name=name):
                mod = name_to_module[name]
                self.assertEqual(
                    analyzer.by_operator(mod),
                    flops[name],
                    ".total(...) failed count test for input module.",
                )

    def test_by_module_and_operator(self) -> None:
        """
        Tests that JitModelAnalysis.by_module_and_operator() returns
        the correct counts in the correct structure.
        """

        model = NestedNet()
        inputs = (torch.randn((1, *model.input_size)),)
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
            "aten::_convolution": conv_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)
        analyzer.skipped_ops_warnings(enabled=False)

        submod_conv_flops = Counter({"conv": model.submod.conv_flops})
        submod_fc_flops = Counter({"addmm": model.submod.fc_flops})
        submod_flops = submod_conv_flops + submod_fc_flops
        conv_flops = Counter({"conv": model.conv_flops})
        fc_flops = Counter({"addmm": model.fc_flops})
        model_flops = submod_flops + conv_flops + fc_flops
        flops = {
            "": model_flops,
            "fc": fc_flops,
            "conv": conv_flops,
            "submod": submod_flops,
            "submod.fc": submod_fc_flops,
            "submod.conv": submod_conv_flops,
        }

        self.assertEqual(
            analyzer.by_module_and_operator(), flops, ".by_module() failed count test."
        )

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

        self.assertEqual(
            analyzer.total("unused"),
            unused_count,
            "Unused module failed to result in .total(unused)=0.",
        )

        self.assertEqual(
            analyzer.by_operator("unused"),
            unused_per_operator,
            "Unused module failed to result in .by_operator(unused)=Counter().",
        )

        self.assertEqual(
            analyzer.total(""),
            model_count,
            "Unused module caused parent to return incorrect total count.",
        )

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

        self.assertEqual(
            analyzer.total("fc1"),
            fc1_count,
            ".total() failed to aggregate counts over repeated module calls.",
        )

        self.assertEqual(
            analyzer.total("fc2"),
            fc2_count,
            ".total() failed to aggregate counts over repeated module calls.",
        )

        self.assertEqual(
            analyzer.total(""),
            total_count,
            "Repeated submodule calls caused incorrect count in parent.",
        )

        self.assertEqual(
            analyzer.by_operator("fc1"),
            fc1_per_operator,
            ".by_operator() failed to aggregate counts over repeated module calls.",
        )

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

        self.assertEqual(
            analyzer.total("submod"),
            submod_count,
            ".total('submod') fails to give 0 count.",
        )

        self.assertEqual(
            analyzer.total("submod.fc"),
            inner_fc_count,
            ".total('submod.fc') fails to give the correct count.",
        )

        self.assertEqual(
            analyzer.total(""),
            total_count,
            ".total('') fails to give the correct count.",
        )

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

        self.assertEqual(
            analyzer.by_module(),
            flops,
            ".by_module() failed to give the expected names for shared modules.",
        )

        # Test access by alternative name
        self.assertEqual(
            analyzer.total("submod2.submod"),
            flops["submod1.submod"],
            ".total(...) fails to return correct result for alternative name "
            "of shared module.",
        )

        self.assertEqual(
            analyzer.total(model.submod2.submod),
            flops["submod1.submod"],
            ".total(...) fails to return correct result for alternative name "
            "of shared module.",
        )

        # multiname2 is not a valid name since it is never encountered
        # in .named_children() or .named_modules(), so the analysis
        # doesn't know anything about it. Access using the model's
        # attribute should still work though:
        self.assertEqual(
            analyzer.total(model.multiname2),
            flops["multiname1"],
            ".total(...) fails to return correct result for alternative name "
            "of submodule with multiple names.",
        )

    def test_data_parallel(self) -> None:
        """
        Tests that a model wrapped in DataParallel still returns results
        labelled by the correct scopes.
        """
        model = NestedNet()
        inputs = (torch.randn((1, *model.input_size)),)
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
            "aten::_convolution": conv_flop_jit,
        }

        submod_flops = model.submod.fc_flops + model.submod.conv_flops
        model_flops = model.conv_flops + model.fc_flops + submod_flops
        flops = {
            "": model_flops,
            "module": model_flops,
            "module.fc": model.fc_flops,
            "module.conv": model.conv_flops,
            "module.submod": submod_flops,
            "module.submod.fc": model.submod.fc_flops,
            "module.submod.conv": model.submod.conv_flops,
        }

        model = torch.nn.DataParallel(model)
        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)
        analyzer.skipped_ops_warnings(enabled=False)

        name_to_module = {
            "": model,
            "module": model.module,
            "module.fc": model.module.fc,
            "module.conv": model.module.conv,
            "module.submod": model.module.submod,
            "module.submod.fc": model.module.submod.fc,
            "module.submod.conv": model.module.submod.conv,
        }

        # Using a string input
        for name in flops:
            with self.subTest(name=name):
                self.assertEqual(
                    analyzer.total(name),
                    flops[name],
                    ".total(...) failed for DataParallel model for input string.",
                )

        # Using a module input
        for name in flops:
            with self.subTest(name=name):
                mod = name_to_module[name]
                self.assertEqual(
                    analyzer.total(mod),
                    flops[name],
                    ".total(...) failed for DataParallel model for input module.",
                )

        # Output as dictionary
        self.assertEqual(
            analyzer.by_module(), flops, ".by_module() failed for DataParallel model."
        )

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

        name_to_module = {
            "": model,
            "fc": model.fc,
            "conv": model.conv,
            "submod": model.submod,
            "submod.fc": model.submod.fc,
            "submod.conv": model.submod.conv,
        }

        # Access by string
        for name in skipped:
            with self.subTest(name=name):
                self.assertEqual(
                    analyzer.skipped_ops(name),
                    skipped[name],
                    ".skipped_ops(...) failed count test for input module.",
                )

        # Access by module
        for name in skipped:
            with self.subTest(name=name):
                mod = name_to_module[name]
                self.assertEqual(
                    analyzer.skipped_ops(mod),
                    skipped[name],
                    ".skipped_ops(...) failed count test for input module.",
                )

    def test_changing_handles(self) -> None:
        """
        Tests .set_ops_handle(), .clear_ops_handles(), and modifying
        .ops_handles directly.
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

        submod_conv_flops = Counter({"conv": model.submod.conv_flops})
        submod_fc_flops = Counter({"addmm": model.submod.fc_flops})
        submod_flops = submod_conv_flops + submod_fc_flops
        conv_flops = Counter({"conv": model.conv_flops})
        fc_flops = Counter({"addmm": model.fc_flops})
        model_flops = submod_flops + conv_flops + fc_flops
        flops = {
            "": model_flops,
            "fc": fc_flops,
            "conv": conv_flops,
            "submod": submod_flops,
            "submod.fc": submod_fc_flops,
            "submod.conv": submod_conv_flops,
        }

        self.assertEqual(
            analyzer.by_module_and_operator(),
            flops,
            ".by_module_and_operator() failed count test when a handle was added.",
        )

        # Overwrite an op handle
        def make_dummy_op(output: int) -> Handle:
            def dummy_ops_handle(
                inputs: List[Any], outputs: List[Any]
            ) -> typing.Counter:
                return Counter({"dummy_op": output})

            return dummy_ops_handle

        dummy_out = 1000
        analyzer.set_ops_handle("aten::addmm", make_dummy_op(dummy_out))

        submod_conv_flops = Counter({"conv": model.submod.conv_flops})
        submod_fc_flops = Counter({"dummy_op": dummy_out})
        submod_flops = submod_conv_flops + submod_fc_flops
        conv_flops = Counter({"conv": model.conv_flops})
        fc_flops = Counter({"dummy_op": dummy_out})
        model_flops = submod_flops + conv_flops + fc_flops
        dummy_flops = {
            "": Counter({"dummy_op": 2000, "conv": 260}),
            "fc": Counter({"dummy_op": 1000}),
            "conv": Counter({"conv": 240}),
            "submod": Counter({"dummy_op": 1000, "conv": 20}),
            "submod.fc": Counter({"dummy_op": 1000}),
            "submod.conv": Counter({"conv": 20}),
        }

        self.assertEqual(
            analyzer.by_module_and_operator(),
            dummy_flops,
            ".by_module_and_operator() failed count test when a handle"
            " was overwritten.",
        )

        # Clear ops handles
        analyzer.clear_ops_handles()

        empty_flops = {
            "": Counter(),
            "fc": Counter(),
            "conv": Counter(),
            "submod": Counter(),
            "submod.fc": Counter(),
            "submod.conv": Counter(),
        }

        self.assertEqual(
            analyzer.by_module_and_operator(),
            empty_flops,
            ".by_module_and_operator() failed count test when handles"
            " were cleared with .clear_ops_handles().",
        )

        # Directly write to JitModelAnalysis ops_handles property
        analyzer.ops_handles = {
            "aten::addmm": addmm_flop_jit,
            "aten::_convolution": conv_flop_jit,
        }

        self.assertEqual(
            analyzer.by_module_and_operator(),
            flops,
            ".by_module_and_operator() failed count test when handles"
            " were set directly.",
        )

    def test_changing_model_and_inputs(self) -> None:
        """
        Tests direct changes to the .model and .inputs properties of
        JitModelAnalysis.
        """
        model = RepeatedNet()
        inputs = (torch.randn((1, *model.input_size)),)
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)

        repeated_net_flops = model.fc1_num * model.fc1_flops
        repeated_net_flops += model.fc2_num * model.fc2_flops

        # Request once to cache results
        _ = analyzer.total()

        # Change model
        new_model = NonForwardNet()
        self.assertEqual(
            new_model.input_size,
            model.input_size,
            "Test failed to be consistent; to test model change valid "
            "input sizes need to be the same for the two models.",
        )

        analyzer.model = new_model

        non_forward_flops = new_model.fc_flops + new_model.submod.fc_flops

        self.assertEqual(
            analyzer.total(),
            non_forward_flops,
            ".total() returned incorrect count after model was changed.",
        )

        # Change the inputs
        bs = 5
        analyzer.inputs = (torch.randn((bs, *new_model.input_size)),)

        self.assertEqual(
            analyzer.total(),
            non_forward_flops * bs,
            ".total() returned incorrect count after inputs were changed.",
        )

    def test_output_scale(self) -> None:
        """
        Tests .set_output_scale(...) and .get_output_scale().
        """

        model = NestedNet()
        inputs = (torch.randn((1, *model.input_size)),)
        ops_handles = {
            "aten::addmm": addmm_flop_jit,
            "aten::_convolution": conv_flop_jit,
        }

        analyzer = JitModelAnalysis(model=model, inputs=inputs, ops_handles=ops_handles)
        analyzer.skipped_ops_warnings(enabled=False)
        analyzer.set_output_scale(scale="kilo")

        def rescale(counter: Dict[str, int], x: int) -> Dict[str, float]:
            out_dict = {k: v / x for k, v in counter.items()}
            if isinstance(counter, Counter):
                out_dict = Counter(out_dict)
            return out_dict

        submod_conv_flops = Counter({"conv": model.submod.conv_flops})
        submod_fc_flops = Counter({"addmm": model.submod.fc_flops})
        submod_flops = submod_conv_flops + submod_fc_flops
        conv_flops = Counter({"conv": model.conv_flops})
        fc_flops = Counter({"addmm": model.fc_flops})
        model_flops = submod_flops + conv_flops + fc_flops
        flops = {
            "": rescale(model_flops, 1e3),
            "fc": rescale(fc_flops, 1e3),
            "conv": rescale(conv_flops, 1e3),
            "submod": rescale(submod_flops, 1e3),
            "submod.fc": rescale(submod_fc_flops, 1e3),
            "submod.conv": rescale(submod_conv_flops, 1e3),
        }

        # Test rescaling in .by_module_and_operator()
        self.assertEqual(
            analyzer.by_module_and_operator(),
            flops,
            "Incorrect results for fractional counter values when outputting "
            "at non-unity scale in .by_module_and_operator().",
        )

        # Test rescaling in .by_operator()
        self.assertEqual(
            analyzer.by_operator(""),
            flops[""],
            "Incorrect results for fractional counter values when outputting "
            "at non-unity scale in .by_operator().",
        )

        submod_flops = model.submod.fc_flops + model.submod.conv_flops
        model_flops = model.conv_flops + model.fc_flops + submod_flops
        flops = {
            "": model_flops,
            "fc": model.fc_flops,
            "conv": model.conv_flops,
            "submod": submod_flops,
            "submod.fc": model.submod.fc_flops,
            "submod.conv": model.submod.conv_flops,
        }
        flops = rescale(flops, 1e3)

        # Test rescaling in .by_module()
        self.assertEqual(
            analyzer.by_module(),
            flops,
            "Incorrect results for fractional counter values when outputting "
            "at non-unity scale in .by_module().",
        )

        # Test rescaling in .total()
        self.assertEqual(
            analyzer.total(""),
            flops[""],
            "Incorrect results for fractional counter values when outputting "
            "at non-unity scale in .by_module().",
        )

        # Test named scales
        named_scales = {
            "unity": 1,
            "kilo": 1e3,
            "mega": 1e6,
            "giga": 1e9,
            "tera": 1e12,
            "peta": 1e15,
        }

        for name, scale in named_scales.items():
            with self.subTest(name=name):
                analyzer.set_output_scale(name)
                self.assertEqual(
                    analyzer.total(),
                    model_flops / scale,
                    ".total() returned incorrect count for scale {}".format(name),
                )
                self.assertEqual(
                    analyzer.get_output_scale(),
                    scale,
                    ".get_output_scale() returned the wrong scale.",
                )

        # Test explicit numeric scale
        test_scale = 2500
        analyzer.set_output_scale(test_scale)
        self.assertEqual(
            analyzer.total(),
            model_flops / test_scale,
            ".total() returned incorrect results for numerical scale.",
        )
        self.assertEqual(
            analyzer.get_output_scale(),
            test_scale,
            ".get_output_scale() returned the wrong scale.",
        )

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
        analyzer.set_output_scale("giga")
        analyzer.skipped_ops_warnings(enabled=False)
        analyzer.tracer_warnings(enabled=False)

        repeated_net_flops = model.fc1_num * model.fc1_flops
        repeated_net_flops += model.fc2_num * model.fc2_flops

        analyzer_copy = analyzer.copy()

        # Outputs are the same
        self.assertEqual(
            analyzer.by_module_and_operator(),
            analyzer_copy.by_module_and_operator(),
            ".copy() gives JitModelAnalysis with different results than.",
        )

        # Settings match
        self.assertEqual(
            analyzer.warn_skipped,
            analyzer_copy.warn_skipped,
            ".copy() gives JitModelAnalysis with different warning settings.",
        )

        self.assertEqual(
            analyzer.warn_trace,
            analyzer_copy.warn_trace,
            ".copy() gives JitModelAnalysis with different warning settings.",
        )

        # Changing copy does not change original
        analyzer_copy.skipped_ops_warnings(enabled=True)
        self.assertNotEqual(
            analyzer.warn_skipped,
            analyzer_copy.warn_skipped,
            "Changing setting from .copy() changed original too.",
        )

        # Copy with new model and inputs
        new_model = NonForwardNet()
        bs = 5
        new_inputs = (torch.randn((bs, *new_model.input_size)),)
        analyzer_new = analyzer.copy(new_model=new_model, new_inputs=new_inputs)

        non_forward_flops = new_model.fc_flops + new_model.submod.fc_flops

        # Total is correct for new model and inputs
        self.assertAlmostEqual(
            analyzer_new.total(),
            non_forward_flops * bs / 1e9,
            msg=".copy() with new model/inputs gives incorrect results.",
        )

        # Original is unaffected
        self.assertAlmostEqual(
            analyzer.total(),
            repeated_net_flops / 1e9,
            msg=".copy() with new model/inputs changed original.",
        )

        # Settings match
        self.assertEqual(
            analyzer.warn_skipped,
            analyzer_new.warn_skipped,
            ".copy() gives JitModelAnalysis with different warning settings.",
        )

        self.assertEqual(
            analyzer.warn_trace,
            analyzer_new.warn_trace,
            ".copy() gives JitModelAnalysis with different warning settings.",
        )

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
        analyzer.tracer_warnings(enabled=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = analyzer.total()
            warning_types = [s.category for s in w]
            self.assertFalse(
                torch.jit._trace.TracerWarning in warning_types,
                "TracerWarning was not correctly suppressed.",
            )

        analyzer.tracer_warnings(enabled=True)
        analyzer.counts = None  # Manually clear cache so trace is rerun

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = analyzer.total()
            warning_types = [s.category for s in w]
            self.assertTrue(
                torch.jit._trace.TracerWarning in warning_types,
                "TracerWarning was incorrectly suppressed.",
            )

        # Skipped ops warnings
        analyzer.skipped_ops_warnings(enabled=False)

        logger = logging.getLogger()
        log_stream = StringIO()
        handler = logging.StreamHandler(stream=log_stream)
        handler.setLevel(logging.WARNING)
        logger.addHandler(handler)
        skipped_string = "Skipped operation aten::add 1 time(s)"

        _ = analyzer.total()
        self.assertFalse(
            skipped_string in log_stream.getvalue(),
            "Skipped op warning was not correctly suppressed.",
        )

        analyzer.skipped_ops_warnings(enabled=True)

        _ = analyzer.total()
        self.assertTrue(
            skipped_string in log_stream.getvalue(),
            "Skipped op warning was incorrectly suppressed.",
        )

        logger.removeHandler(handler)
