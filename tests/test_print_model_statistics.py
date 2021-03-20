# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from typing import Dict

import torch
import torch.nn as nn
from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis
from fvcore.nn.print_model_statistics import (
    _fill_missing_statistics,
    _group_by_module,
    _indicate_uncalled_modules,
    _model_stats_str,
    _model_stats_table,
    _pretty_statistics,
    _remove_zero_statistics,
    flop_count_str,
    flop_count_table,
)


test_statistics = {
    "": {"stat1": 4000, "stat2": 600000, "stat3": 0},
    "a1": {"stat1": 20, "stat2": 3000},
    "a1.b1": {"stat1": 20, "stat2": 3000},
    "a1.b1.c1": {"stat1": 20, "stat2": 3000},
    "a1.b1.c1.d1": {"stat1": 0, "stat2": 0},
    "a1.b1.c1.d2": {"stat1": 100},
    "a2": {"stat1": 123456, "stat2": 654321},
    "a2.b1": {"stat1": 0, "stat2": 100, "stat3": 40},
    "a2.b1.c1": {"stat1": 200, "stat2": 300},
    "a2.b1.c2": {"stat1": 0},
}

string_statistics = {
    "": {"stat1": "4K", "stat2": "0.6M", "stat3": "0"},
    "a1": {"stat1": "20", "stat2": "3K"},
    "a1.b1": {"stat1": "20", "stat2": "3K"},
    "a1.b1.c1": {"stat1": "20", "stat2": "3K"},
    "a1.b1.c1.d1": {"stat1": "0", "stat2": "0"},
    "a1.b1.c1.d2": {"stat1": "100"},
    "a2": {"stat1": "0.123M", "stat2": "0.654M"},
    "a2.b1": {"stat1": "0", "stat2": "100", "stat3": "40"},
    "a2.b1.c1": {"stat1": "0.2K", "stat2": "0.3K"},
    "a2.b1.c2": {"stat1": "0"},
}


stat1 = {
    "": 4000,
    "a1": 20,
    "a1.b1": 20,
    "a1.b1.c1": 20,
    "a1.b1.c1.d1": 0,
    "a1.b1.c1.d2": 100,
    "a2": 123456,
    "a2.b1": 0,
    "a2.b1.c1": 200,
    "a2.b1.c2": 0,
}

stat2 = {
    "": 600000,
    "a1": 3000,
    "a1.b1": 3000,
    "a1.b1.c1": 3000,
    "a1.b1.c1.d1": 0,
    "a2": 654321,
    "a2.b1": 100,
    "a2.b1.c1": 300,
}

stat3 = {"": 0, "a2.b1": 40}

ungrouped_stats: Dict[str, Dict[str, int]] = {
    "stat1": stat1,
    "stat2": stat2,
    "stat3": stat3,
}


class A2B1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Linear(10, 10)
        self.c2 = nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c1(self.c2(x))


class A2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.b1 = A2B1()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b1(x)


class A1B1C1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.d1 = nn.Linear(10, 10)
        self.d2 = nn.ReLU()  # type: nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.d1(self.d2(x))


class A1B1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = A1B1C1()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c1.forward(x)


class A1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.b1 = A1B1()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b1(x)


class TestNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a1 = A1()
        self.a2 = A2()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a1(self.a2(x))


class TestPrintModelStatistics(unittest.TestCase):
    """
    Unittest for printing model statistics.
    """

    maxDiff = 1000

    def test_pretty_statistics(self) -> None:
        """
        Tests converting integer statistics to pretty strings.
        """

        # Test default settings
        formatted = _pretty_statistics(test_statistics)
        self.assertEqual(string_statistics, formatted)

        # Test hiding zeros
        formatted = _pretty_statistics(test_statistics, hide_zero=True)
        self.assertEqual(formatted[""]["stat3"], "")
        self.assertEqual(formatted["a1.b1.c1.d1"]["stat1"], "")
        self.assertEqual(formatted["a1.b1.c1.d1"]["stat2"], "")
        self.assertEqual(formatted["a2.b1"]["stat1"], "")
        self.assertEqual(formatted["a2.b1.c2"]["stat1"], "")
        self.assertEqual(formatted["a2"]["stat1"], "0.123M")  # Others unaffected

        # Test changing significant figures
        formatted = _pretty_statistics(test_statistics, sig_figs=2)
        self.assertEqual(formatted["a2"]["stat1"], "0.12M")
        formatted = _pretty_statistics(test_statistics, sig_figs=4)
        self.assertEqual(formatted["a2"]["stat1"], "0.1235M")

    def test_group_by_module(self) -> None:
        """
        Tests changing stats[mods[values]] into mods[stats[values]]
        """
        grouped = _group_by_module(ungrouped_stats)
        self.assertEqual(grouped, test_statistics)

    def test_indicate_uncalled_modules(self) -> None:
        """
        Tests replacing stats for uncalled modules with an indicator string.
        """

        stat1_uncalled = {"", "a2.b1"}
        stat3_uncalled = {"", "a1"}

        stats = _indicate_uncalled_modules(
            statistics=string_statistics,
            stat_name="stat1",
            uncalled_modules=stat1_uncalled,
        )

        self.assertEqual(stats[""]["stat1"], "N/A")
        self.assertEqual(stats["a2.b1"]["stat1"], "N/A")
        self.assertEqual(stats["a1.b1"]["stat1"], "20")  # Other mod unaffected
        self.assertEqual(stats[""]["stat2"], "0.6M")  # Other stat unaffected

        # Test alternate string and setting a stat not in the dict
        stats = _indicate_uncalled_modules(
            statistics=string_statistics,
            stat_name="stat3",
            uncalled_modules=stat3_uncalled,
            uncalled_indicator="*",
        )

        self.assertEqual(stats[""]["stat3"], "*")
        self.assertEqual(stats["a1"]["stat3"], "*")

    def test_remove_zero_statistics(self) -> None:
        """
        Tests removing mods whose statistics are all zero.
        """

        stats = _remove_zero_statistics(test_statistics)

        self.assertFalse("a1.b1.c1.d1" in stats)
        self.assertFalse("a2.b1.c2" in stats)
        self.assertEqual(stats["a2.b1"]["stat1"], 0)  # Partial zeros remain

        # Test forcing a module to be kept
        keep = {"a1.b1.c1.d1"}
        stats = _remove_zero_statistics(test_statistics, force_keep=keep)

        self.assertFalse("a2.b1.c2" in stats)
        self.assertEqual(stats["a1.b1.c1.d1"]["stat1"], 0)
        self.assertEqual(stats["a1.b1.c1.d1"]["stat2"], 0)

        # Test requiring children to be zero
        modified_test = {mod: stats.copy() for mod, stats in test_statistics.items()}
        modified_test["a1.b1.c1.d1.e1"] = {"stat1": 40}  # Non-zero child
        modified_test["a2.b1.c2.d1"] = {"stat1": 0}  # Zero child

        stats = _remove_zero_statistics(modified_test, require_trivial_children=True)

        self.assertTrue("a1.b1.c1.d1" in stats)  # Kept because of non-zero child
        self.assertTrue("a1.b1.c1.d1.e1" in stats)  # Non-zero child
        self.assertFalse("a2.b1.c2" in stats)  # Removed because child is zero
        self.assertFalse("a2.b1.c2.d1" in stats)  # Removed child

    def test_fill_missing_statistics(self) -> None:
        """
        Tests filling missing statistics.
        """
        stat1 = {
            "": 4000,
            "a1": 20,
            "a1.b1": 20,
            "a1.b1.c1": 20,
            "a1.b1.c1.d1": 0,
            "a1.b1.c1.d2": 100,
            "a2": 123456,
            "a2.b1": 0,
            "a2.b1.c1": 200,
            "a2.b1.c2": 0,
        }

        stat2 = {
            "": 600000,
            "a1": 3000,
            "a1.b1": 3000,
            "a1.b1.c1": 3000,
            "a1.b1.c1.d1": 0,
            "a1.b1.c1.d2": 0,
            "a2": 654321,
            "a2.b1": 100,
            "a2.b1.c1": 300,
            "a2.b1.c2": 0,
        }

        stat3 = {
            "": 0,
            "a1": 0,
            "a1.b1": 0,
            "a1.b1.c1": 0,
            "a1.b1.c1.d1": 0,
            "a1.b1.c1.d2": 0,
            "a2": 0,
            "a2.b1": 40,
            "a2.b1.c1": 0,
            "a2.b1.c2": 0,
        }

        filled_stats = {"stat1": stat1, "stat2": stat2, "stat3": stat3}
        model = TestNet()
        filled = _fill_missing_statistics(model, ungrouped_stats)
        self.assertEqual(filled_stats, filled)

    def test_model_stats_str(self) -> None:
        """
        Test the output of printing a model with statistics.
        """

        model = TestNet()
        model_str = _model_stats_str(model, string_statistics)

        self.assertTrue("stat1: 4K, stat2: 0.6M, stat3: 0" in model_str)
        self.assertTrue("ReLU(stat1: 100)" in model_str)  # Inline
        self.assertTrue("stat1: 0\n" in model_str)  # Single stat
        self.assertTrue("      (c1): A1B1C1(\n" in model_str)  # submod with indent

        # Expected:

        # "TestNet(\n"
        # "  stat1: 4K, stat2: 0.6M, stat3: 0\n"
        # "  (a1): A1(\n"
        # "    stat1: 20, stat2: 3K\n"
        # "    (b1): A1B1(\n"
        # "      stat1: 20, stat2: 3K\n"
        # "      (c1): A1B1C1(\n"
        # "        stat1: 20, stat2: 3K\n"
        # "        (d1): Linear(\n"
        # "          in_features=10, out_features=10, bias=True\n"
        # "          stat1: 0, stat2: 0\n"
        # "        )\n"
        # "        (d2): ReLU(stat1: 100)\n"
        # "      )\n"
        # "    )\n"
        # "  )\n"
        # "  (a2): A2(\n"
        # "    stat1: 0.123M, stat2: 0.654M\n"
        # "    (b1): A2B1(\n"
        # "      stat1: 0, stat2: 100, stat3: 40\n"
        # "      (c1): Linear(\n"
        # "        in_features=10, out_features=10, bias=True\n"
        # "        stat1: 0.2K, stat2: 0.3K\n"
        # "      )\n"
        # "      (c2): Linear(\n"
        # "        in_features=10, out_features=10, bias=True\n"
        # "        stat1: 0\n"
        # "      )\n"
        # "    )\n"
        # "  )\n"
        # ")"

    def test_model_stats_table(self) -> None:

        stat_columns = ["stat1", "stat2", "stat3"]
        table = _model_stats_table(string_statistics, stat_columns=stat_columns)

        self.assertTrue("a1.b1.c1" in table)  # Don't remove end of wrapper
        self.assertFalse(" a1.b1 " in table)  # Remove multilevel wrappers
        self.assertTrue("a2" in table)  # Keep modules with different stats
        self.assertTrue("a2.b1" in table)  # Keep modules with multiple children
        self.assertTrue("   100" in table)  # Correct indentation
        self.assertFalse("    100" in table)  # Correct indentation

        # Expected:

        # "| module        | stat1   | stat2   | stat3   |\n"
        # "|:--------------|:--------|:--------|:--------|\n"
        # "| model         | 4.0K    | 0.6M    | 0       |\n"
        # "|  a1.b1.c1     |  20     |  3.0K   |         |\n"
        # "|   a1.b1.c1.d1 |   0     |   0     |         |\n"
        # "|   a1.b1.c1.d2 |   100   |         |         |\n"
        # "|  a2           |  0.123M |  0.654M |         |\n"
        # "|   a2.b1       |   0     |   100   |   40    |\n"
        # "|    a2.b1.c1   |    0.2K |    0.3K |         |\n"
        # "|    a2.b1.c2   |    0    |         |         |"

        # Test changing max depth
        table = _model_stats_table(
            string_statistics, stat_columns=stat_columns, max_depth=2
        )

        self.assertTrue("a1.b1.c1.d1" in table)  # Skipping wrappers reaches deeper
        self.assertTrue(" a2.b1 " in table)  # Get to depth 2
        self.assertFalse(" a2.b1.c1 " in table)  # Don't get to depth 3

        # Expected:

        # "| module        | stat1   | stat2   | stat3   |\n"
        # "|:--------------|:--------|:--------|:--------|\n"
        # "| model         | 4.0K    | 0.6M    | 0       |\n"
        # "|  a1.b1.c1     |  20     |  3.0K   |         |\n"
        # "|   a1.b1.c1.d1 |   0     |   0     |         |\n"
        # "|   a1.b1.c1.d2 |   100   |         |         |\n"
        # "|  a2           |  0.123M |  0.654M |         |\n"
        # "|   a2.b1       |   0     |   100   |   40    |"

    def test_flop_count_table(self) -> None:

        model = TestNet()
        inputs = (torch.randn((1, 10)),)

        table = flop_count_table(FlopCountAnalysis(model, inputs))

        self.assertFalse(" a1 " in table)  # Wrapper skipping successful
        self.assertFalse("a1.b1.c1.d1.bias" in table)  # Didn't go to depth 4
        self.assertTrue("a1.b1.c1.d1" in table)  # Did go to depth 3
        self.assertTrue(" a1.b1 " in table)  # Didn't skip different stats
        self.assertTrue("a2.b1.c1.weight" in table)  # Weights incuded
        self.assertTrue("(10, 10)" in table)  # Shapes included
        self.assertTrue(" a2.b1 " in table)  # Didn't skip through mod with >1 child
        self.assertFalse("#activations" in table)  # No activations
        self.assertTrue(" 0.33K" in table)  # Pretty stats, correct indentation
        self.assertFalse("  0.33K" in table)  # Correct indentation
        self.assertTrue("#parameters or shape" in table)  # Correct header

        # Expected:
        # | module            | #parameters or shape   | #flops   |
        # |:-------------------|:-----------------------|:---------|
        # | model              | 0.33K                  | 0.3K     |
        # |  a1.b1             |  0.11K                 |  100     |
        # |   a1.b1.c1         |   0.11K                |   N/A    |
        # |    a1.b1.c1.d1     |    0.11K               |    100   |
        # |  a2.b1             |  0.22K                 |  0.2K    |
        # |   a2.b1.c1         |   0.11K                |   100    |
        # |    a2.b1.c1.weight |    (10, 10)            |          |
        # |    a2.b1.c1.bias   |    (10,)               |          |
        # |   a2.b1.c2         |   0.11K                |   100    |
        # |    a2.b1.c2.weight |    (10, 10)            |          |
        # |    a2.b1.c2.bias   |    (10,)               |          |

        # Test activations and no parameter shapes
        table = flop_count_table(
            flops=FlopCountAnalysis(model, inputs),
            activations=ActivationCountAnalysis(model, inputs),
            show_param_shapes=False,
        )

        self.assertTrue("#activations" in table)  # Activation header
        self.assertTrue("  20" in table)  # Activation value with correct indent
        self.assertFalse("#parameters or shape" in table)  # Correct header
        self.assertTrue("#parameters")  # Correct header
        self.assertFalse("a2.b1.c1.weight" in table)  # Weights not included
        self.assertFalse("(10, 10)" in table)  # Shapes not included
        self.assertFalse("a2.b1.c1.d2" in table)  # Skipped empty

        # Expected:

        # | module        | #parameters   | #flops   | #activations   |
        # |:---------------|:--------------|:---------|:---------------|
        # | model          | 0.33K         | 0.3K     | 30             |
        # |  a1.b1         |  0.11K        |  100     |  10            |
        # |   a1.b1.c1     |   0.11K       |   N/A    |   N/A          |
        # |    a1.b1.c1.d1 |    0.11K      |    100   |    10          |
        # |  a2.b1         |  0.22K        |  0.2K    |  20            |
        # |   a2.b1.c1     |   0.11K       |   100    |   10           |
        # |   a2.b1.c2     |   0.11K       |   100    |   10           |

    def test_flop_count_str(self) -> None:
        """
        Tests calculating model flops and outputing them in model print format.
        """

        model = TestNet()
        inputs = (torch.randn((1, 10)),)
        model_str = flop_count_str(FlopCountAnalysis(model, inputs))

        self.assertTrue("N/A indicates a possibly missing statistic" in model_str)
        self.assertTrue("n_params: 0.11K, n_flops: 100" in model_str)
        self.assertTrue("ReLU()" in model_str)  # Suppress trivial statistics
        self.assertTrue("n_params: 0.11K, n_flops: N/A" in model_str)  # Uncalled stats
        self.assertTrue("[[1, 10]]")  # Input sizes

        # Expected:

        # "Input sizes (torch.Tensor only): [[1, 10]]\n"
        # "N/A indicates a possibly missing statistic due to how the "
        # "module was called. Missing values are still included in the "
        # "parent's total.\n"
        # "TestNet(\n"
        # "  n_params: 0.33K, n_flops: 0.3K\n"
        # "  (a1): A1(\n"
        # "    n_params: 0.11K, n_flops: 100\n"
        # "    (b1): A1B1(\n"
        # "      n_params: 0.11K, n_flops: 100\n"
        # "      (c1): A1B1C1(\n"
        # "        n_params: 0.11K, n_flops: N/A\n"
        # "        (d1): Linear(\n"
        # "          in_features=10, out_features=10, bias=True\n"
        # "          n_params: 0.11K, n_flops: 100\n"
        # "        )\n"
        # "        (d2): ReLU()\n"
        # "      )\n"
        # "    )\n"
        # "  )\n"
        # "  (a2): A2(\n"
        # "    n_params: 0.22K, n_flops: 0.2K\n"
        # "    (b1): A2B1(\n"
        # "      n_params: 0.22K, n_flops: 0.2K\n"
        # "      (c1): Linear(\n"
        # "        in_features=10, out_features=10, bias=True\n"
        # "        n_params: 0.11K, n_flops: 100\n"
        # "      )\n"
        # "      (c2): Linear(\n"
        # "        in_features=10, out_features=10, bias=True\n"
        # "        n_params: 0.11K, n_flops: 100\n"
        # "      )\n"
        # "    )\n"
        # "  )\n"
        # ")"

        # Test with activations
        model_str = flop_count_str(
            FlopCountAnalysis(model, inputs),
            activations=ActivationCountAnalysis(model, inputs),
        )

        self.assertTrue("n_params: 0.33K, n_flops: 0.3K, n_acts: 30" in model_str)
        self.assertTrue("n_params: 0.11K, n_flops: N/A, n_acts: N/A" in model_str)

        # Expected:

        # "Input sizes (torch.Tensor only): [[1, 10]]\n"
        # "N/A indicates a possibly missing statistic due to how the "
        # "module was called. Missing values are still included in the "
        # "parent's total.\n"
        # "TestNet(\n"
        # "  n_params: 0.33K, n_flops: 0.3K, n_acts: 30\n"
        # "  (a1): A1(\n"
        # "    n_params: 0.11K, n_flops: 100, n_acts: 10\n"
        # "    (b1): A1B1(\n"
        # "      n_params: 0.11K, n_flops: 100, n_acts: 10\n"
        # "      (c1): A1B1C1(\n"
        # "        n_params: 0.11K, n_flops: N/A, n_acts: N/A\n"
        # "        (d1): Linear(\n"
        # "          in_features=10, out_features=10, bias=True\n"
        # "          n_params: 0.11K, n_flops: 100, n_acts: 10\n"
        # "        )\n"
        # "        (d2): ReLU()\n"
        # "      )\n"
        # "    )\n"
        # "  )\n"
        # "  (a2): A2(\n"
        # "    n_params: 0.22K, n_flops: 0.2K, n_acts: 20\n"
        # "    (b1): A2B1(\n"
        # "      n_params: 0.22K, n_flops: 0.2K, n_acts: 20\n"
        # "      (c1): Linear(\n"
        # "        in_features=10, out_features=10, bias=True\n"
        # "        n_params: 0.11K, n_flops: 100, n_acts: 10\n"
        # "      )\n"
        # "      (c2): Linear(\n"
        # "        in_features=10, out_features=10, bias=True\n"
        # "        n_params: 0.11K, n_flops: 100, n_acts: 10\n"
        # "      )\n"
        # "    )\n"
        # "  )\n"
        # ")"
