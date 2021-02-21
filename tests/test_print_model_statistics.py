# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
import torch.nn as nn
from fvcore.nn.print_model_statistics import (
    fill_missing_statistics,
    indicate_uncalled_modules,
    merge_records,
    model_flops_str,
    model_flops_table,
    model_stats_str,
    model_stats_table,
    pretty_statistics,
    remove_zero_statistics,
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
    "": {"stat1": "4.0K", "stat2": "0.6M", "stat3": "0"},
    "a1": {"stat1": "20", "stat2": "3.0K"},
    "a1.b1": {"stat1": "20", "stat2": "3.0K"},
    "a1.b1.c1": {"stat1": "20", "stat2": "3.0K"},
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

unmerged_stats = {"stat1": stat1, "stat2": stat2, "stat3": stat3}


class A2B1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Linear(10, 10)
        self.c2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.c1(self.c2(x))


class A2(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = A2B1()

    def forward(self, x):
        return self.b1(x)


class A1B1C1(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Linear(10, 10)
        self.d2 = nn.ReLU()

    def forward(self, x):
        return self.d1(self.d2(x))


class A1B1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = A1B1C1()

    def forward(self, x):
        return self.c1.forward(x)


class A1(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = A1B1()

    def forward(self, x):
        return self.b1(x)


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = A1()
        self.a2 = A2()

    def forward(self, x):
        return self.a1(self.a2(x))


class TestPrintModelStatistics(unittest.TestCase):
    """
    Unittest for printing model statistics.
    """

    def test_pretty_statistics(self) -> None:
        """
        Tests converting integer statistics to pretty strings.
        """

        # Test default settings
        formatted = pretty_statistics(test_statistics)
        self.assertEqual(string_statistics, formatted)

        # Test hiding zeros
        formatted = pretty_statistics(test_statistics, hide_zero=True)
        self.assertEqual(formatted[""]["stat3"], "")
        self.assertEqual(formatted["a1.b1.c1.d1"]["stat1"], "")
        self.assertEqual(formatted["a1.b1.c1.d1"]["stat2"], "")
        self.assertEqual(formatted["a2.b1"]["stat1"], "")
        self.assertEqual(formatted["a2.b1.c2"]["stat1"], "")
        self.assertEqual(formatted["a2"]["stat1"], "0.123M")  # Others unaffected

        # Test changing significant figures
        formatted = pretty_statistics(test_statistics, sig_figs=2)
        self.assertEqual(formatted["a2"]["stat1"], "0.12M")
        formatted = pretty_statistics(test_statistics, sig_figs=4)
        self.assertEqual(formatted["a2"]["stat1"], "0.1235M")

    def test_merge_records(self) -> None:
        """
        Tests changing stats[mods[values]] into mods[stats[values]]
        """
        merged = merge_records(unmerged_stats)
        self.assertEqual(merged, test_statistics)

    def test_indicate_uncalled_modules(self) -> None:
        """
        Tests replacing stats for uncalled modules with an indicator string.
        """

        stat1_uncalled = {"", "a2.b1"}
        stat3_uncalled = {"", "a1"}

        stats = indicate_uncalled_modules(
            statistics=string_statistics,
            stat_name="stat1",
            uncalled_modules=stat1_uncalled,
        )

        self.assertEqual(stats[""]["stat1"], "N/A")
        self.assertEqual(stats["a2.b1"]["stat1"], "N/A")
        self.assertEqual(stats["a1.b1"]["stat1"], "20")  # Other mod unaffected
        self.assertEqual(stats[""]["stat2"], "0.6M")  # Other stat unaffected

        # Test alternate string and setting a stat not in the dict
        stats = indicate_uncalled_modules(
            statistics=string_statistics,
            stat_name="stat3",
            uncalled_modules=stat3_uncalled,
            indicator="*",
        )

        self.assertEqual(stats[""]["stat3"], "*")
        self.assertEqual(stats["a1"]["stat3"], "*")

    def test_remove_zero_statistics(self) -> None:
        """
        Tests removing mods whose statistics are all zero.
        """

        stats = remove_zero_statistics(test_statistics)

        self.assertFalse("a1.b1.c1.d1" in stats)
        self.assertFalse("a2.b1.c2" in stats)
        self.assertEqual(stats["a2.b1"]["stat1"], 0)  # Partial zeros remain

        # Test forcing a module to be kept
        keep = {"a1.b1.c1.d1"}
        stats = remove_zero_statistics(test_statistics, force_keep=keep)

        self.assertFalse("a2.b1.c2" in stats)
        self.assertEqual(stats["a1.b1.c1.d1"]["stat1"], 0)
        self.assertEqual(stats["a1.b1.c1.d1"]["stat2"], 0)

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
        filled = fill_missing_statistics(model, unmerged_stats)
        self.assertEqual(filled_stats, filled)

    def test_model_stats_str(self) -> None:
        """
        Test the output of printing a model with statistics.
        """

        model = TestNet()
        model_string = model_stats_str(model, string_statistics)

        gt_string = (
            "TestNet(\n"
            "  stat1: 4.0K, stat2: 0.6M, stat3: 0\n"
            "  (a1): A1(\n"
            "    stat1: 20, stat2: 3.0K\n"
            "    (b1): A1B1(\n"
            "      stat1: 20, stat2: 3.0K\n"
            "      (c1): A1B1C1(\n"
            "        stat1: 20, stat2: 3.0K\n"
            "        (d1): Linear(\n"
            "          in_features=10, out_features=10, bias=True\n"
            "          stat1: 0, stat2: 0\n"
            "        )\n"
            "        (d2): ReLU(stat1: 100)\n"
            "      )\n"
            "    )\n"
            "  )\n"
            "  (a2): A2(\n"
            "    stat1: 0.123M, stat2: 0.654M\n"
            "    (b1): A2B1(\n"
            "      stat1: 0, stat2: 100, stat3: 40\n"
            "      (c1): Linear(\n"
            "        in_features=10, out_features=10, bias=True\n"
            "        stat1: 0.2K, stat2: 0.3K\n"
            "      )\n"
            "      (c2): Linear(\n"
            "        in_features=10, out_features=10, bias=True\n"
            "        stat1: 0\n"
            "      )\n"
            "    )\n"
            "  )\n"
            ")"
        )

        self.assertEqual(model_string, gt_string)

    def test_model_stats_table(self) -> None:

        stat_columns = ["stat1", "stat2", "stat3"]
        table = model_stats_table(string_statistics, stat_columns=stat_columns)
        gt_table = (
            "| model         | stat1   | stat2   | stat3   |\n"
            "|:--------------|:--------|:--------|:--------|\n"
            "| model         | 4.0K    | 0.6M    | 0       |\n"
            "|  a1.b1.c1     |  20     |  3.0K   |         |\n"
            "|   a1.b1.c1.d1 |   0     |   0     |         |\n"
            "|   a1.b1.c1.d2 |   100   |         |         |\n"
            "|  a2           |  0.123M |  0.654M |         |\n"
            "|   a2.b1       |   0     |   100   |   40    |\n"
            "|    a2.b1.c1   |    0.2K |    0.3K |         |\n"
            "|    a2.b1.c2   |    0    |         |         |"
        )

        self.assertEqual(table, gt_table)

        # Test fast forwarding with missing data
        uncalled_stat1 = {"a1", "a1.b1"}
        uncalled_stat2 = {"a1", "a1.b1.c1"}
        stats = indicate_uncalled_modules(string_statistics, "stat1", uncalled_stat1)
        stats = indicate_uncalled_modules(stats, "stat2", uncalled_stat2)
        table = model_stats_table(
            stats, stat_columns=stat_columns, missing_indicator="N/A"
        )

        self.assertEqual(table, gt_table)

        # Test without fast forwarding
        table = model_stats_table(
            string_statistics, stat_columns=stat_columns, skip_wrappers=False
        )
        gt_table = (
            "| model       | stat1   | stat2   | stat3   |\n"
            "|:------------|:--------|:--------|:--------|\n"
            "| model       | 4.0K    | 0.6M    | 0       |\n"
            "|  a1         |  20     |  3.0K   |         |\n"
            "|   a1.b1     |   20    |   3.0K  |         |\n"
            "|    a1.b1.c1 |    20   |    3.0K |         |\n"
            "|  a2         |  0.123M |  0.654M |         |\n"
            "|   a2.b1     |   0     |   100   |   40    |\n"
            "|    a2.b1.c1 |    0.2K |    0.3K |         |\n"
            "|    a2.b1.c2 |    0    |         |         |"
        )

        self.assertEqual(table, gt_table)

        # Test changing max depth
        table = model_stats_table(
            string_statistics, stat_columns=stat_columns, max_depth=2
        )
        gt_table = (
            "| model         | stat1   | stat2   | stat3   |\n"
            "|:--------------|:--------|:--------|:--------|\n"
            "| model         | 4.0K    | 0.6M    | 0       |\n"
            "|  a1.b1.c1     |  20     |  3.0K   |         |\n"
            "|   a1.b1.c1.d1 |   0     |   0     |         |\n"
            "|   a1.b1.c1.d2 |   100   |         |         |\n"
            "|  a2           |  0.123M |  0.654M |         |\n"
            "|   a2.b1       |   0     |   100   |   40    |"
        )

        self.assertEqual(table, gt_table)

    def test_model_flops_table(self) -> None:

        model = TestNet()
        inputs = (torch.randn((1, 10)),)

        table = model_flops_table(model, inputs)
        gt_table = (
            "| model                | #parameters or shape   | #flops   |\n"
            "|:---------------------|:-----------------------|:---------|\n"
            "| model                | 0.33K                  | 0.3K     |\n"
            "|  a1.b1.c1.d1         |  0.11K                 |  100     |\n"
            "|   a1.b1.c1.d1.weight |   (10, 10)             |          |\n"
            "|   a1.b1.c1.d1.bias   |   (10,)                |          |\n"
            "|  a2.b1               |  0.22K                 |  0.2K    |\n"
            "|   a2.b1.c1           |   0.11K                |   100    |\n"
            "|    a2.b1.c1.weight   |    (10, 10)            |          |\n"
            "|    a2.b1.c1.bias     |    (10,)               |          |\n"
            "|   a2.b1.c2           |   0.11K                |   100    |\n"
            "|    a2.b1.c2.weight   |    (10, 10)            |          |\n"
            "|    a2.b1.c2.bias     |    (10,)               |          |"
        )

        self.assertEqual(table, gt_table)

        # Test activations and no parameter shapes
        table = model_flops_table(
            model, inputs, activations=True, show_param_shapes=False
        )
        gt_table = (
            "| model        | #parameters   | #flops   |   #activations |\n"
            "|:-------------|:--------------|:---------|---------------:|\n"
            "| model        | 0.33K         | 0.3K     |             30 |\n"
            "|  a1.b1.c1.d1 |  0.11K        |  100     |             10 |\n"
            "|  a2.b1       |  0.22K        |  0.2K    |             20 |\n"
            "|   a2.b1.c1   |   0.11K       |   100    |             10 |\n"
            "|   a2.b1.c2   |   0.11K       |   100    |             10 |"
        )

        self.assertEqual(table, gt_table)

        # Test not skipping wrappers
        table = model_flops_table(model, inputs, skip_wrappers=False)
        gt_table = (
            "| model       | #parameters or shape   | #flops   |\n"
            "|:------------|:-----------------------|:---------|\n"
            "| model       | 0.33K                  | 0.3K     |\n"
            "|  a1         |  0.11K                 |  100     |\n"
            "|   a1.b1     |   0.11K                |   100    |\n"
            "|    a1.b1.c1 |    0.11K               |    N/A   |\n"
            "|  a2         |  0.22K                 |  0.2K    |\n"
            "|   a2.b1     |   0.22K                |   0.2K   |\n"
            "|    a2.b1.c1 |    0.11K               |    100   |\n"
            "|    a2.b1.c2 |    0.11K               |    100   |"
        )

        self.assertEqual(table, gt_table)

        # Test not skipping empty
        table = model_flops_table(model, inputs, skip_empty=False)
        gt_table = (
            "| model                 | #parameters or shape   | #flops   |\n"
            "|:----------------------|:-----------------------|:---------|\n"
            "| model                 | 0.33K                  | 0.3K     |\n"
            "|  a1.b1.c1             |  0.11K                 |  100     |\n"
            "|   a1.b1.c1.d1         |   0.11K                |   100    |\n"
            "|    a1.b1.c1.d1.weight |    (10, 10)            |          |\n"
            "|    a1.b1.c1.d1.bias   |    (10,)               |          |\n"
            "|   a1.b1.c1.d2         |                        |          |\n"
            "|  a2.b1                |  0.22K                 |  0.2K    |\n"
            "|   a2.b1.c1            |   0.11K                |   100    |\n"
            "|    a2.b1.c1.weight    |    (10, 10)            |          |\n"
            "|    a2.b1.c1.bias      |    (10,)               |          |\n"
            "|   a2.b1.c2            |   0.11K                |   100    |\n"
            "|    a2.b1.c2.weight    |    (10, 10)            |          |\n"
            "|    a2.b1.c2.bias      |    (10,)               |          |"
        )

        self.assertEqual(table, gt_table)

    def test_model_flops_str(self) -> None:
        """
        Tests calculating model flops and outputing them in model print format.
        """

        model = TestNet()
        inputs = (torch.randn((1, 10)),)
        model_str = model_flops_str(model, inputs)

        gt_str = (
            "Input sizes (torch.Tensor only): [[1, 10]]\n"
            "N/A indicates a possibly missing statistic due to how the "
            "module was called. Missing values are still included in the "
            "parent's total.\n"
            "TestNet(\n"
            "  n_params: 0.33K, n_flops: 0.3K\n"
            "  (a1): A1(\n"
            "    n_params: 0.11K, n_flops: 100\n"
            "    (b1): A1B1(\n"
            "      n_params: 0.11K, n_flops: 100\n"
            "      (c1): A1B1C1(\n"
            "        n_params: 0.11K, n_flops: N/A\n"
            "        (d1): Linear(\n"
            "          in_features=10, out_features=10, bias=True\n"
            "          n_params: 0.11K, n_flops: 100\n"
            "        )\n"
            "        (d2): ReLU()\n"
            "      )\n"
            "    )\n"
            "  )\n"
            "  (a2): A2(\n"
            "    n_params: 0.22K, n_flops: 0.2K\n"
            "    (b1): A2B1(\n"
            "      n_params: 0.22K, n_flops: 0.2K\n"
            "      (c1): Linear(\n"
            "        in_features=10, out_features=10, bias=True\n"
            "        n_params: 0.11K, n_flops: 100\n"
            "      )\n"
            "      (c2): Linear(\n"
            "        in_features=10, out_features=10, bias=True\n"
            "        n_params: 0.11K, n_flops: 100\n"
            "      )\n"
            "    )\n"
            "  )\n"
            ")"
        )

        self.assertEqual(model_str, gt_str)

        # Test with activations
        model_str = model_flops_str(model, inputs, activations=True)

        gt_str = (
            "Input sizes (torch.Tensor only): [[1, 10]]\n"
            "N/A indicates a possibly missing statistic due to how the "
            "module was called. Missing values are still included in the "
            "parent's total.\n"
            "TestNet(\n"
            "  n_params: 0.33K, n_flops: 0.3K, n_acts: 30\n"
            "  (a1): A1(\n"
            "    n_params: 0.11K, n_flops: 100, n_acts: 10\n"
            "    (b1): A1B1(\n"
            "      n_params: 0.11K, n_flops: 100, n_acts: 10\n"
            "      (c1): A1B1C1(\n"
            "        n_params: 0.11K, n_flops: N/A, n_acts: N/A\n"
            "        (d1): Linear(\n"
            "          in_features=10, out_features=10, bias=True\n"
            "          n_params: 0.11K, n_flops: 100, n_acts: 10\n"
            "        )\n"
            "        (d2): ReLU()\n"
            "      )\n"
            "    )\n"
            "  )\n"
            "  (a2): A2(\n"
            "    n_params: 0.22K, n_flops: 0.2K, n_acts: 20\n"
            "    (b1): A2B1(\n"
            "      n_params: 0.22K, n_flops: 0.2K, n_acts: 20\n"
            "      (c1): Linear(\n"
            "        in_features=10, out_features=10, bias=True\n"
            "        n_params: 0.11K, n_flops: 100, n_acts: 10\n"
            "      )\n"
            "      (c2): Linear(\n"
            "        in_features=10, out_features=10, bias=True\n"
            "        n_params: 0.11K, n_flops: 100, n_acts: 10\n"
            "      )\n"
            "    )\n"
            "  )\n"
            ")"
        )

        self.assertEqual(model_str, gt_str)
