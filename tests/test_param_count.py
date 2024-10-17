# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# pyre-strict


import unittest

from fvcore.nn.parameter_count import parameter_count, parameter_count_table
from torch import nn


class NetWithReuse(nn.Module):
    # pyre-fixme[11]: Annotation `bool` is not defined as a type.
    def __init__(self, reuse: bool = False) -> None:
        super().__init__()
        # pyre-fixme[29]: `type[Conv2d]` is not a function.
        self.conv1 = nn.Conv2d(100, 100, 3)
        # pyre-fixme[29]: `type[Conv2d]` is not a function.
        self.conv2 = nn.Conv2d(100, 100, 3)
        if reuse:
            self.conv2.weight = self.conv1.weight


class NetWithDupPrefix(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # pyre-fixme[29]: `type[Conv2d]` is not a function.
        self.conv1 = nn.Conv2d(100, 100, 3)
        # pyre-fixme[29]: `type[Conv2d]` is not a function.
        self.conv111 = nn.Conv2d(100, 100, 3)


class TestParamCount(unittest.TestCase):
    def test_param(self) -> None:
        # pyre-fixme[29]: `type[NetWithReuse]` is not a function.
        net = NetWithReuse()
        count = parameter_count(net)
        self.assertTrue(count[""], 180200)
        self.assertTrue(count["conv2"], 90100)

    def test_param_with_reuse(self) -> None:
        # pyre-fixme[29]: `type[NetWithReuse]` is not a function.
        net = NetWithReuse(reuse=True)
        count = parameter_count(net)
        self.assertTrue(count[""], 90200)
        self.assertTrue(count["conv2"], 100)

    def test_param_with_same_prefix(self) -> None:
        # pyre-fixme[29]: `type[NetWithDupPrefix]` is not a function.
        net = NetWithDupPrefix()
        table = parameter_count_table(net)
        c = ["conv111.weight" in line for line in table.split("\n")]
        self.assertEqual(
            sum(c), 1
        )  # it only appears once, despite being a prefix of conv1
