# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import unittest
from torch import nn

from fvcore.nn.parameter_count import parameter_count


class NetWithReuse(nn.Module):
    def __init__(self, reuse: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(100, 100, 3)
        self.conv2 = nn.Conv2d(100, 100, 3)
        if reuse:
            self.conv2.weight = self.conv1.weight  # pyre-ignore


class TestParamCount(unittest.TestCase):
    def test_param(self) -> None:
        net = NetWithReuse()
        count = parameter_count(net)
        self.assertTrue(count[""], 180200)
        self.assertTrue(count["conv2"], 90100)

    def test_param_with_reuse(self) -> None:
        net = NetWithReuse(reuse=True)
        count = parameter_count(net)
        self.assertTrue(count[""], 90200)
        self.assertTrue(count["conv2"], 100)
