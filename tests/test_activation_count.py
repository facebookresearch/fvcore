# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# pyre-ignore-all-errors[2]

import typing
import unittest
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from fvcore.nn.activation_count import activation_count, ActivationCountAnalysis
from fvcore.nn.jit_handles import Handle
from numpy import prod


class SmallConvNet(nn.Module):
    """
    A network with three conv layers. This is used for testing convolution
    layers for activation count.
    """

    def __init__(self, input_dim: int) -> None:
        super(SmallConvNet, self).__init__()
        conv_dim1 = 8
        conv_dim2 = 4
        conv_dim3 = 2
        self.conv1 = nn.Conv2d(input_dim, conv_dim1, 1, 1)
        self.conv2 = nn.Conv2d(conv_dim1, conv_dim2, 1, 2)
        self.conv3 = nn.Conv2d(conv_dim2, conv_dim3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def get_gt_activation(self, x: torch.Tensor) -> Tuple[int, int, int]:
        x = self.conv1(x)
        count1 = prod(list(x.size()))
        x = self.conv2(x)
        count2 = prod(list(x.size()))
        x = self.conv3(x)
        count3 = prod(list(x.size()))
        return (count1, count2, count3)


class TestActivationCountAnalysis(unittest.TestCase):
    """
    Unittest for activation_count.
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

    def test_conv2d(self) -> None:
        """
        Test the activation count for convolutions.
        """
        batch_size = 1
        input_dim = 3
        spatial_dim = 32
        x = torch.randn(batch_size, input_dim, spatial_dim, spatial_dim)
        convNet = SmallConvNet(input_dim)
        ac_dict, _ = activation_count(convNet, (x,))
        gt_count = sum(convNet.get_gt_activation(x))

        gt_dict = defaultdict(float)
        gt_dict["conv"] = gt_count / 1e6
        self.assertDictEqual(
            gt_dict,
            ac_dict,
            "ConvNet with 3 layers failed to pass the activation count test.",
        )

    def test_linear(self) -> None:
        """
        Test the activation count for fully connected layer.
        """
        batch_size = 1
        input_dim = 10
        output_dim = 20
        netLinear = nn.Linear(input_dim, output_dim)
        x = torch.randn(batch_size, input_dim)
        ac_dict, _ = activation_count(netLinear, (x,))
        gt_count = batch_size * output_dim
        gt_dict = defaultdict(float)
        gt_dict[self.lin_op] = gt_count / 1e6
        self.assertEquals(
            gt_dict, ac_dict, "FC layer failed to pass the activation count test."
        )

    def test_supported_ops(self) -> None:
        """
        Test the activation count for user provided handles.
        """

        def dummy_handle(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
            return Counter({"conv": 100})

        batch_size = 1
        input_dim = 3
        spatial_dim = 32
        x = torch.randn(batch_size, input_dim, spatial_dim, spatial_dim)
        convNet = SmallConvNet(input_dim)
        sp_ops: Dict[str, Handle] = {"aten::_convolution": dummy_handle}
        ac_dict, _ = activation_count(convNet, (x,), sp_ops)
        gt_dict = defaultdict(float)
        conv_layers = 3
        gt_dict["conv"] = 100 * conv_layers / 1e6
        self.assertDictEqual(
            gt_dict,
            ac_dict,
            "ConvNet with 3 layers failed to pass the activation count test.",
        )

    def test_activation_count_class(self) -> None:
        """
        Tests ActivationCountAnalysis.
        """
        batch_size = 1
        input_dim = 10
        output_dim = 20
        netLinear = nn.Linear(input_dim, output_dim)
        x = torch.randn(batch_size, input_dim)
        gt_count = batch_size * output_dim
        gt_dict = Counter(
            {
                "": gt_count,
            }
        )
        acts_counter = ActivationCountAnalysis(netLinear, (x,))
        self.assertEqual(acts_counter.by_module(), gt_dict)

        batch_size = 1
        input_dim = 3
        spatial_dim = 32
        x = torch.randn(batch_size, input_dim, spatial_dim, spatial_dim)
        convNet = SmallConvNet(input_dim)
        acts_counter = ActivationCountAnalysis(convNet, (x,))
        gt_counts = convNet.get_gt_activation(x)
        gt_dict = Counter(
            {
                "": sum(gt_counts),
                "conv1": gt_counts[0],
                "conv2": gt_counts[1],
                "conv3": gt_counts[2],
            }
        )

        self.assertDictEqual(gt_dict, acts_counter.by_module())
