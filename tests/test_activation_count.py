# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# pyre-ignore-all-errors[2]

import typing
import unittest
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from fvcore.nn.activation_count import ActivationCountAnalysis, activation_count
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


class LSTMNet(nn.Module):
    """
    A network with LSTM layers. This is used for testing flop
    count for LSTM layers.
    """

    def __init__(
            self,
            input_dim,
            hidden_dim,
            lstm_layers,
            bias,
            batch_first,
            bidirectional,
            proj_size
    ) -> None:
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim,
                            hidden_dim,
                            lstm_layers,
                            bias=bias,
                            batch_first=batch_first,
                            bidirectional=bidirectional,
                            proj_size=proj_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lstm(x)
        return x


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

    def test_lstm(self) -> None:
        """
        Test a network with a single fully connected layer.
        """

        class LSTMCellNet(nn.Module):
            """
            A network with a single LSTM cell. This is used for testing if the flop
            count of LSTM layers equals the flop count of an LSTM cell for one time-step.
            """

            def __init__(
                    self,
                    input_dim,
                    hidden_dim,
                    bias: bool
            ) -> None:
                super(LSTMCellNet, self).__init__()
                self.lstm_cell = nn.LSTMCell(input_size=input_dim,
                                             hidden_size=hidden_dim,
                                             bias=bias)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.lstm_cell(x[0])
                return x

        def _test_lstm(
                batch_size,
                time_dim,
                input_dim,
                hidden_dim,
                lstm_layers,
                proj_size,
                bidirectional=False,
                bias=True,
                batch_first=True,
        ):
            lstmNet = LSTMNet(input_dim, hidden_dim, lstm_layers, bias, batch_first, bidirectional, proj_size)
            x = torch.randn(time_dim, batch_size, input_dim)
            ac_dict, _ = activation_count(lstmNet, (x,))

            lstmcellNet = LSTMCellNet(input_dim, hidden_dim, bias)
            lstmcell_ac_dict, _ = activation_count(lstmcellNet, (x,))

            if time_dim == 1 and lstm_layers == 1:
                gt_dict = defaultdict(float)
                gt_dict["lstm"] = sum(e for _, e in lstmcell_ac_dict.items())
            elif time_dim == 5 and lstm_layers == 5 and bidirectional:
                gt_dict = defaultdict(float)
                gt_dict["lstm"] = sum(e for _, e in lstmcell_ac_dict.items()) * time_dim * lstm_layers * 2
            elif time_dim == 5 and lstm_layers == 5:
                gt_dict = defaultdict(float)
                gt_dict["lstm"] = sum(e for _, e in lstmcell_ac_dict.items()) * time_dim * lstm_layers
            else:
                raise ValueError(
                    f'No test implemented for parameters "time_dim": {time_dim}, "lstm_layers": {lstm_layers}'
                    f' and "bidirectional": {bidirectional}.'
                )

            self.assertAlmostEqual(
                ac_dict['lstm'],
                gt_dict['lstm'],
                msg="LSTM layer failed to pass the flop count test.",
            )

        # Test LSTM for 1 layer and 1 time step.
        batch_size1 = 5
        time_dim1 = 1
        input_dim1 = 3
        hidden_dim1 = 4
        lstm_layers1 = 1
        bidirectional1 = False
        proj_size1 = 0

        _test_lstm(
            batch_size1,
            time_dim1,
            input_dim1,
            hidden_dim1,
            lstm_layers1,
            proj_size1,
            bidirectional1,
        )

        # Test LSTM for 5 layers and 5 time steps.
        batch_size2 = 5
        time_dim2 = 5
        input_dim2 = 3
        hidden_dim2 = 4
        lstm_layers2 = 5
        bidirectional2 = False
        proj_size2 = 0

        _test_lstm(
            batch_size2,
            time_dim2,
            input_dim2,
            hidden_dim2,
            lstm_layers2,
            proj_size2,
            bidirectional2,
        )

        # Test bidirectional LSTM for 5 layers and 5 time steps.
        batch_size3 = 5
        time_dim3 = 5
        input_dim3 = 3
        hidden_dim3 = 4
        lstm_layers3 = 5
        bidirectional3 = True
        proj_size3 = 0

        _test_lstm(
            batch_size3,
            time_dim3,
            input_dim3,
            hidden_dim3,
            lstm_layers3,
            proj_size3,
            bidirectional3,
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
