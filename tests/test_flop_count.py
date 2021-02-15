# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# pyre-ignore-all-errors[2]
import typing
import unittest
from collections import Counter, defaultdict
from typing import Any, Dict

import torch
import torch.nn as nn
from fvcore.nn.flop_count import Handle, flop_count
from fvcore.nn.jit_handles import batchnorm_flop_jit


class ThreeNet(nn.Module):
    """
    A network with three layers. This is used for testing a network with more
    than one operation. The network has a convolution layer followed by two
    fully connected layers.
    """

    def __init__(self, input_dim: int, conv_dim: int, linear_dim: int) -> None:
        super(ThreeNet, self).__init__()
        self.conv = nn.Conv2d(input_dim, conv_dim, 1, 1)
        out_dim = 1
        self.pool = nn.AdaptiveAvgPool2d((out_dim, out_dim))  # type: nn.Module
        self.linear1 = nn.Linear(conv_dim, linear_dim)
        self.linear2 = nn.Linear(linear_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class ConvNet(nn.Module):
    """
    A network with a single convolution layer. This is used for testing flop
    count for convolution layers.
    """

    def __init__(
        self,
        conv_dim: int,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        spatial_dim: int,
        stride: int,
        padding: int,
        groups_num: int,
    ) -> None:
        super(ConvNet, self).__init__()
        if conv_dim == 1:
            convLayer = nn.Conv1d
        elif conv_dim == 2:
            convLayer = nn.Conv2d
        else:
            convLayer = nn.Conv3d

        self.conv = convLayer(
            input_dim, output_dim, kernel_size, stride, padding, groups=groups_num
        )  # type: nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class LinearNet(nn.Module):
    """
    A network with a single fully connected layer. This is used for testing flop
    count for fully connected layers.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x


class EinsumNet(nn.Module):
    """
    A network with a single torch.einsum operation. This is used for testing
    flop count for torch.einsum.
    """

    def __init__(self, equation: str) -> None:
        super(EinsumNet, self).__init__()
        self.eq = equation  # type: str

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.einsum(self.eq, x, y)
        return x


class MatmulNet(nn.Module):
    """
    A network with a single torch.matmul operation. This is used for testing
    flop count for torch.matmul.
    """

    def __init__(self) -> None:
        super(MatmulNet, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x, y)
        return x


class BMMNet(nn.Module):
    """
    A network with a single torch.bmm operation. This is used for testing
    flop count for torch.bmm.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.bmm(x, y)
        return x


class CustomNet(nn.Module):
    """
    A network with a fully connected layer followed by a sigmoid layer. This is
    used for testing customized operation handles.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(CustomNet, self).__init__()
        self.conv = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  # type: nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class TestFlopCount(unittest.TestCase):
    """
    Unittest for flop_count.
    """

    def test_customized_ops(self) -> None:
        """
        Test the use of customized operation handles. The first test checks the
        case when a new handle for a new operation is passed as an argument.
        The second case checks when a new handle for a default operation is
        passed. The new handle should overwrite the default handle.
        """
        # New handle for a new operation.
        def dummy_sigmoid_flop_jit(
            inputs: typing.List[Any], outputs: typing.List[Any]
        ) -> typing.Counter[str]:
            """
            A dummy handle function for sigmoid. Note the handle here does
            not compute actual flop count. This is used for test only.
            """
            flop_dict = Counter()
            flop_dict["sigmoid"] = 10000
            return flop_dict

        batch_size = 10
        input_dim = 5
        output_dim = 4
        customNet = CustomNet(input_dim, output_dim)
        custom_ops: Dict[str, Handle] = {"aten::sigmoid": dummy_sigmoid_flop_jit}
        x = torch.rand(batch_size, input_dim)
        flop_dict1, _ = flop_count(customNet, (x,), supported_ops=custom_ops)
        flop_sigmoid = 10000 / 1e9
        self.assertEqual(
            flop_dict1["sigmoid"],
            flop_sigmoid,
            "Customized operation handle failed to pass the flop count test.",
        )

        # New handle that overwrites a default handle addmm. So now the new
        # handle counts flops for the fully connected layer.
        def addmm_dummy_flop_jit(
            inputs: typing.List[object], outputs: typing.List[object]
        ) -> typing.Counter[str]:
            """
            A dummy handle function for fully connected layers. This overwrites
            the default handle. Note the handle here does not compute actual
            flop count. This is used for test only.
            """
            flop_dict = Counter()
            flop_dict["addmm"] = 400000
            return flop_dict

        custom_ops2: Dict[str, Handle] = {"aten::addmm": addmm_dummy_flop_jit}
        flop_dict2, _ = flop_count(customNet, (x,), supported_ops=custom_ops2)
        flop = 400000 / 1e9
        self.assertEqual(
            flop_dict2["addmm"],
            flop,
            "Customized operation handle failed to pass the flop count test.",
        )

    def test_nn(self) -> None:
        """
        Test a model which is a pre-defined nn.module without defining a new
        customized network.
        """
        batch_size = 5
        input_dim = 8
        output_dim = 4
        x = torch.randn(batch_size, input_dim)
        flop_dict, _ = flop_count(nn.Linear(input_dim, output_dim), (x,))
        gt_flop = batch_size * input_dim * output_dim / 1e9
        gt_dict = defaultdict(float)
        gt_dict["addmm"] = gt_flop
        self.assertDictEqual(
            flop_dict, gt_dict, "nn.Linear failed to pass the flop count test."
        )

    def test_skip_ops(self) -> None:
        """
        Test the return of skipped operations.
        """
        batch_size = 10
        input_dim = 5
        output_dim = 4
        customNet = CustomNet(input_dim, output_dim)
        x = torch.rand(batch_size, input_dim)
        _, skip_dict = flop_count(customNet, (x,))
        gt_dict = Counter()
        gt_dict["aten::sigmoid"] = 1
        self.assertDictEqual(
            skip_dict, gt_dict, "Skipped operations failed to pass the flop count test."
        )

    def test_linear(self) -> None:
        """
        Test a network with a single fully connected layer.
        """
        batch_size = 5
        input_dim = 10
        output_dim = 20
        linearNet = LinearNet(input_dim, output_dim)
        x = torch.randn(batch_size, input_dim)
        flop_dict, _ = flop_count(linearNet, (x,))
        gt_flop = batch_size * input_dim * output_dim / 1e9
        gt_dict = defaultdict(float)
        gt_dict["addmm"] = gt_flop
        self.assertDictEqual(
            flop_dict,
            gt_dict,
            "Fully connected layer failed to pass the flop count test.",
        )

    def test_conv(self) -> None:
        """
        Test a network with a single convolution layer. The test cases are: 1)
        2D convolution; 2) 2D convolution with change in spatial dimensions; 3)
        group convolution; 4) depthwise convolution； 5) 1d convolution; 6) 3d
        convolution.
        """

        def _test_conv(
            conv_dim: int,
            batch_size: int,
            input_dim: int,
            output_dim: int,
            spatial_dim: int,
            kernel_size: int,
            padding: int,
            stride: int,
            group_size: int,
        ) -> None:
            convNet = ConvNet(
                conv_dim,
                input_dim,
                output_dim,
                kernel_size,
                spatial_dim,
                stride,
                padding,
                group_size,
            )
            assert conv_dim in [1, 2, 3], "Convolution dimension needs to be 1, 2, or 3"
            if conv_dim == 1:
                x = torch.randn(batch_size, input_dim, spatial_dim)
            elif conv_dim == 2:
                x = torch.randn(batch_size, input_dim, spatial_dim, spatial_dim)
            else:
                x = torch.randn(
                    batch_size, input_dim, spatial_dim, spatial_dim, spatial_dim
                )

            flop_dict, _ = flop_count(convNet, (x,))
            spatial_out = ((spatial_dim + 2 * padding) - kernel_size) // stride + 1
            gt_flop = (
                batch_size
                * input_dim
                * output_dim
                * (kernel_size ** conv_dim)
                * (spatial_out ** conv_dim)
                / group_size
                / 1e9
            )
            gt_dict = defaultdict(float)
            gt_dict["conv"] = gt_flop
            self.assertDictEqual(
                flop_dict,
                gt_dict,
                "Convolution layer failed to pass the flop count test.",
            )

        # Test flop count for 2d convolution.
        conv_dim1 = 2
        batch_size1 = 5
        input_dim1 = 10
        output_dim1 = 3
        spatial_dim1 = 15
        kernel_size1 = 3
        padding1 = 1
        stride1 = 1
        group_size1 = 1
        _test_conv(
            conv_dim1,
            batch_size1,
            input_dim1,
            output_dim1,
            spatial_dim1,
            kernel_size1,
            padding1,
            stride1,
            group_size1,
        )

        # Test flop count for convolution with spatial change in output.
        conv_dim2 = 2
        batch_size2 = 2
        input_dim2 = 10
        output_dim2 = 6
        spatial_dim2 = 20
        kernel_size2 = 3
        padding2 = 1
        stride2 = 2
        group_size2 = 1
        _test_conv(
            conv_dim2,
            batch_size2,
            input_dim2,
            output_dim2,
            spatial_dim2,
            kernel_size2,
            padding2,
            stride2,
            group_size2,
        )

        # Test flop count for group convolution.
        conv_dim3 = 2
        batch_size3 = 5
        input_dim3 = 16
        output_dim3 = 8
        spatial_dim3 = 15
        kernel_size3 = 5
        padding3 = 2
        stride3 = 1
        group_size3 = 4
        _test_conv(
            conv_dim3,
            batch_size3,
            input_dim3,
            output_dim3,
            spatial_dim3,
            kernel_size3,
            padding3,
            stride3,
            group_size3,
        )

        # Test the special case of group convolution when group = output_dim.
        # This is equivalent to depthwise convolution.
        conv_dim4 = 2
        batch_size4 = 5
        input_dim4 = 16
        output_dim4 = 8
        spatial_dim4 = 15
        kernel_size4 = 3
        padding4 = 1
        stride4 = 1
        group_size4 = output_dim4
        _test_conv(
            conv_dim4,
            batch_size4,
            input_dim4,
            output_dim4,
            spatial_dim4,
            kernel_size4,
            padding4,
            stride4,
            group_size4,
        )

        # Test flop count for 1d convolution.
        conv_dim5 = 1
        batch_size5 = 5
        input_dim5 = 10
        output_dim5 = 3
        spatial_dim5 = 15
        kernel_size5 = 3
        padding5 = 1
        stride5 = 1
        group_size5 = 1
        _test_conv(
            conv_dim5,
            batch_size5,
            input_dim5,
            output_dim5,
            spatial_dim5,
            kernel_size5,
            padding5,
            stride5,
            group_size5,
        )

        # Test flop count for 3d convolution.
        conv_dim6 = 3
        batch_size6 = 5
        input_dim6 = 10
        output_dim6 = 3
        spatial_dim6 = 15
        kernel_size6 = 3
        padding6 = 1
        stride6 = 1
        group_size6 = 1
        _test_conv(
            conv_dim6,
            batch_size6,
            input_dim6,
            output_dim6,
            spatial_dim6,
            kernel_size6,
            padding6,
            stride6,
            group_size6,
        )

    def test_matmul(self) -> None:
        """
        Test flop count for operation matmul.
        """
        m = 20
        n = 10
        p = 100
        mNet = MatmulNet()
        x = torch.randn(m, n)
        y = torch.randn(n, p)
        flop_dict, _ = flop_count(mNet, (x, y))
        gt_flop = m * n * p / 1e9
        gt_dict = defaultdict(float)
        gt_dict["matmul"] = gt_flop
        self.assertDictEqual(
            flop_dict, gt_dict, "Matmul operation failed to pass the flop count test."
        )

    def test_matmul_broadcast(self) -> None:
        """
        Test flop count for operation matmul.
        """
        m = 20
        n = 10
        p = 100
        mNet = MatmulNet()
        x = torch.randn(1, m, n)
        y = torch.randn(1, n, p)
        flop_dict, _ = flop_count(mNet, (x, y))
        gt_flop = m * n * p / 1e9
        gt_dict = defaultdict(float)
        gt_dict["matmul"] = gt_flop
        self.assertDictEqual(
            flop_dict, gt_dict, "Matmul operation failed to pass the flop count test."
        )

        x = torch.randn(2, 2, m, n)
        y = torch.randn(2, 2, n, p)
        flop_dict, _ = flop_count(mNet, (x, y))
        gt_flop = 4 * m * n * p / 1e9
        gt_dict = defaultdict(float)
        gt_dict["matmul"] = gt_flop
        self.assertDictEqual(
            flop_dict, gt_dict, "Matmul operation failed to pass the flop count test."
        )


    def test_bmm(self) -> None:
        """
        Test flop count for operation torch.bmm. The case checkes
        torch.bmm with equation nct,ntp->ncp.
        """
        n = 2
        c = 5
        t = 2
        p = 12
        eNet = BMMNet()
        x = torch.randn(n, c, t)
        y = torch.randn(n, t, p)
        flop_dict, _ = flop_count(eNet, (x, y))
        gt_flop = n * t * p * c / 1e9
        gt_dict = defaultdict(float)
        gt_dict["bmm"] = gt_flop
        self.assertDictEqual(
            flop_dict,
            gt_dict,
            "bmm operation nct,ncp->ntp failed to pass the flop count test.",
        )

    def test_einsum(self) -> None:
        """
        Test flop count for operation torch.einsum. The first case checkes
        torch.einsum with equation nct,ncp->ntp. The second case checkes
        torch.einsum with equation "ntg,ncg->nct".
        """
        equation = "nct,ncp->ntp"
        n = 1
        c = 5
        t = 2
        p = 12
        eNet = EinsumNet(equation)
        x = torch.randn(n, c, t)
        y = torch.randn(n, c, p)
        flop_dict, _ = flop_count(eNet, (x, y))
        gt_flop = n * t * p * c / 1e9
        gt_dict = defaultdict(float)
        gt_dict["einsum"] = gt_flop
        self.assertDictEqual(
            flop_dict,
            gt_dict,
            "Einsum operation nct,ncp->ntp failed to pass the flop count test.",
        )

        equation = "ntg,ncg->nct"
        g = 6
        eNet = EinsumNet(equation)
        x = torch.randn(n, t, g)
        y = torch.randn(n, c, g)
        flop_dict, _ = flop_count(eNet, (x, y))
        gt_flop = n * t * g * c / 1e9
        gt_dict = defaultdict(float)
        gt_dict["einsum"] = gt_flop
        self.assertDictEqual(
            flop_dict,
            gt_dict,
            "Einsum operation ntg,ncg->nct failed to pass the flop count test.",
        )

    def test_batchnorm(self) -> None:
        """
        Test flop count for operation batchnorm. The test cases include
        BatchNorm1d, BatchNorm2d and BatchNorm3d.
        """
        # Test for BatchNorm1d.
        supported_ops: Dict[str, Handle] = {"aten::batch_norm": batchnorm_flop_jit}
        batch_size = 10
        input_dim = 10
        batch_1d = nn.BatchNorm1d(input_dim, affine=False)
        x = torch.randn(batch_size, input_dim)
        flop_dict, _ = flop_count(batch_1d, (x,), supported_ops)
        gt_flop = 4 * batch_size * input_dim / 1e9
        gt_dict = defaultdict(float)
        gt_dict["batchnorm"] = gt_flop
        self.assertDictEqual(
            flop_dict, gt_dict, "BatchNorm1d failed to pass the flop count test."
        )

        # Test for BatchNorm2d.
        batch_size = 10
        input_dim = 10
        spatial_dim_x = 5
        spatial_dim_y = 5
        batch_2d = nn.BatchNorm2d(input_dim, affine=False)
        x = torch.randn(batch_size, input_dim, spatial_dim_x, spatial_dim_y)
        flop_dict, _ = flop_count(batch_2d, (x,), supported_ops)
        gt_flop = 4 * batch_size * input_dim * spatial_dim_x * spatial_dim_y / 1e9
        gt_dict = defaultdict(float)
        gt_dict["batchnorm"] = gt_flop
        self.assertDictEqual(
            flop_dict, gt_dict, "BatchNorm2d failed to pass the flop count test."
        )

        # Test for BatchNorm3d.
        batch_size = 10
        input_dim = 10
        spatial_dim_x = 5
        spatial_dim_y = 5
        spatial_dim_z = 5
        batch_3d = nn.BatchNorm3d(input_dim, affine=False)
        x = torch.randn(
            batch_size, input_dim, spatial_dim_x, spatial_dim_y, spatial_dim_z
        )
        flop_dict, _ = flop_count(batch_3d, (x,), supported_ops)
        gt_flop = (
            4
            * batch_size
            * input_dim
            * spatial_dim_x
            * spatial_dim_y
            * spatial_dim_z
            / 1e9
        )
        gt_dict = defaultdict(float)
        gt_dict["batchnorm"] = gt_flop
        self.assertDictEqual(
            flop_dict, gt_dict, "BatchNorm3d failed to pass the flop count test."
        )

    def test_threeNet(self) -> None:
        """
        Test a network with more than one layer. The network has a convolution
        layer followed by two fully connected layers.
        """
        batch_size = 4
        input_dim = 2
        conv_dim = 5
        spatial_dim = 10
        linear_dim = 3
        x = torch.randn(batch_size, input_dim, spatial_dim, spatial_dim)
        threeNet = ThreeNet(input_dim, conv_dim, linear_dim)
        flop1 = batch_size * conv_dim * input_dim * spatial_dim * spatial_dim / 1e9
        flop_linear1 = batch_size * conv_dim * linear_dim / 1e9
        flop_linear2 = batch_size * linear_dim * 1 / 1e9
        flop2 = flop_linear1 + flop_linear2
        flop_dict, _ = flop_count(threeNet, (x,))
        gt_dict = defaultdict(float)
        gt_dict["conv"] = flop1
        gt_dict["addmm"] = flop2
        self.assertDictEqual(
            flop_dict,
            gt_dict,
            "The three-layer network failed to pass the flop count test.",
        )
