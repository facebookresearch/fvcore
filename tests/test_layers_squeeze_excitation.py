# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import itertools
import unittest
from typing import Iterable

import torch

from fvcore.nn.squeeze_excitation import (
    ChannelSpatialSqueezeExcitation,
    SpatialSqueezeExcitation,
    SqueezeExcitation,
)


class TestSqueezeExcitation(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

    def test_build_se(self) -> None:
        """
        Test SE model builder.
        """
        for layer, num_channels, is_3d in itertools.product(
            (
                SqueezeExcitation,
                SpatialSqueezeExcitation,
                ChannelSpatialSqueezeExcitation,
            ),
            (16, 32),
            (True, False),
        ):
            model = layer(
                num_channels=num_channels,
                is_3d=is_3d,
            )

            # Test forwarding.
            for input_tensor in TestSqueezeExcitation._get_inputs(
                num_channels=num_channels, is_3d=is_3d
            ):
                if input_tensor.shape[1] != num_channels:
                    with self.assertRaises(RuntimeError):
                        output_tensor = model(input_tensor)
                    continue
                else:
                    output_tensor = model(input_tensor)

                input_shape = input_tensor.shape
                output_shape = output_tensor.shape

                self.assertEqual(
                    input_shape,
                    output_shape,
                    "Input shape {} is different from output shape {}".format(
                        input_shape, output_shape
                    ),
                )

    @staticmethod
    def _get_inputs3d(num_channels: int = 8) -> Iterable[torch.Tensor]:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random tensor as test cases.
        shapes = (
            # Forward succeeded.
            (1, num_channels, 5, 7, 7),
            (2, num_channels, 5, 7, 7),
            (4, num_channels, 5, 7, 7),
            (4, num_channels, 5, 7, 7),
            (4, num_channels, 7, 7, 7),
            (4, num_channels, 7, 7, 14),
            (4, num_channels, 7, 14, 7),
            (4, num_channels, 7, 14, 14),
            # Forward failed.
            (8, num_channels * 2, 3, 7, 7),
            (8, num_channels * 4, 5, 7, 7),
        )
        for shape in shapes:
            yield torch.rand(shape)

    @staticmethod
    def _get_inputs2d(num_channels: int = 8) -> Iterable[torch.Tensor]:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random tensor as test cases.
        shapes = (
            # Forward succeeded.
            (1, num_channels, 7, 7),
            (2, num_channels, 7, 7),
            (4, num_channels, 7, 7),
            (4, num_channels, 7, 14),
            (4, num_channels, 14, 7),
            (4, num_channels, 14, 14),
            # Forward failed.
            (8, num_channels * 2, 7, 7),
            (8, num_channels * 4, 7, 7),
        )
        for shape in shapes:
            yield torch.rand(shape)

    @staticmethod
    def _get_inputs(
        num_channels: int = 8, is_3d: bool = False
    ) -> Iterable[torch.Tensor]:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        if is_3d:
            return TestSqueezeExcitation._get_inputs3d(num_channels=num_channels)
        else:
            return TestSqueezeExcitation._get_inputs2d(num_channels=num_channels)
