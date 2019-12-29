#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import math
import unittest
import torch
import torch.nn as nn

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill


class TestWeightInit(unittest.TestCase):
    """
    Test creation of WeightInit.
    """

    def setUp(self):
        torch.set_rng_state(torch.manual_seed(42).get_state())

    @staticmethod
    def msra_fill_std(fan_out: int) -> float:
        # Given the fan_out, calculate the expected standard deviation for msra
        # fill.
        return torch.as_tensor(math.sqrt(2.0 / fan_out))

    @staticmethod
    def xavier_fill_std(fan_in: int) -> float:
        # Given the fan_in, calculate the expected standard deviation for
        # xavier fill.
        return torch.as_tensor(math.sqrt(1.0 / fan_in))

    @staticmethod
    def weight_and_bias_dist_match(
        weight: torch.Tensor, bias: torch.Tensor, target_std: torch.Tensor
    ) -> bool:
        # When the size of the weight is relative small, sampling on a small
        # number of elements would not give us a standard deviation that close
        # enough to the expected distribution. So the default rtol of 1e-8 will
        # break some test cases. Therefore a larger rtol is used.
        weight_dist_match = torch.allclose(
            target_std, torch.std(weight), rtol=1e-2, atol=0
        )
        bias_dist_match = torch.nonzero(bias).nelement() == 0
        return weight_dist_match and bias_dist_match

    def test_conv_weight_init(self):
        # Test weight initialization for convolutional layers.
        kernel_sizes = [1, 3]
        channel_in_dims = [128, 256, 512, 1024]
        channel_out_dims = [256, 512, 1024, 2048]

        for layer in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            for k_size, c_in_dim, c_out_dim in itertools.product(
                kernel_sizes, channel_in_dims, channel_out_dims
            ):
                p = {
                    "kernel_size": k_size,
                    "in_channels": c_in_dim,
                    "out_channels": c_out_dim,
                }
                model = layer(**p)

                if layer is nn.Conv1d:
                    spatial_dim = k_size
                elif layer is nn.Conv2d:
                    spatial_dim = k_size ** 2
                elif layer is nn.Conv3d:
                    spatial_dim = k_size ** 3

                # Calculate fan_in and fan_out.
                fan_in = c_in_dim * spatial_dim
                fan_out = c_out_dim * spatial_dim

                # Msra weight init check.
                c2_msra_fill(model)
                self.assertTrue(
                    TestWeightInit.weight_and_bias_dist_match(
                        model.weight,
                        model.bias,
                        TestWeightInit.msra_fill_std(fan_out),
                    )
                )

                # Xavier weight init check.
                c2_xavier_fill(model)
                self.assertTrue(
                    TestWeightInit.weight_and_bias_dist_match(
                        model.weight,
                        model.bias,
                        TestWeightInit.xavier_fill_std(fan_in),
                    )
                )

    def test_linear_weight_init(self):
        # Test weight initialization for linear layer.
        channel_in_dims = [128, 256, 512, 1024]
        channel_out_dims = [256, 512, 1024, 2048]

        for layer in [nn.Linear]:
            for c_in_dim, c_out_dim in itertools.product(
                channel_in_dims, channel_out_dims
            ):
                p = {"in_features": c_in_dim, "out_features": c_out_dim}
                model = layer(**p)

                # Calculate fan_in and fan_out.
                fan_in = c_in_dim
                fan_out = c_out_dim

                # Msra weight init check.
                c2_msra_fill(model)
                self.assertTrue(
                    TestWeightInit.weight_and_bias_dist_match(
                        model.weight,
                        model.bias,
                        TestWeightInit.msra_fill_std(fan_out),
                    )
                )

                # Xavier weight init check.
                c2_xavier_fill(model)
                self.assertTrue(
                    TestWeightInit.weight_and_bias_dist_match(
                        model.weight,
                        model.bias,
                        TestWeightInit.xavier_fill_std(fan_in),
                    )
                )
