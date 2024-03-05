# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# pyre-strict

from typing import Optional

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """
    Generic 2d/3d extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    Squeezing spatially and exciting channel-wise
    """

    block: nn.Module
    is_3d: bool

    def __init__(
        self,
        num_channels: int,
        num_channels_reduced: Optional[int] = None,
        reduction_ratio: float = 2.0,
        is_3d: bool = False,
        activation: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            num_channels (int): Number of input channels.
            num_channels_reduced (int):
                Number of reduced channels. If none, uses reduction_ratio to calculate.
            reduction_ratio (float):
                How much num_channels should be reduced if num_channels_reduced is not provided.
            is_3d (bool): Whether we're operating on 3d data (or 2d), default 2d.
            activation (nn.Module): Activation function used, defaults to ReLU.
        """
        super().__init__()

        if num_channels_reduced is None:
            num_channels_reduced = int(num_channels // reduction_ratio)

        if activation is None:
            activation = nn.ReLU()

        if is_3d:
            conv1 = nn.Conv3d(
                num_channels, num_channels_reduced, kernel_size=1, bias=True
            )
            conv2 = nn.Conv3d(
                num_channels_reduced, num_channels, kernel_size=1, bias=True
            )
        else:
            conv1 = nn.Conv2d(
                num_channels, num_channels_reduced, kernel_size=1, bias=True
            )
            conv2 = nn.Conv2d(
                num_channels_reduced, num_channels, kernel_size=1, bias=True
            )

        self.is_3d = is_3d
        self.block = nn.Sequential(
            conv1,
            activation,
            conv2,
            nn.Sigmoid(),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: X, shape = (batch_size, num_channels, H, W).
                For 3d X, shape = (batch_size, num_channels, T, H, W).
            output tensor
        """
        mean_tensor = (
            input_tensor.mean(dim=[2, 3, 4], keepdim=True)
            if self.is_3d
            else input_tensor.mean(dim=[2, 3], keepdim=True)
        )
        output_tensor = torch.mul(input_tensor, self.block(mean_tensor))

        return output_tensor


class SpatialSqueezeExcitation(nn.Module):
    """
    Generic 2d/3d extension of SE block
        squeezing channel-wise and exciting spatially described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation
        in Fully Convolutional Networks, MICCAI 2018*
    """

    block: nn.Module

    def __init__(
        self,
        num_channels: int,
        is_3d: bool = False,
    ) -> None:
        """
        Args:
            num_channels (int): Number of input channels.
            is_3d (bool): Whether we're operating on 3d data.
        """
        super().__init__()

        if is_3d:
            conv = nn.Conv3d(num_channels, 1, kernel_size=1, bias=True)
        else:
            conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=True)

        self.block = nn.Sequential(
            conv,
            nn.Sigmoid(),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: X, shape = (batch_size, num_channels, H, W).
                For 3d X, shape = (batch_size, num_channels, T, H, W).
            output tensor
        """
        output_tensor = torch.mul(input_tensor, self.block(input_tensor))

        return output_tensor


class ChannelSpatialSqueezeExcitation(nn.Module):
    """
    Generic 2d/3d extension of concurrent spatial and channel squeeze & excitation:
         *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation
         in Fully Convolutional Networks, arXiv:1803.02579*
    """

    def __init__(
        self,
        num_channels: int,
        num_channels_reduced: Optional[int] = None,
        reduction_ratio: float = 16.0,
        is_3d: bool = False,
        activation: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            num_channels (int): Number of input channels.
            num_channels_reduced (int):
                Number of reduced channels. If none, uses reduction_ratio to calculate.
            reduction_ratio (float):
                How much num_channels should be reduced if num_channels_reduced is not provided.
            is_3d (bool): Whether we're operating on 3d data (or 2d), default 2d.
            activation (nn.Module): Activation function used, defaults to ReLU.
        """
        super().__init__()
        self.channel = SqueezeExcitation(
            num_channels=num_channels,
            num_channels_reduced=num_channels_reduced,
            reduction_ratio=reduction_ratio,
            is_3d=is_3d,
            activation=activation,
        )
        self.spatial = SpatialSqueezeExcitation(num_channels=num_channels, is_3d=is_3d)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: X, shape = (batch_size, num_channels, H, W)
                For 3d X, shape = (batch_size, num_channels, T, H, W)
            output tensor
        """
        output_tensor = torch.max(
            self.channel(input_tensor), self.spatial(input_tensor)
        )

        return output_tensor
