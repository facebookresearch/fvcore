#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import unittest
from typing import Tuple
import torch
from torch import nn

from fvcore.nn import update_bn_stats


class TestPreciseBN(unittest.TestCase):
    @staticmethod
    def compute_bn_stats(
        tensors: list, dims: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a list of random initialized tensors, compute the mean and
            variance.
        Args:
            tensors (list): list of randomly initialized tensors.
            dims (list): list of dimensions to compute the mean and variance.
        """
        mean = (
            torch.stack([tensor.mean(dim=dims) for tensor in tensors])
            .mean(dim=0)
            .numpy()
        )
        var = (
            torch.stack([tensor.var(dim=dims) for tensor in tensors])
            .mean(dim=0)
            .numpy()
        )
        return mean, var

    def test_precise_bn(self):
        # Number of batches to test.
        NB = 5
        _bn_types = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        _stats_dims = [[0, 2], [0, 2, 3], [0, 2, 3, 4]]
        _input_dims = [(16, 32, 24), (16, 32, 24, 24), (16, 32, 4, 12, 12)]
        assert len({len(_bn_types), len(_stats_dims), len(_input_dims)}) == 1

        for bn, stats_dim, input_dim in zip(
            _bn_types, _stats_dims, _input_dims
        ):
            model = bn(input_dim[1])
            model.train()
            tensors = [torch.randn(input_dim) for _ in range(NB)]
            mean, var = TestPreciseBN.compute_bn_stats(tensors, stats_dim)

            old_weight = model.weight.detach().numpy()
            update_bn_stats(model, itertools.cycle(tensors), NB * 100)

            self.assertTrue(np.allclose(model.running_mean.numpy(), mean))
            self.assertTrue(np.allclose(model.running_var.numpy(), var))
            self.assertTrue(
                np.allclose(model.weight.detach().numpy(), old_weight)
            )
