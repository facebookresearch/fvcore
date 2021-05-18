# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# -*- coding: utf-8 -*-

import itertools
import unittest
from typing import List, Tuple

import numpy as np
import torch
from fvcore.nn import update_bn_stats
from torch import nn


class TestPreciseBN(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_rng_state(torch.manual_seed(42).get_state())

    @staticmethod
    def compute_bn_stats(
        tensors: List[torch.Tensor], dims: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        mean_of_batch_var = (
            # pyre-ignore
            torch.stack([tensor.var(dim=dims, unbiased=True) for tensor in tensors])
            .mean(dim=0)
            .numpy()
        )
        var = torch.cat(tensors, dim=0).var(dim=dims, unbiased=False).numpy()
        return mean, mean_of_batch_var, var

    def test_precise_bn(self) -> None:
        # Number of batches to test.
        NB = 8
        _bn_types = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        _stats_dims = [[0, 2], [0, 2, 3], [0, 2, 3, 4]]
        _input_dims = [(16, 8, 24), (16, 8, 24, 8), (16, 8, 4, 12, 6)]
        assert len({len(_bn_types), len(_stats_dims), len(_input_dims)}) == 1

        for bn, stats_dim, input_dim in zip(_bn_types, _stats_dims, _input_dims):
            model = bn(input_dim[1])
            model.train()
            tensors = [torch.randn(input_dim) for _ in range(NB)]
            mean, mean_of_batch_var, var = TestPreciseBN.compute_bn_stats(
                tensors, stats_dim
            )

            old_weight = model.weight.detach().numpy()

            update_bn_stats(
                model,
                itertools.cycle(tensors),
                len(tensors),
            )
            self.assertTrue(np.allclose(model.running_mean.numpy(), mean))
            self.assertTrue(np.allclose(model.running_var.numpy(), var))

            # Test that the new estimator can handle varying batch size
            # It should obtain same results as earlier if the same input data are
            # split into different batch sizes.
            tensors = torch.split(torch.cat(tensors, dim=0), [2, 2, 4, 8, 16, 32, 64])
            update_bn_stats(
                model,
                itertools.cycle(tensors),
                len(tensors),
            )
            self.assertTrue(np.allclose(model.running_mean.numpy(), mean))
            self.assertTrue(np.allclose(model.running_var.numpy(), var))
            self.assertTrue(np.allclose(model.weight.detach().numpy(), old_weight))

    def test_precise_bn_insufficient_data(self) -> None:
        input_dim = (16, 32, 24, 24)
        model = nn.BatchNorm2d(input_dim[1])
        model.train()
        tensor = torch.randn(input_dim)
        with self.assertRaises(AssertionError):
            update_bn_stats(model, itertools.repeat(tensor, 10), 20)
