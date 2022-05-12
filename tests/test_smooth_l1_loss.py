# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import numpy as np
import torch
from fvcore.nn import smooth_l1_loss


class TestSmoothL1Loss(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)

    def test_smooth_l1_loss(self) -> None:
        inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
        targets = torch.tensor([1.1, 2, 4.5], dtype=torch.float32)
        beta = 0.5
        loss = smooth_l1_loss(inputs, targets, beta=beta, reduction="none").numpy()
        self.assertTrue(np.allclose(loss, [0.5 * 0.1**2 / beta, 0, 1.5 - 0.5 * beta]))

        beta = 0.05
        loss = smooth_l1_loss(inputs, targets, beta=beta, reduction="none").numpy()
        self.assertTrue(np.allclose(loss, [0.1 - 0.5 * beta, 0, 1.5 - 0.5 * beta]))

    def test_empty_inputs(self) -> None:
        inputs = torch.empty([0, 10], dtype=torch.float32).requires_grad_()
        targets = torch.empty([0, 10], dtype=torch.float32)
        loss = smooth_l1_loss(inputs, targets, beta=0.5, reduction="mean")
        loss.backward()

        self.assertEqual(loss.detach().numpy(), 0.0)
        self.assertIsNotNone(inputs.grad)
