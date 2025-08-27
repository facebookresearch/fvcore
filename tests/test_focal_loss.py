# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# pyre-strict
# pyre-ignore-all-errors[58]

import typing
import unittest

import numpy as np
import torch
from fvcore.nn import (
    sigmoid_focal_loss,
    sigmoid_focal_loss_jit,
    sigmoid_focal_loss_star,
    sigmoid_focal_loss_star_jit,
)
from torch.nn import functional as F


def logit(p: torch.Tensor) -> torch.Tensor:
    return torch.log(p / (1 - p))


class TestFocalLoss(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)

    def test_focal_loss_equals_ce_loss(self) -> None:
        """
        No weighting of easy/hard (gamma = 0) or positive/negative (alpha = 0).
        """
        inputs = logit(
            torch.tensor([[[0.95], [0.90], [0.98], [0.99]]], dtype=torch.float32)
        )
        targets = torch.tensor([[[1], [1], [1], [1]]], dtype=torch.float32)
        inputs_fl = inputs.clone().requires_grad_()
        targets_fl = targets.clone()
        inputs_ce = inputs.clone().requires_grad_()
        targets_ce = targets.clone()

        focal_loss = sigmoid_focal_loss(
            inputs_fl, targets_fl, gamma=0, alpha=-1, reduction="mean"
        )
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs_ce, targets_ce, reduction="mean"
        )

        self.assertEqual(ce_loss, focal_loss.data)
        focal_loss.backward()
        ce_loss.backward()
        # pyre-fixme[16]: Optional type has no attribute `data`.
        self.assertTrue(torch.allclose(inputs_fl.grad.data, inputs_ce.grad.data))

    def test_easy_ex_focal_loss_less_than_ce_loss(self) -> None:
        """
        With gamma = 2 loss of easy examples is downweighted.
        """
        N = 5
        inputs = logit(torch.rand(N))
        targets = torch.randint(0, 2, (N,)).float()
        focal_loss = sigmoid_focal_loss(inputs, targets, gamma=2, alpha=-1)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss_ratio = (ce_loss / focal_loss).squeeze()
        prob = torch.sigmoid(inputs)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        correct_ratio = 1.0 / ((1.0 - p_t) ** 2)
        self.assertTrue(np.allclose(loss_ratio, correct_ratio))

    def test_easy_ex_focal_loss_weighted_less_than_ce_loss(self) -> None:
        """
        With gamma = 2, alpha = 0.5 loss of easy examples is downweighted.
        """
        inputs = logit(
            torch.tensor([[[0.95], [0.90], [0.6], [0.3]]], dtype=torch.float64)
        )
        targets = torch.tensor([[[1], [1], [1], [1]]], dtype=torch.float64)
        focal_loss = sigmoid_focal_loss(inputs, targets, gamma=2, alpha=0.5)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss_ratio = (ce_loss / focal_loss).squeeze()
        correct_ratio = 2.0 / ((1.0 - inputs.squeeze().sigmoid()) ** 2)
        self.assertTrue(np.allclose(loss_ratio, correct_ratio))

    def test_hard_ex_focal_loss_similar_to_ce_loss(self) -> None:
        """
        With gamma = 2 loss of hard examples is unchanged.
        """
        inputs = logit(torch.tensor([0.05, 0.12, 0.09, 0.17], dtype=torch.float32))
        targets = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
        focal_loss = sigmoid_focal_loss(inputs, targets, gamma=2, alpha=-1)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss_ratio = (ce_loss / focal_loss).squeeze()
        correct_ratio = 1.0 / ((1.0 - inputs.sigmoid()) ** 2)
        self.assertTrue(np.allclose(loss_ratio, correct_ratio))

    def test_negatives_ignored_focal_loss(self) -> None:
        """
        With alpha = 1 negative examples have focal loss of 0.
        """
        inputs = logit(
            torch.tensor([[[0.05], [0.12], [0.89], [0.79]]], dtype=torch.float32)
        )
        targets = torch.tensor([[[1], [1], [0], [0]]], dtype=torch.float32)
        focal_loss = (
            sigmoid_focal_loss(inputs, targets, gamma=2, alpha=1).squeeze().numpy()
        )
        ce_loss = (
            F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            .squeeze()
            .numpy()
        )
        targets = targets.squeeze().numpy()
        self.assertTrue(np.all(ce_loss[targets == 0] > 0))
        self.assertTrue(np.all(focal_loss[targets == 0] == 0))

    def test_positives_ignored_focal_loss(self) -> None:
        """
        With alpha = 0 postive examples have focal loss of 0.
        """
        inputs = logit(
            torch.tensor([[[0.05], [0.12], [0.89], [0.79]]], dtype=torch.float32)
        )
        targets = torch.tensor([[[1], [1], [0], [0]]], dtype=torch.float32)
        focal_loss = (
            sigmoid_focal_loss(inputs, targets, gamma=2, alpha=0).squeeze().numpy()
        )
        ce_loss = (
            F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            .squeeze()
            .numpy()
        )
        targets = targets.squeeze().numpy()
        self.assertTrue(np.all(ce_loss[targets == 1] > 0))
        self.assertTrue(np.all(focal_loss[targets == 1] == 0))

    def test_mean_focal_loss_equals_ce_loss(self) -> None:
        """
        Mean value of focal loss across all examples matches ce loss.
        """
        inputs = logit(
            torch.tensor(
                [[0.05, 0.9], [0.52, 0.45], [0.89, 0.8], [0.39, 0.5]],
                dtype=torch.float32,
            )
        )
        targets = torch.tensor([[1, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)
        focal_loss = sigmoid_focal_loss(
            inputs, targets, gamma=0, alpha=-1, reduction="mean"
        )
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
        self.assertEqual(ce_loss, focal_loss)

    def test_sum_focal_loss_equals_ce_loss(self) -> None:
        """
        Sum of focal loss across all examples matches ce loss.
        """
        inputs = logit(
            torch.tensor([[[0.05], [0.12], [0.89], [0.79]]], dtype=torch.float32)
        )
        targets = torch.tensor([[[1], [1], [0], [0]]], dtype=torch.float32)
        focal_loss = sigmoid_focal_loss(
            inputs, targets, gamma=0, alpha=-1, reduction="sum"
        )
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="sum")
        self.assertEqual(ce_loss, focal_loss)

    def test_focal_loss_equals_ce_loss_multi_class(self) -> None:
        """
        Focal loss with predictions for multiple classes matches ce loss.
        """
        inputs = logit(
            torch.tensor(
                [
                    [
                        [0.95, 0.55, 0.12, 0.05],
                        [0.09, 0.95, 0.36, 0.11],
                        [0.06, 0.12, 0.56, 0.07],
                        [0.09, 0.15, 0.25, 0.45],
                    ]
                ],
                dtype=torch.float32,
            )
        )
        targets = torch.tensor(
            [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
            dtype=torch.float32,
        )
        focal_loss = sigmoid_focal_loss(
            inputs, targets, gamma=0, alpha=-1, reduction="mean"
        )
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
        self.assertEqual(ce_loss, focal_loss)

    # pyre-fixme[56]: Argument `not torch.cuda.is_available()` to decorator factory
    #  `unittest.skipIf` could not be resolved in a global scope.
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_focal_loss_equals_ce_loss_jit(self) -> None:
        """
        No weighting of easy/hard (gamma = 0) or positive/negative (alpha = 0).
        """
        device = torch.device("cuda:0")
        N = 5
        inputs = logit(torch.rand(N)).to(device)
        targets = torch.randint(0, 2, (N,)).float().to(device)
        focal_loss = sigmoid_focal_loss_jit(inputs, targets, gamma=0, alpha=-1)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs.cpu(), targets.cpu(), reduction="none"
        )
        self.assertTrue(np.allclose(ce_loss, focal_loss.cpu()))

    @staticmethod
    def focal_loss_with_init(N: int, alpha: float = -1) -> typing.Callable[[], None]:
        device = torch.device("cuda:0")
        inputs: torch.Tensor = logit(torch.rand(N)).to(device).requires_grad_()
        targets: torch.Tensor = (
            torch.randint(0, 2, (N,)).float().to(device).requires_grad_()
        )
        torch.cuda.synchronize()

        def run_focal_loss() -> None:
            fl = sigmoid_focal_loss(
                inputs, targets, gamma=0, alpha=alpha, reduction="mean"
            )
            fl.backward()
            torch.cuda.synchronize()

        return run_focal_loss

    @staticmethod
    def focal_loss_jit_with_init(
        N: int, alpha: float = -1
    ) -> typing.Callable[[], None]:
        device = torch.device("cuda:0")
        inputs: torch.Tensor = logit(torch.rand(N)).to(device).requires_grad_()
        targets: torch.Tensor = (
            torch.randint(0, 2, (N,)).float().to(device).requires_grad_()
        )
        torch.cuda.synchronize()

        def run_focal_loss_jit() -> None:
            fl = sigmoid_focal_loss_jit(
                inputs, targets, gamma=0, alpha=alpha, reduction="mean"
            )
            fl.backward()
            torch.cuda.synchronize()

        return run_focal_loss_jit


class TestFocalLossStar(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)

    def test_focal_loss_star_equals_ce_loss(self) -> None:
        """
        No weighting of easy/hard (gamma = 1) or positive/negative (alpha = -1).
        """
        inputs = logit(
            torch.tensor([[[0.95], [0.90], [0.98], [0.99]]], dtype=torch.float32)
        )
        targets = torch.tensor([[[1], [1], [1], [1]]], dtype=torch.float32)
        inputs_fl = inputs.clone().requires_grad_()
        targets_fl = targets.clone()
        inputs_ce = inputs.clone().requires_grad_()
        targets_ce = targets.clone()

        focal_loss_star = sigmoid_focal_loss_star(
            inputs_fl, targets_fl, gamma=1, alpha=-1, reduction="mean"
        )
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs_ce, targets_ce, reduction="mean"
        )

        self.assertEqual(ce_loss, focal_loss_star.data)
        focal_loss_star.backward()
        ce_loss.backward()
        # pyre-fixme[16]: Optional type has no attribute `data`.
        self.assertTrue(torch.allclose(inputs_fl.grad.data, inputs_ce.grad.data))

    def test_easy_ex_focal_loss_star_less_than_ce_loss(self) -> None:
        """
        With gamma = 3 loss of easy examples is downweighted.
        """
        inputs = logit(torch.tensor([0.75, 0.8, 0.12, 0.05], dtype=torch.float32))
        targets = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
        focal_loss_star = sigmoid_focal_loss_star(inputs, targets, gamma=3, alpha=-1)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss_ratio = (ce_loss / focal_loss_star).squeeze()
        self.assertTrue(torch.all(loss_ratio > 10.0))

    def test_focal_loss_star_positive_weights(self) -> None:
        """
        With alpha = 0.5 loss of positive examples is downweighted.
        """
        N = 5
        inputs = logit(torch.rand(N))
        targets = torch.ones((N,)).float()
        focal_loss_star = sigmoid_focal_loss_star(inputs, targets, gamma=2, alpha=-1)
        focal_loss_half = sigmoid_focal_loss_star(inputs, targets, gamma=2, alpha=0.5)
        loss_ratio = (focal_loss_star / focal_loss_half).squeeze()
        correct_ratio = torch.zeros((N,)).float() + 2.0
        self.assertTrue(np.allclose(loss_ratio, correct_ratio))

    def test_hard_ex_focal_loss_star_similar_to_ce_loss(self) -> None:
        """
        With gamma = 2 loss of hard examples is roughly unchanged.
        """
        inputs = logit(torch.tensor([0.05, 0.12, 0.91, 0.85], dtype=torch.float64))
        targets = torch.tensor([1, 1, 0, 0], dtype=torch.float64)
        focal_loss_star = sigmoid_focal_loss_star(inputs, targets, gamma=2, alpha=-1)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss_ratio = (ce_loss / focal_loss_star).squeeze()
        rough_ratio = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        self.assertTrue(torch.allclose(loss_ratio, rough_ratio, atol=0.1))

    def test_negatives_ignored_focal_loss_star(self) -> None:
        """
        With alpha = 1 negative examples have focal loss of 0.
        """
        inputs = logit(
            torch.tensor([[[0.05], [0.12], [0.89], [0.79]]], dtype=torch.float32)
        )
        targets = torch.tensor([[[1], [1], [0], [0]]], dtype=torch.float32)
        focal_loss_star = (
            sigmoid_focal_loss_star(inputs, targets, gamma=3, alpha=1).squeeze().numpy()
        )
        ce_loss = (
            F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            .squeeze()
            .numpy()
        )
        targets = targets.squeeze().numpy()
        self.assertTrue(np.all(ce_loss[targets == 0] > 0))
        self.assertTrue(np.all(focal_loss_star[targets == 0] == 0))

    def test_positives_ignored_focal_loss_star(self) -> None:
        """
        With alpha = 0 postive examples have focal loss of 0.
        """
        inputs = logit(
            torch.tensor([[[0.05], [0.12], [0.89], [0.79]]], dtype=torch.float32)
        )
        targets = torch.tensor([[[1], [1], [0], [0]]], dtype=torch.float32)
        focal_loss_star = (
            sigmoid_focal_loss_star(inputs, targets, gamma=3, alpha=0).squeeze().numpy()
        )
        ce_loss = (
            F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            .squeeze()
            .numpy()
        )
        targets = targets.squeeze().numpy()
        self.assertTrue(np.all(ce_loss[targets == 1] > 0))
        self.assertTrue(np.all(focal_loss_star[targets == 1] == 0))

    def test_mean_focal_loss_star_equals_ce_loss(self) -> None:
        """
        Mean value of focal loss across all examples matches ce loss.
        """
        inputs = logit(
            torch.tensor(
                [[0.05, 0.9], [0.52, 0.45], [0.89, 0.8], [0.39, 0.5]],
                dtype=torch.float32,
            )
        )
        targets = torch.tensor([[1, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)
        focal_loss_star = sigmoid_focal_loss_star(
            inputs, targets, gamma=1, alpha=-1, reduction="mean"
        )
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
        self.assertTrue(torch.allclose(ce_loss, focal_loss_star))

    def test_sum_focal_loss_star_equals_ce_loss(self) -> None:
        """
        Sum of focal loss across all examples matches ce loss.
        """
        inputs = logit(
            torch.tensor([[[0.05], [0.12], [0.89], [0.79]]], dtype=torch.float32)
        )
        targets = torch.tensor([[[1], [1], [0], [0]]], dtype=torch.float32)
        focal_loss_star = sigmoid_focal_loss_star(
            inputs, targets, gamma=1, alpha=-1, reduction="sum"
        )
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="sum")
        self.assertTrue(torch.allclose(ce_loss, focal_loss_star))

    # pyre-fixme[56]: Argument `not torch.cuda.is_available()` to decorator factory
    #  `unittest.skipIf` could not be resolved in a global scope.
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_focal_loss_star_equals_ce_loss_jit(self) -> None:
        """
        No weighting of easy/hard (gamma = 1) or positive/negative (alpha = 0).
        """
        device = torch.device("cuda:0")
        N = 5
        inputs = logit(torch.rand(N)).to(device)
        targets = torch.randint(0, 2, (N,)).float().to(device)
        focal_loss_star = sigmoid_focal_loss_star_jit(inputs, targets, gamma=1)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs.cpu(), targets.cpu(), reduction="none"
        )
        self.assertTrue(np.allclose(ce_loss, focal_loss_star.cpu()))

    @staticmethod
    def focal_loss_star_with_init(
        N: int, alpha: float = -1
    ) -> typing.Callable[[], None]:
        device = torch.device("cuda:0")
        inputs: torch.Tensor = logit(torch.rand(N)).to(device).requires_grad_()
        targets: torch.Tensor = (
            torch.randint(0, 2, (N,)).float().to(device).requires_grad_()
        )
        torch.cuda.synchronize()

        def run_focal_loss_star() -> None:
            fl = sigmoid_focal_loss_star(
                inputs, targets, gamma=1, alpha=alpha, reduction="mean"
            )
            fl.backward()
            torch.cuda.synchronize()

        return run_focal_loss_star

    @staticmethod
    def focal_loss_star_jit_with_init(
        N: int, alpha: float = -1
    ) -> typing.Callable[[], None]:
        device = torch.device("cuda:0")
        inputs: torch.Tensor = logit(torch.rand(N)).to(device).requires_grad_()
        targets: torch.Tensor = (
            torch.randint(0, 2, (N,)).float().to(device).requires_grad_()
        )
        torch.cuda.synchronize()

        def run_focal_loss_star_jit() -> None:
            fl = sigmoid_focal_loss_star_jit(
                inputs, targets, gamma=1, alpha=alpha, reduction="mean"
            )
            fl.backward()
            torch.cuda.synchronize()

        return run_focal_loss_star_jit
