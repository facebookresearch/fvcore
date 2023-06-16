# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import numpy as np
import torch

from fvcore.nn import giou_loss


class TestGIoULoss(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)

    def test_giou_loss(self) -> None:
        # Identical boxes should have loss of 0
        box = torch.tensor([-1, -1, 1, 1], dtype=torch.float32)
        loss = giou_loss(box, box)
        self.assertTrue(np.allclose(loss, [0.0]))

        # quarter size box inside other box = IoU of 0.25
        box2 = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        loss = giou_loss(box, box2)
        self.assertTrue(np.allclose(loss, [0.75]))

        # Two side by side boxes, area=union
        # IoU=0 and GIoU=0 (loss 1.0)
        box3 = torch.tensor([0, 1, 1, 2], dtype=torch.float32)
        loss = giou_loss(box2, box3)
        self.assertTrue(np.allclose(loss, [1.0]))

        # Two diagonally adjacent boxes, area=2*union
        # IoU=0 and GIoU=-0.5 (loss 1.5)
        box4 = torch.tensor([1, 1, 2, 2], dtype=torch.float32)
        loss = giou_loss(box2, box4)
        self.assertTrue(np.allclose(loss, [1.5]))

        # Test batched loss and reductions
        box1s = torch.stack([box2, box2], dim=0)
        box2s = torch.stack([box3, box4], dim=0)

        loss = giou_loss(box1s, box2s, reduction="sum")
        self.assertTrue(np.allclose(loss, [2.5]))

        loss = giou_loss(box1s, box2s, reduction="mean")
        self.assertTrue(np.allclose(loss, [1.25]))

    def test_empty_inputs(self) -> None:
        box1 = torch.randn([0, 4], dtype=torch.float32).requires_grad_()
        box2 = torch.randn([0, 4], dtype=torch.float32).requires_grad_()
        loss = giou_loss(box1, box2, reduction="mean")
        loss.backward()

        self.assertEqual(loss.detach().numpy(), 0.0)
        self.assertIsNotNone(box1.grad)
        self.assertIsNotNone(box2.grad)

        loss = giou_loss(box1, box2, reduction="none")
        self.assertEqual(loss.numel(), 0)
