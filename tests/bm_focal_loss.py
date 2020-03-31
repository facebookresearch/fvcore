# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from fvcore.common.benchmark import benchmark
from test_focal_loss import TestFocalLoss, TestFocalLossStar


def bm_focal_loss() -> None:
    if not torch.cuda.is_available():
        print("Skipped: CUDA unavailable")
        return

    kwargs_list = [
        {"N": 100},
        {"N": 100, "alpha": 0},
        {"N": 1000},
        {"N": 1000, "alpha": 0},
        {"N": 10000},
        {"N": 10000, "alpha": 0},
    ]
    benchmark(
        TestFocalLoss.focal_loss_with_init, "Focal_loss", kwargs_list, warmup_iters=1
    )
    benchmark(
        TestFocalLoss.focal_loss_jit_with_init,
        "Focal_loss_JIT",
        kwargs_list,
        warmup_iters=1,
    )


def bm_focal_loss_star() -> None:
    if not torch.cuda.is_available():
        print("Skipped: CUDA unavailable")
        return

    kwargs_list = [
        {"N": 100},
        {"N": 100, "alpha": 0},
        {"N": 1000},
        {"N": 1000, "alpha": 0},
        {"N": 10000},
        {"N": 10000, "alpha": 0},
    ]
    benchmark(
        TestFocalLossStar.focal_loss_star_with_init,
        "Focal_loss_star",
        kwargs_list,
        warmup_iters=1,
    )
    benchmark(
        TestFocalLossStar.focal_loss_star_jit_with_init,
        "Focal_loss_star_JIT",
        kwargs_list,
        warmup_iters=1,
    )
