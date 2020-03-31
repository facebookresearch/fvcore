# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fvcore.common.benchmark import benchmark
from test_common import TestHistoryBuffer


def bm_history_buffer_update() -> None:
    kwargs_list = [
        {"num_values": 100},
        {"num_values": 1000},
        {"num_values": 10000},
        {"num_values": 100000},
        {"num_values": 1000000},
    ]
    benchmark(
        TestHistoryBuffer.create_buffer_with_init,
        "BM_UPDATE",
        kwargs_list,
        warmup_iters=1,
    )
