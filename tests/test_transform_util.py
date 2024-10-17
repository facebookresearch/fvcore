# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# pyre-strict

import unittest

import numpy as np
from fvcore.transforms.transform_util import to_float_tensor, to_numpy


class TestTransformUtil(unittest.TestCase):
    def test_convert(self) -> None:
        N, C, H, W = 4, 64, 14, 14
        np.random.seed(0)
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        array_HW: np.ndarray = np.random.rand(H, W)
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        array_HWC: np.ndarray = np.random.rand(H, W, C)
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        array_NHWC: np.ndarray = np.random.rand(N, H, W, C)
        arrays = [
            array_HW,
            (array_HW * 255).astype(np.uint8),
            array_HWC,
            (array_HWC * 255).astype(np.uint8),
            array_NHWC,
            (array_NHWC * 255).astype(np.uint8),
        ]

        for array in arrays:
            converted_tensor = to_float_tensor(array)
            # pyre-fixme[6]: For 2nd argument expected `List[Any]` but got
            #  `tuple[int, ...]`.
            converted_array = to_numpy(converted_tensor, array.shape, array.dtype)
            self.assertTrue(np.allclose(array, converted_array))
