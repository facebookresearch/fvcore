# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import unittest

from fvcore.transforms.transform_util import to_float_tensor, to_numpy


class TestTransformUtil(unittest.TestCase):
    def test_convert(self) -> None:
        N, C, H, W = 4, 64, 14, 14
        np.random.seed(0)
        array_HW: np.ndarray = np.random.rand(H, W)
        array_HWC: np.ndarray = np.random.rand(H, W, C)
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
            converted_array = to_numpy(
                converted_tensor, array.shape, array.dtype
            )
            self.assertTrue(np.allclose(array, converted_array))
