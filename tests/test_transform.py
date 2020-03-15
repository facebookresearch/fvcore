#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import numpy as np
import unittest
from typing import Any, Tuple
import torch

from fvcore.transforms import transform as T
from fvcore.transforms.transform_util import to_float_tensor, to_numpy


# pyre-ignore-all-errors
class TestTransforms(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)

    def test_register(self):
        """
        Test register.
        """
        dtype = "int"

        def add1(t, x):
            return x + 1

        def flip_sub_width(t, x):
            return x - t.width

        T.Transform.register_type(dtype, add1)
        T.HFlipTransform.register_type(dtype, flip_sub_width)

        transforms = T.TransformList(
            [
                T.ScaleTransform(0, 0, 0, 0, 0),
                T.CropTransform(0, 0, 0, 0),
                T.HFlipTransform(3),
            ]
        )
        self.assertEqual(transforms.apply_int(3), 2)

        with self.assertRaises(AssertionError):
            T.HFlipTransform.register_type(dtype, lambda x: 1)

    @staticmethod
    def BlendTransform_img_gt(img, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying the blend transformation.
        Args:
            imgs (array): image(s) array before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            img (array): expected output array after apply the transformation.
            (list): expected shape of the output array.
        """
        src_image, src_weight, dst_weight = args
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = src_weight * src_image + dst_weight * img
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = src_weight * src_image + dst_weight * img
        return img, img.shape

    @staticmethod
    def CropTransform_img_gt(imgs, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying the crop transformation.
        Args:
            imgs (array): image(s) array before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            img (array): expected output array after apply the transformation.
            (list): expected shape of the output array.
        """
        x0, y0, w, h = args
        if len(imgs.shape) <= 3:
            ret = imgs[y0 : y0 + h, x0 : x0 + w]
        else:
            ret = imgs[..., y0 : y0 + h, x0 : x0 + w, :]
        return ret, ret.shape

    @staticmethod
    def GridSampleTransform_img_gt(imgs, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying the grid sampling transformation. Currently only dummy gt is
        prepared.
        Args:
            imgs (array): image(s) array before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            img (array): expected output array after apply the transformation.
            (list): expected shape of the output array.
        """
        return imgs, imgs.shape

    @staticmethod
    def VFlipTransform_img_gt(imgs, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying the vertical flip transformation.
        Args:
            imgs (array): image(s) array before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            img (array): expected output array after apply the transformation.
            (list): expected shape of the output array.
        """
        if len(imgs.shape) <= 3:
            # HxW or HxWxC.
            return imgs[::-1, :], imgs.shape
        else:
            # TxHxWxC.
            return imgs[:, ::-1, :], imgs.shape

    @staticmethod
    def HFlipTransform_img_gt(imgs, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying the horizontal flip transformation.
        Args:
            imgs (array): image(s) array before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            img (array): expected output array after apply the transformation.
            (list): expected shape of the output array.
        """
        if len(imgs.shape) <= 3:
            # HxW or HxWxC.
            return imgs[:, ::-1], imgs.shape
        else:
            # TxHxWxC.
            return imgs[:, :, ::-1], imgs.shape

    @staticmethod
    def NoOpTransform_img_gt(imgs, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying no transformation.
        Args:
            imgs (array): image(s) array before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            img (array): expected output array after apply the transformation.
            (list): expected shape of the output array.

        """
        return imgs, imgs.shape

    @staticmethod
    def ScaleTransform_img_gt(imgs, *args) -> Tuple[Any, Any]:
        """
        Given the input array, return the expected output array and shape after
        applying the resize transformation.
        Args:
            imgs (array): image(s) array before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            img (array): expected output array after apply the transformation.
                None means does not have expected output array for sanity check.
            (list): expected shape of the output array. None means does not have
                expected output shape for sanity check.
        """
        h, w, new_h, new_w, interp = args
        float_tensor = to_float_tensor(imgs)
        if interp == "nearest":
            if float_tensor.dim() == 3:
                float_tensor = torch._C._nn.upsample_nearest1d(
                    float_tensor, (new_h, new_w)
                )
            elif float_tensor.dim() == 4:
                float_tensor = torch._C._nn.upsample_nearest2d(
                    float_tensor, (new_h, new_w)
                )
            elif float_tensor.dim() == 5:
                float_tensor = torch._C._nn.upsample_nearest3d(
                    float_tensor, (new_h, new_w)
                )
            else:
                return None, None
        elif interp == "bilinear":
            if float_tensor.dim() == 4:
                float_tensor = torch._C._nn.upsample_bilinear2d(
                    float_tensor, (new_h, new_w), False
                )
            else:
                return None, None
        numpy_tensor = to_numpy(float_tensor, imgs.shape, imgs.dtype)
        return numpy_tensor, numpy_tensor.shape

    @staticmethod
    def _seg_provider(n: int = 8, h: int = 10, w: int = 10) -> np.ndarray:
        """
        Provide different segmentations as test cases.
        Args:
            n (int): number of points to generate in the image as segmentations.
            h, w (int): height and width dimensions.
        Returns:
            (np.ndarray): the segmentation to test on.
        """
        # Prepare random segmentation as test cases.
        for _ in range(n):
            yield np.random.randint(2, size=(h, w))

    @staticmethod
    def _img_provider(
        n: int = 8, c: int = 3, h: int = 10, w: int = 10
    ) -> Tuple[np.ndarray, type, str]:
        """
        Provide different image inputs as test cases.
        Args:
            n, c, h, w (int): batch, channel, height, and width dimensions.
        Returns:
            (np.ndarray): an image to test on.
            (type): type of the current array.
            (str): string to represent the shape. Options include `hw`, `hwc`,
                `nhwc`.
        """
        # Prepare mesh grid as test case.
        img_h_grid, img_w_grid = np.mgrid[0 : h * 2 : 2, 0 : w * 2 : 2]
        img_hw_grid = img_h_grid * w + img_w_grid
        img_hwc_grid = np.repeat(img_hw_grid[:, :, None], c, axis=2)
        img_nhwc_grid = np.repeat(img_hwc_grid[None, :, :, :], n, axis=0)
        for b in range(img_nhwc_grid.shape[0]):
            img_nhwc_grid[b] = img_nhwc_grid[b] + b

        # Prepare random array as test case.
        img_hw_random = np.random.rand(h, w)
        img_hwc_random = np.random.rand(h, w, c)
        img_nhwc_random = np.random.rand(n, h, w, c)

        for array_type, input_shape, init in itertools.product(
            [np.uint8, np.float32], ["hw", "hwc", "nhwc"], ["grid", "random"]
        ):
            yield locals()["img_{}_{}".format(input_shape, init)].astype(
                array_type
            ), array_type, input_shape

    def test_abstract(self):
        with self.assertRaises(TypeError):
            T.Transform()

    def test_blend_img_transforms(self):
        """
        Test BlendTransform.
        """
        _trans_name = "BlendTransform"
        blend_src_hw = np.ones((10, 10))
        blend_src_hwc = np.ones((10, 10, 3))
        blend_src_nhwc = np.ones((8, 10, 10, 3))

        for img, array_type, shape_str in TestTransforms._img_provider():
            blend_src = locals()["blend_src_{}".format(shape_str)].astype(
                array_type
            )
            params = (
                (blend_src, 0.0, 1.0),
                (blend_src, 0.3, 0.7),
                (blend_src, 0.5, 0.5),
                (blend_src, 0.7, 0.3),
                (blend_src, 1.0, 0.0),
            )
            for param in params:
                gt_transformer = getattr(self, "{}_img_gt".format(_trans_name))
                transformer = getattr(T, _trans_name)(*param)

                result = transformer.apply_image(img)
                img_gt, shape_gt = gt_transformer(img, *param)

                self.assertEqual(
                    shape_gt,
                    result.shape,
                    "transform {} failed to pass the shape check with"
                    "params {} given input with shape {} and type {}".format(
                        _trans_name, param, shape_str, array_type
                    ),
                )
                self.assertTrue(
                    np.allclose(result, img_gt),
                    "transform {} failed to pass the value check with"
                    "params {} given input with shape {} and type {}".format(
                        _trans_name, param, shape_str, array_type
                    ),
                )

    def test_crop_img_transforms(self):
        """
        Test CropTransform..
        """
        _trans_name = "CropTransform"
        params = (
            (0, 0, 0, 0),
            (0, 0, 1, 1),
            (0, 0, 6, 1),
            (0, 0, 1, 6),
            (0, 0, 6, 6),
            (1, 3, 6, 6),
            (3, 1, 6, 6),
            (3, 3, 6, 6),
            (6, 6, 6, 6),
        )
        for (img, array_type, shape_str), param in itertools.product(
            TestTransforms._img_provider(), params
        ):
            gt_transformer = getattr(self, "{}_img_gt".format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)

            result = transformer.apply_image(img)
            img_gt, shape_gt = gt_transformer(img, *param)

            self.assertEqual(
                shape_gt,
                result.shape,
                "transform {} failed to pass the shape check with"
                "params {} given input with shape {} and type {}".format(
                    _trans_name, param, shape_str, array_type
                ),
            )
            self.assertTrue(
                np.allclose(result, img_gt),
                "transform {} failed to pass the value check with"
                "params {} given input with shape {} and type {}".format(
                    _trans_name, param, shape_str, array_type
                ),
            )

    def test_vflip_img_transforms(self):
        """
        Test VFlipTransform..
        """
        _trans_name = "VFlipTransform"
        params = ((0,), (1,))

        for (img, array_type, shape_str), param in itertools.product(
            TestTransforms._img_provider(), params
        ):
            gt_transformer = getattr(self, "{}_img_gt".format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)

            result = transformer.apply_image(img)
            img_gt, shape_gt = gt_transformer(img, *param)

            self.assertEqual(
                shape_gt,
                result.shape,
                "transform {} failed to pass the shape check with"
                "params {} given input with shape {} and type {}".format(
                    _trans_name, param, shape_str, array_type
                ),
            )
            self.assertTrue(
                np.allclose(result, img_gt),
                "transform {} failed to pass the value check with"
                "params {} given input with shape {} and type {}.\n"
                "Output: {} -> {}".format(
                    _trans_name, param, shape_str, array_type, result, img_gt
                ),
            )

    def test_hflip_img_transforms(self):
        """
        Test HFlipTransform..
        """
        _trans_name = "HFlipTransform"
        params = ((0,), (1,))

        for (img, array_type, shape_str), param in itertools.product(
            TestTransforms._img_provider(), params
        ):
            gt_transformer = getattr(self, "{}_img_gt".format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)

            result = transformer.apply_image(img)
            img_gt, shape_gt = gt_transformer(img, *param)

            self.assertEqual(
                shape_gt,
                result.shape,
                "transform {} failed to pass the shape check with"
                "params {} given input with shape {} and type {}".format(
                    _trans_name, param, shape_str, array_type
                ),
            )
            self.assertTrue(
                np.allclose(result, img_gt),
                "transform {} failed to pass the value check with"
                "params {} given input with shape {} and type {}.\n"
                "Output: {} -> {}".format(
                    _trans_name, param, shape_str, array_type, result, img_gt
                ),
            )

    def test_no_op_img_transforms(self):
        """
        Test NoOpTransform..
        """
        _trans_name = "NoOpTransform"
        params = ()

        for (img, array_type, shape_str), param in itertools.product(
            TestTransforms._img_provider(), params
        ):
            gt_transformer = getattr(self, "{}_img_gt".format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)

            result = transformer.apply_image(img)
            img_gt, shape_gt = gt_transformer(img, *param)

            self.assertEqual(
                shape_gt,
                result.shape,
                "transform {} failed to pass the shape check with"
                "params {} given input with shape {} and type {}".format(
                    _trans_name, param, shape_str, array_type
                ),
            )
            self.assertTrue(
                np.allclose(result, img_gt),
                "transform {} failed to pass the value check with"
                "params {} given input with shape {} and type {}".format(
                    _trans_name, param, shape_str, array_type
                ),
            )

    def test_scale_img_transforms(self):
        """
        Test ScaleTransform.
        """
        _trans_name = "ScaleTransform"
        # Testing success cases.
        params = (
            (10, 20, 20, 20, "nearest"),
            (10, 20, 10, 20, "nearest"),
            (10, 20, 20, 10, "nearest"),
            (10, 20, 1, 1, "nearest"),
            (10, 20, 3, 3, "nearest"),
            (10, 20, 5, 10, "nearest"),
            (10, 20, 10, 5, "nearest"),
            (10, 20, 20, 20, "bilinear"),
            (10, 20, 10, 20, "bilinear"),
            (10, 20, 20, 10, "bilinear"),
            (10, 20, 1, 1, "bilinear"),
            (10, 20, 3, 3, "bilinear"),
            (10, 20, 5, 10, "bilinear"),
            (10, 20, 10, 5, "bilinear"),
        )

        for (img, array_type, shape_str), param in itertools.product(
            TestTransforms._img_provider(h=10, w=20), params
        ):
            gt_transformer = getattr(self, "{}_img_gt".format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)

            result = transformer.apply_image(img)
            img_gt, shape_gt = gt_transformer(img, *param)

            if shape_gt is not None:
                self.assertEqual(
                    shape_gt,
                    result.shape,
                    "transform {} failed to pass the shape check with"
                    "params {} given input with shape {} and type {}".format(
                        _trans_name, param, shape_str, array_type
                    ),
                )
            if img_gt is not None:
                self.assertTrue(
                    np.allclose(result, img_gt),
                    "transform {} failed to pass the value check with"
                    "params {} given input with shape {} and type {}".format(
                        _trans_name, param, shape_str, array_type
                    ),
                )

        # Testing failure cases.
        params = (
            (0, 0, 20, 20, "nearest"),
            (0, 0, 0, 0, "nearest"),
            (-1, 0, 0, 0, "nearest"),
            (0, -1, 0, 0, "nearest"),
            (0, 0, -1, 0, "nearest"),
            (0, 0, 0, -1, "nearest"),
            (20, 10, 0, -1, "nearest"),
        )

        for (img, _, _), param in itertools.product(
            TestTransforms._img_provider(h=10, w=20), params
        ):
            gt_transformer = getattr(self, "{}_img_gt".format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)
            with self.assertRaises((RuntimeError, AssertionError)):
                result = transformer.apply_image(img)

    def test_grid_sample_img_transform(self):
        """
        Test grid sampling tranformation.
        """
        # TODO: add more complex test case for grid sample.
        for interp in ["nearest"]:
            grid_2d = np.stack(
                np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)),
                axis=2,
            ).astype(np.float)
            grid = np.tile(grid_2d[None, :, :, :], [8, 1, 1, 1])
            transformer = T.GridSampleTransform(grid, interp)

            img_h, img_w = np.mgrid[0:10:1, 0:10:1].astype(np.float)
            img_hw = img_h * 10 + img_w
            img_hwc = np.repeat(img_hw[:, :, None], 3, axis=2)
            img_bhwc = np.repeat(img_hwc[None, :, :, :], 8, axis=0)

            result = transformer.apply_image(img_bhwc)
            img_gt, shape_gt = TestTransforms.GridSampleTransform_img_gt(
                img_bhwc, *(grid_2d, interp)
            )

            self.assertEqual(shape_gt, result.shape)
            self.assertTrue(np.allclose(result, img_gt))

    def test_crop_polygons(self):
        # Ensure that shapely produce an extra vertex at the end
        # This is assumed when copping polygons
        try:
            import shapely.geometry as geometry
        except ImportError:
            return

        polygon = np.asarray([3, 3.5, 11, 10.0, 38, 98, 15.0, 100.0]).reshape(
            -1, 2
        )
        g = geometry.Polygon(polygon)
        coords = np.asarray(g.exterior.coords)
        self.assertEqual(coords[0].tolist(), coords[-1].tolist())
