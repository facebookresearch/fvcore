# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import inspect
from abc import ABCMeta, abstractmethod
from typing import Callable, TypeVar

import numpy as np
import torch

from .transform_util import to_float_tensor, to_numpy


__all__ = [
    "BlendTransform",
    "CropTransform",
    "GridSampleTransform",
    "HFlipTransform",
    "VFlipTransform",
    "NoOpTransform",
    "ScaleTransform",
    "Transform",
    "TransformList",
]


class Transform(metaclass=ABCMeta):
    """
    Base class for implementations of __deterministic__ transformations for
    image and other data structures. "Deterministic" requires that the output of
    all methods of this class are deterministic w.r.t their input arguments. In
    training, there should be a higher-level policy that generates (likely with
    random variations) these transform ops. Each transform op may handle several
    data types, e.g.: image, coordinates, segmentation, bounding boxes. Some of
    them have a default implementation, but can be overwritten if the default
    isn't appropriate. The implementation of each method may choose to modify
    its input data in-place for efficient transformation.
    """

    def _set_attributes(self, params: list = None):
        """
        Set attributes from the input list of parameters.

        Args:
            params (list): list of parameters.
        """

        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def apply_image(self, img: np.ndarray):
        """
        Apply the transform on an image.

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: image after apply the transformation.
        """

    @abstractmethod
    def apply_coords(self, coords: np.ndarray):
        """
        Apply the transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            ndarray: coordinates after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates inside an image of
            shape (H, W) are in range [0, W] or [0, H].
            This function should correctly transform coordinates outside the image as well.
        """

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply the transform on a full-image segmentation.
        By default will just perform "apply_image".

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
            or bool dtype.

        Returns:
            ndarray: segmentation after apply the transformation.
        """
        return self.apply_image(segmentation)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        Apply the transform on an axis-aligned box.
        By default will transform the corner points and use their
        minimum/maximum to create a new axis-aligned box.
        Note that this default may change the size of your box, e.g. in
        rotations.

        Args:
            box (ndarray): Nx4 floating point array of XYXY format in absolute
                coordinates.
        Returns:
            ndarray: box after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates inside an image of
            shape (H, W) are in range [0, W] or [0, H].

            This function does not clip boxes to force them inside the image.
            It is up to the application that uses the boxes to decide.
        """
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)
        return trans_boxes

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply the transform on a list of polygons, each represented by a Nx2
        array.
        By default will just transform all the points.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            list[ndarray]: polygon after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        return [self.apply_coords(p) for p in polygons]

    @classmethod
    def register_type(cls, data_type: str, func: Callable):
        """
        Register the given function as a handler that this transform will use
        for a specific data type.

        Args:
            data_type (str): the name of the data type (e.g., box)
            func (callable): takes a transform and a data, returns the
                transformed data.

        Examples:

        .. code-block:: python

            def func(flip_transform, voxel_data):
                return transformed_voxel_data
            HFlipTransform.register_type("voxel", func)

            # ...
            transform = HFlipTransform(...)
            transform.apply_voxel(voxel_data)  # func will be called
        """
        assert callable(
            func
        ), "You can only register a callable to a Transform. Got {} instead.".format(
            func
        )
        argspec = inspect.getfullargspec(func)
        assert len(argspec.args) == 2, (
            "You can only register a function that takes two positional "
            "arguments to a Transform! Got a function with spec {}".format(str(argspec))
        )
        setattr(cls, "apply_" + data_type, func)


_T = TypeVar("_T")


# pyre-ignore-all-errors
class TransformList:
    """
    Maintain a list of transform operations which will be applied in sequence.
    Attributes:
        transforms (list[Transform])
    """

    def __init__(self, transforms: list):
        """
        Args:
            transforms (list[Transform]): list of transforms to perform.
        """
        super().__init__()
        for t in transforms:
            assert isinstance(t, Transform), t
        self.transforms = transforms

    def _apply(self, x: _T, meth: str) -> _T:
        """
        Apply the transforms on the input.
        Args:
            x: input to apply the transform operations.
            meth (str): meth.
        Returns:
            x: after apply the transformation.
        """
        for t in self.transforms:
            x = getattr(t, meth)(x)
        return x

    def __getattr__(self, name: str):
        """
        Args:
            name (str): name of the attribute.
        """
        if name.startswith("apply_"):
            return lambda x: self._apply(x, name)
        raise AttributeError("TransformList object has no attribute {}".format(name))

    def __add__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(self.transforms + others)

    def __iadd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        self.transforms.extend(others)
        return self

    def __radd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(others + self.transforms)

    def __len__(self) -> int:
        """
        Returns:
            Number of transforms contained in the TransformList.
        """
        return len(self.transforms)


class HFlipTransform(Transform):
    """
    Perform horizontal flip.
    """

    def __init__(self, width: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        # NOTE: opencv would be faster:
        # https://github.com/pytorch/pytorch/issues/16424#issuecomment-580695672
        if img.ndim <= 3:  # HxW, HxWxC
            return np.flip(img, axis=1)
        else:
            return np.flip(img, axis=-2)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        coords[:, 0] = self.width - coords[:, 0]
        return coords


class VFlipTransform(Transform):
    """
    Perform vertical flip.
    """

    def __init__(self, height: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        tensor = torch.from_numpy(np.ascontiguousarray(img))
        if len(tensor.shape) == 2:
            # For dimension of HxW.
            tensor = tensor.flip((-2))
        elif len(tensor.shape) > 2:
            # For dimension of HxWxC, NxHxWxC.
            tensor = tensor.flip((-3))
        return tensor.numpy()

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        coords[:, 1] = self.height - coords[:, 1]
        return coords


class NoOpTransform(Transform):
    """
    A transform that does nothing.
    """

    def __init__(self):
        super().__init__()

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords


class ScaleTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h: int, w: int, new_h: int, new_w: int, interp: str = None):
        """
        Args:
            h, w (int): original image size.
            new_h, new_w (int): new image size.
            interp (str): interpolation methods. Options includes `nearest`, `linear`
                (3D-only), `bilinear`, `bicubic` (4D-only), and `area`.
                Details can be found in:
                https://pytorch.org/docs/stable/nn.functional.html
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Resize the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): interpolation methods. Options includes `nearest`, `linear`
                (3D-only), `bilinear`, `bicubic` (4D-only), and `area`.
                Details can be found in:
                https://pytorch.org/docs/stable/nn.functional.html

        Returns:
            ndarray: resized image(s).
        """
        if len(img.shape) == 4:
            h, w = img.shape[1:3]
        elif len(img.shape) in (2, 3):
            h, w = img.shape[:2]
        else:
            raise ("Unsupported input with shape of {}".format(img.shape))
        assert (
            self.h == h and self.w == w
        ), "Input size mismatch h w {}:{} -> {}:{}".format(self.h, self.w, h, w)
        interp_method = interp if interp is not None else self.interp
        # Option of align_corners is only supported for linear, bilinear,
        # and bicubic.
        if interp_method in ["linear", "bilinear", "bicubic"]:
            align_corners = False
        else:
            align_corners = None

        # note: this is quite slow for int8 images because torch does not
        # support it https://github.com/pytorch/pytorch/issues/5580
        float_tensor = torch.nn.functional.interpolate(
            to_float_tensor(img),
            size=(self.new_h, self.new_w),
            mode=interp_method,
            align_corners=align_corners,
        )
        return to_numpy(float_tensor, img.shape, img.dtype)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute the coordinates after resize.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: resized coordinates.
        """
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply resize on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: resized segmentation.
        """
        segmentation = self.apply_image(segmentation, interp="nearest")
        return segmentation


class GridSampleTransform(Transform):
    def __init__(self, grid: np.ndarray, interp: str):
        """
        Args:
            grid (ndarray): grid has x and y input pixel locations which are
                used to compute output. Grid has values in the range of [-1, 1],
                which is normalized by the input height and width. The dimension
                is `N x H x W x 2`.
            interp (str): interpolation methods. Options include `nearest` and
                `bilinear`.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply grid sampling on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): interpolation methods. Options include `nearest` and
                `bilinear`.
        Returns:
            ndarray: grid sampled image(s).
        """
        interp_method = interp if interp is not None else self.interp
        float_tensor = torch.nn.functional.grid_sample(
            to_float_tensor(img),  # NxHxWxC -> NxCxHxW.
            torch.from_numpy(self.grid),
            mode=interp_method,
            padding_mode="border",
            align_corners=False,
        )
        return to_numpy(float_tensor, img.shape, img.dtype)

    def apply_coords(self, coords: np.ndarray):
        """
        Not supported.
        """
        raise NotImplementedError()

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply grid sampling on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: grid sampled segmentation.
        """
        segmentation = self.apply_image(segmentation, interp="nearest")
        return segmentation


class CropTransform(Transform):
    def __init__(self, x0: int, y0: int, w: int, h: int):
        # TODO: flip the order of w and h.
        """
        Args:
            x0, y0, w, h (int): crop the image(s) by img[y0:y0+h, x0:x0+w].
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: cropped image(s).
        """
        if len(img.shape) <= 3:
            return img[self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w]
        else:
            return img[..., self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w, :]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: cropped coordinates.
        """
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply crop transform on a list of polygons, each represented by a Nx2 array.
        It will crop the polygon with the box, therefore the number of points in the
        polygon might change.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            ndarray: cropped polygons.
        """
        import shapely.geometry as geometry

        # Create a window that will be used to crop
        crop_box = geometry.box(
            self.x0, self.y0, self.x0 + self.w, self.y0 + self.h
        ).buffer(0.0)

        cropped_polygons = []

        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0.0)
            # polygon must be valid to perform intersection.
            assert polygon.is_valid, polygon
            cropped = polygon.intersection(crop_box)
            if cropped.is_empty:
                continue
            if not isinstance(cropped, geometry.collection.BaseMultipartGeometry):
                cropped = [cropped]
            # one polygon may be cropped to multiple ones
            for poly in cropped:
                # It could produce lower dimensional objects like lines or
                # points, which we want to ignore
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                    continue
                coords = np.asarray(poly.exterior.coords)
                # NOTE This process will produce an extra identical vertex at
                # the end. So we remove it. This is tested by
                # `tests/test_data_transform.py`
                cropped_polygons.append(coords[:-1])
        return [self.apply_coords(p) for p in cropped_polygons]


class BlendTransform(Transform):
    """
    Transforms pixel colors with PIL enhance functions.
    """

    def __init__(self, src_image: np.ndarray, src_weight: float, dst_weight: float):
        """
        Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``

        Args:
            src_image (ndarray): Input image is blended with this image
            src_weight (float): Blend weighting of src_image
            dst_weight (float): Blend weighting of dst_image
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply blend transform on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency, perform blend would not
                require interpolation.
        Returns:
            ndarray: blended image(s).
        """
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = self.src_weight * self.src_image + self.dst_weight * img
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return self.src_weight * self.src_image + self.dst_weight * img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation
