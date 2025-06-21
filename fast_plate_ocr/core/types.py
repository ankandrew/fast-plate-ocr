"""
Common custom types used across the lib.
"""

import os
from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np
from numpy import typing as npt

ImageInterpolation: TypeAlias = Literal["nearest", "linear", "cubic", "area", "lanczos4"]
"""Interpolation method used for resizing the input image."""
ImageColorMode: TypeAlias = Literal["grayscale", "rgb"]
"""Input image color mode. Use 'grayscale' for single-channel input or 'rgb' for 3-channel input."""
PaddingColor: TypeAlias = tuple[int, int, int] | int
"""Padding colour for letterboxing (only used when keeping image aspect ratio)."""
PathLike: TypeAlias = str | os.PathLike
"""Path-like objects."""
ImgLike: TypeAlias = PathLike | npt.NDArray[np.uint8]
"""Image-like objects, including paths to image files and NumPy arrays of images."""
BatchOrImgLike: TypeAlias = ImgLike | Sequence[ImgLike]
"""
Image-like objects, including paths to image files and NumPy arrays of images, or a batch of images.
"""
BatchArray: TypeAlias = npt.NDArray[np.uint8]
"""Numpy array of images, representing a batch of images."""
TensorDataFormat: TypeAlias = Literal["channels_last", "channels_first"]
"""
Data format of the input tensor. It can be either 'channels_last' or 'channels_first'.
'channels_last' corresponds to inputs with shape (batch, height, width, channels), while
'channels_first' corresponds to inputs with shape (batch, channels, height, width).
"""
KerasDtypes: TypeAlias = Literal[
    "float16",
    "float32",
    "float64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "bfloat16",
    "bool",
    "string",
    "float8_e4m3fn",
    "float8_e5m2",
    "complex64",
    "complex128",
]
"""
Keras data types supported by the library.
"""
