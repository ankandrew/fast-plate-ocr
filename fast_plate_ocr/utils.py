"""
Utility functions module
"""

import os

import cv2
import numpy as np
import numpy.typing as npt

from fast_plate_ocr.config import MAX_PLATE_SLOTS, MODEL_ALPHABET, PAD_CHAR
from fast_plate_ocr.custom_types import Framework


def one_hot_plate(plate: str, alphabet: str = MODEL_ALPHABET) -> list[list[int]]:
    return [[0 if char != letter else 1 for char in alphabet] for letter in plate]


def target_transform(
    plate_text: str,
    max_plate_slots: int = MAX_PLATE_SLOTS,
    alphabet: str = MODEL_ALPHABET,
    pad_char: str = PAD_CHAR,
) -> npt.NDArray[np.uint8]:
    # Pad the plates which length is smaller than 'max_plate_slots'
    plate_text = plate_text.ljust(max_plate_slots, pad_char)
    # Generate numpy arrays with one-hot encoding of plates
    encoded_plate = np.array(one_hot_plate(plate_text, alphabet=alphabet), dtype=np.uint8)
    return encoded_plate


def set_tensorflow_backend() -> None:
    """Set Keras backend to tensorflow."""
    set_keras_backend("tensorflow")


def set_jax_backend() -> None:
    """Set Keras backend to jax."""
    set_keras_backend("jax")


def set_pytorch_backend() -> None:
    """Set Keras backend to pytorch."""
    set_keras_backend("torch")


def set_keras_backend(framework: Framework) -> None:
    """Set the Keras backend to a given framework."""
    os.environ["KERAS_BACKEND"] = framework


def read_plate_image(image_path: str, img_height: int, img_width: int) -> npt.NDArray:
    """
    Read and resize a license plate image.

    :param str image_path: The path to the license plate image.
    :param int img_height: The desired height of the resized image.
    :param int img_width: The desired width of the resized image.
    :return: The resized license plate image as a NumPy array.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    img = np.expand_dims(img, -1)
    return img
