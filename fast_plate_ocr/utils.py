"""
Utility functions module
"""

import os

import numpy as np
import numpy.typing as npt
import pandas as pd

from fast_plate_ocr.config import MAX_PLATE_SLOTS, MODEL_ALPHABET, PAD_CHAR
from fast_plate_ocr.custom_types import Framework


def one_hot_plate(plate: str, alphabet: str = MODEL_ALPHABET) -> list[list[int]]:
    return [[0 if char != letter else 1 for char in alphabet] for letter in plate]


def one_hot_plates(plates: list[str], alphabet: str = MODEL_ALPHABET) -> npt.NDArray[np.uint8]:
    # Convert the list of plates into a 2D NumPy array of characters
    plates_array = np.array([list(plate) for plate in plates])
    # Convert the alphabet string into a 1D NumPy array of characters
    alphabet_array = np.array(list(alphabet))
    # Find indices of each character in plates_array within alphabet_array
    char_matches = plates_array[:, :, np.newaxis] == alphabet_array
    # Convert boolean matches to integers and sum across the alphabet dimension.
    char_indices = char_matches.astype(np.uint8).argmax(axis=2)
    # Initialize a 3D array for the one-hot encoding: (num_plates, plate_length, alphabet_length)
    one_hot_encoded = np.zeros_like(char_matches, dtype=np.uint8)
    # Fill the one-hot encoded array using advanced indexing
    one_hot_encoded[
        np.arange(char_indices.shape[0])[:, np.newaxis],
        np.arange(char_indices.shape[1]),
        char_indices,
    ] = 1
    return one_hot_encoded


def target_transform(
    plate_text: pd.Series,
    max_plate_slots: int = MAX_PLATE_SLOTS,
    alphabet: str = MODEL_ALPHABET,
    pad_char: str = PAD_CHAR,
) -> npt.NDArray[np.uint8]:
    # Pad the plates which length is smaller than 'max_plate_slots'
    plate_text = plate_text.str.ljust(max_plate_slots, pad_char)
    # Generate numpy arrays with one-hot encoding of plates
    encoded_plates = one_hot_plates(plate_text.tolist(), alphabet=alphabet)
    return encoded_plates


def set_tensorflow_backend() -> None:
    set_keras_backend("tensorflow")


def set_jax_backend() -> None:
    set_keras_backend("jax")


def set_pytorch_backend() -> None:
    set_keras_backend("torch")


def set_keras_backend(framework: Framework) -> None:
    os.environ["KERAS_BACKEND"] = framework
