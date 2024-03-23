"""
Utility functions module
"""

import logging
import pathlib
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager

import cv2
import keras
import numpy as np
import numpy.typing as npt

from fast_plate_ocr.config import (
    DEFAULT_IMG_HEIGHT,
    DEFAULT_IMG_WIDTH,
    MAX_PLATE_SLOTS,
    MODEL_ALPHABET,
    PAD_CHAR,
    VOCABULARY_SIZE,
)
from fast_plate_ocr.custom import cat_acc_metric, cce_loss, plate_acc_metric, top_3_k_metric


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


def read_plate_image(
    image_path: str, img_height: int = DEFAULT_IMG_HEIGHT, img_width: int = DEFAULT_IMG_WIDTH
) -> npt.NDArray:
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


def load_keras_model(
    model_path: pathlib.Path,
    vocab_size: int = VOCABULARY_SIZE,
    max_plate_slots: int = MAX_PLATE_SLOTS,
) -> keras.Model:
    """
    Utility helper function to load the keras OCR model.
    """
    custom_objects = {
        "cce": cce_loss(vocabulary_size=vocab_size),
        "cat_acc": cat_acc_metric(max_plate_slots=max_plate_slots, vocabulary_size=vocab_size),
        "plate_acc": plate_acc_metric(max_plate_slots=max_plate_slots, vocabulary_size=vocab_size),
        "top_3_k": top_3_k_metric(vocabulary_size=vocab_size),
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    return model


IMG_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
"""Valid image extensions for the scope of this script."""


def load_images_from_folder(
    img_dir: pathlib.Path,
    width: int = DEFAULT_IMG_WIDTH,
    height: int = DEFAULT_IMG_HEIGHT,
    shuffle: bool = False,
    limit: int | None = None,
) -> list[npt.NDArray]:
    """
    Return all images read from a directory. This uses the same read function used during training.
    """
    image_paths = sorted(
        str(f.resolve()) for f in img_dir.iterdir() if f.is_file() and f.suffix in IMG_EXTENSIONS
    )
    if limit:
        image_paths = image_paths[:limit]
    if shuffle:
        random.shuffle(image_paths)
    images = [read_plate_image(i, img_height=height, img_width=width) for i in image_paths]
    return images


@contextmanager
def log_time_taken(process_name: str) -> Iterator[None]:
    """A concise context manager to time code snippets and log the result."""
    time_start: float = time.perf_counter()
    try:
        yield
    finally:
        time_end: float = time.perf_counter()
        time_elapsed: float = time_end - time_start
        logging.info("Computation time of '%s' = %.3fms", process_name, 1000 * time_elapsed)


def postprocess_model_output(
    prediction: npt.NDArray,
    alphabet: str = MODEL_ALPHABET,
    max_plate_slots: int = MAX_PLATE_SLOTS,
    vocab_size: int = VOCABULARY_SIZE,
) -> tuple[str, npt.NDArray]:
    """
    Return plate text and confidence scores from raw model output.
    """
    prediction = prediction.reshape((max_plate_slots, vocab_size))
    probs = np.max(prediction, axis=-1)
    prediction = np.argmax(prediction, axis=-1)
    plate = "".join([alphabet[x] for x in prediction])
    return plate, probs
