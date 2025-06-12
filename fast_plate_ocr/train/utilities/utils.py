"""
Utility functions module
"""

import logging
import pathlib
import pkgutil
import random
from collections.abc import Iterator
from importlib import import_module

import cv2
import keras
import numpy as np
import numpy.typing as npt

from fast_plate_ocr.core.process import read_and_resize_plate_image
from fast_plate_ocr.core.types import ImageColorMode, ImageInterpolation, PaddingColor
from fast_plate_ocr.train.model.config import PlateOCRConfig
from fast_plate_ocr.train.model.loss import cce_loss, focal_cce_loss
from fast_plate_ocr.train.model.metric import (
    cat_acc_metric,
    plate_acc_metric,
    plate_len_acc_metric,
    top_3_k_metric,
)


def one_hot_plate(plate: str, alphabet: str) -> list[list[int]]:
    return [[0 if char != letter else 1 for char in alphabet] for letter in plate]


def target_transform(
    plate_text: str,
    max_plate_slots: int,
    alphabet: str,
    pad_char: str,
) -> npt.NDArray[np.uint8]:
    # Pad the plates which length is smaller than 'max_plate_slots'
    plate_text = plate_text.ljust(max_plate_slots, pad_char)
    # Generate numpy arrays with one-hot encoding of plates
    encoded_plate = np.array(one_hot_plate(plate_text, alphabet=alphabet), dtype=np.uint8)
    return encoded_plate


def _register_custom_keras():
    base_pkg = "fast_plate_ocr.train.model"
    for _, name, _ in pkgutil.walk_packages(
        import_module(base_pkg).__path__, prefix=f"{base_pkg}."
    ):
        if any(m in name for m in ("layers",)):
            import_module(name)


def load_keras_model(
    model_path: str | pathlib.Path,
    plate_config: PlateOCRConfig,
) -> keras.Model:
    """
    Utility helper function to load the keras OCR model.
    """
    _register_custom_keras()
    custom_objects = {
        "cce": cce_loss(
            vocabulary_size=plate_config.vocabulary_size,
        ),
        "focal_cce": focal_cce_loss(
            vocabulary_size=plate_config.vocabulary_size,
        ),
        "cat_acc": cat_acc_metric(
            max_plate_slots=plate_config.max_plate_slots,
            vocabulary_size=plate_config.vocabulary_size,
        ),
        "plate_acc": plate_acc_metric(
            max_plate_slots=plate_config.max_plate_slots,
            vocabulary_size=plate_config.vocabulary_size,
        ),
        "top_3_k": top_3_k_metric(
            vocabulary_size=plate_config.vocabulary_size,
        ),
        "plate_len_acc": plate_len_acc_metric(
            max_plate_slots=plate_config.max_plate_slots,
            vocabulary_size=plate_config.vocabulary_size,
            pad_token_index=plate_config.pad_idx,
        ),
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    return model


IMG_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
"""Valid image extensions for the scope of this script."""


def load_images_from_folder(  # noqa: PLR0913
    img_dir: pathlib.Path,
    width: int,
    height: int,
    image_color_mode: ImageColorMode = "grayscale",
    keep_aspect_ratio: bool = False,
    interpolation_method: ImageInterpolation = "linear",
    padding_color: PaddingColor = (114, 114, 114),
    shuffle: bool = False,
    limit: int | None = None,
) -> Iterator[npt.NDArray]:
    """
    Return all images read from a directory. This uses the same read function used during training.
    """
    # pylint: disable=too-many-arguments
    image_paths = sorted(
        str(f.resolve()) for f in img_dir.iterdir() if f.is_file() and f.suffix in IMG_EXTENSIONS
    )
    if limit:
        image_paths = image_paths[:limit]
    if shuffle:
        random.shuffle(image_paths)
    yield from (
        read_and_resize_plate_image(
            i,
            img_height=height,
            img_width=width,
            image_color_mode=image_color_mode,
            keep_aspect_ratio=keep_aspect_ratio,
            interpolation_method=interpolation_method,
            padding_color=padding_color,
        )
        for i in image_paths
    )


def postprocess_model_output(
    prediction: npt.NDArray,
    alphabet: str,
    max_plate_slots: int,
    vocab_size: int,
) -> tuple[str, npt.NDArray]:
    """
    Return plate text and confidence scores from raw model output.
    """
    prediction = prediction.reshape((max_plate_slots, vocab_size))
    probs = np.max(prediction, axis=-1)
    prediction = np.argmax(prediction, axis=-1)
    plate = "".join([alphabet[x] for x in prediction])
    return plate, probs


def low_confidence_positions(probs, thresh=0.3) -> npt.NDArray:
    """Returns indices of elements in `probs` less than `thresh`, indicating low confidence."""
    return np.where(np.array(probs) < thresh)[0]


def display_predictions(
    image: npt.NDArray,
    plate: str,
    probs: npt.NDArray,
    low_conf_thresh: float,
) -> None:
    """
    Display plate and corresponding prediction.
    """
    plate_str = "".join(plate)
    logging.info("Plate: %s", plate_str)
    logging.info("Confidence: %s", probs)
    image_to_show = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    if len(image_to_show.shape) == 2:
        image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_GRAY2RGB)
    elif image_to_show.shape[2] == 3:
        image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_BGR2RGB)
    # Average probabilities
    avg_prob = np.mean(probs) * 100
    cv2.putText(
        image_to_show,
        f"{plate_str}  {avg_prob:.{2}f}%",
        org=(5, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 0),
        lineType=1,
        thickness=6,
    )
    cv2.putText(
        image_to_show,
        f"{plate_str}  {avg_prob:.{2}f}%",
        org=(5, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        lineType=1,
        thickness=2,
    )
    # Display character with low confidence
    low_conf_chars = "Low conf. on: " + " ".join(
        [plate[i] for i in low_confidence_positions(probs, thresh=low_conf_thresh)]
    )
    cv2.putText(
        image_to_show,
        low_conf_chars,
        org=(5, 200),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=(0, 0, 220),
        lineType=1,
        thickness=2,
    )
    cv2.imshow("plates", image_to_show)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        return
