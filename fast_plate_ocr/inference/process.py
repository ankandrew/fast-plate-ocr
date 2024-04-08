"""
Utility functions for processing model input/output.
"""

import os

import cv2
import numpy as np
import numpy.typing as npt


def read_plate_image(image_path: str) -> npt.NDArray:
    """
    Read image from disk as a grayscale image.

    :param image_path: The path to the license plate image.
    :return: The image as a NumPy array.
    """
    if not os.path.exists(image_path):
        raise ValueError(f"{image_path} file doesn't exist!")
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def preprocess_image(
    image: npt.NDArray | list[npt.NDArray], img_height: int, img_width: int
) -> npt.NDArray:
    """
    Preprocess the image(s), so they're ready to be fed to the model.

    Note: We don't normalize the pixel values between [0, 1] here, because that the model first
    layer does that.

    :param image: The image(s) contained in a NumPy array.
    :param img_height: The desired height of the resized image.
    :param img_width: The desired width of the resized image.
    :return: A numpy array with shape (N, H, W, 1).
    """
    # Add batch dimension: (H, W) -> (1, H, W)
    if isinstance(image, np.ndarray):
        image = np.expand_dims(image, axis=0)

    imgs = np.array(
        [
            cv2.resize(im.squeeze(), (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            for im in image
        ]
    )
    # Add channel dimension
    imgs = np.expand_dims(imgs, axis=-1)
    return imgs


def postprocess_output(
    model_output: npt.NDArray,
    max_plate_slots: int,
    model_alphabet: str,
    return_confidence: bool = False,
) -> tuple[list[str], npt.NDArray] | list[str]:
    """
    Post-processes model output and return license plate string, and optionally the probabilities.

    :param model_output: Output from the model containing predictions.
    :param max_plate_slots: Maximum number of characters in a license plate.
    :param model_alphabet: Alphabet used by the model for character encoding.
    :param return_confidence: Flag to indicate whether to return confidence scores along with plate
     predictions.
    :return: Decoded license plate characters as a list, optionally with confidence scores. The
     confidence scores have shape (N, max_plate_slots) where N is the batch size.
    """
    predictions = model_output.reshape((-1, max_plate_slots, len(model_alphabet)))
    prediction_indices = np.argmax(predictions, axis=-1)
    alphabet_array = np.array(list(model_alphabet))
    plate_chars = alphabet_array[prediction_indices]
    plates: list[str] = np.apply_along_axis("".join, 1, plate_chars).tolist()
    if return_confidence:
        probs = np.max(predictions, axis=-1)
        return plates, probs
    return plates
