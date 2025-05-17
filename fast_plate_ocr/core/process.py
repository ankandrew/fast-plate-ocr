"""
Utility functions for processing model input/output.
"""

import os
from typing import Literal, TypeAlias

import cv2
import numpy as np

ImageInterpolation: TypeAlias = Literal["nearest", "linear", "cubic", "area", "lanczos4"]
"""Interpolation method used for resizing the input image."""
ImageColorMode: TypeAlias = Literal["grayscale", "rgb"]
"""Input image color mode. Use 'grayscale' for single-channel input or 'rgb' for 3-channel input."""
PaddingColor: TypeAlias = tuple[int, int, int] | int
"""Padding colour for letterboxing (only used when keeping image aspect ratio)."""


INTERPOLATION_MAP: dict[ImageInterpolation, int] = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos4": cv2.INTER_LANCZOS4,
}


def read_plate_image(
    image_path: str,
    image_color_mode: ImageColorMode = "grayscale",
) -> np.ndarray:
    """
    Read an image from disk in the specified color mode.

    :param image_path: Path to the image file.
    :param image_color_mode: "grayscale" for single-channel or "rgb" for 3-channel color.
    :return: The image as a NumPy array. Grayscale images have shape (H, W), RGB images have shape
     (H, W, 3).
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the image cannot be decoded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if image_color_mode == "rgb":
        raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError(f"Failed to decode image: {image_path}")
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to decode image: {image_path}")

    return img


def resize_image(
    img: np.ndarray,
    img_height: int,
    img_width: int,
    image_color_mode: ImageColorMode = "grayscale",
    keep_aspect_ratio: bool = False,
    interpolation_method: ImageInterpolation = "linear",
    padding_color: PaddingColor = (114, 114, 114),
) -> np.ndarray:
    """
    Resize an in-memory image with optional aspect-ratio preservation and padding.

    :param img: Input image as a NumPy array (either grayscale or RGB).
    :param img_height: Target image height.
    :param img_width: Target image width.
    :param image_color_mode: Output color mode, "grayscale" or "rgb".
    :param keep_aspect_ratio: If True, maintain aspect ratio and apply letterbox padding.
    :param interpolation_method: Interpolation method used for resizing.
    :param padding_color: Padding color (int for grayscale, tuple for RGB).
    :return: The resized image as a NumPy array with shape (H, W, C).
    """
    interpolation = INTERPOLATION_MAP[interpolation_method]

    if not keep_aspect_ratio:
        img = cv2.resize(img, (img_width, img_height), interpolation=interpolation)

    else:
        orig_h, orig_w = img.shape[:2]
        # Scale ratio (new / old) – choose the limiting dimension
        r = min(img_height / orig_h, img_width / orig_w)
        # Compute the size of the resized (unpadded) image
        new_unpad_w, new_unpad_h = int(round(orig_w * r)), int(round(orig_h * r))
        # Resize if necessary
        if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
            img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=interpolation)
        # Padding on each side
        dw, dh = (img_width - new_unpad_w) / 2, (img_height - new_unpad_h) / 2
        top, bottom, left, right = (
            int(round(dh - 0.1)),
            int(round(dh + 0.1)),
            int(round(dw - 0.1)),
            int(round(dw + 0.1)),
        )
        # Ensure padding colour matches channel count
        if image_color_mode == "grayscale":
            if isinstance(padding_color, tuple):
                border_color = int(padding_color[0])
            else:
                border_color = int(padding_color)
        # RGB
        else:
            if isinstance(padding_color, tuple):
                if len(padding_color) != 3:
                    raise ValueError("padding_color must be length-3 for RGB images")
                border_color = tuple(int(c) for c in padding_color)
            else:
                border_color = (int(padding_color),) * 3
        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color,
        )

    # Add channel axis for gray so output is HxWxC
    if image_color_mode == "grayscale" and img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    return img


def read_and_resize_plate_image(
    image_path: str,
    img_height: int,
    img_width: int,
    image_color_mode: ImageColorMode = "grayscale",
    keep_aspect_ratio: bool = False,
    interpolation_method: ImageInterpolation = "linear",
    padding_color: PaddingColor = (114, 114, 114),
) -> np.ndarray:
    """
    Convenience function to read an image from disk and resize it for model input.

    :param image_path: Path to the input image.
    :param img_height: Desired height after resizing.
    :param img_width: Desired width after resizing.
    :param image_color_mode: "grayscale" or "rgb" mode for reading and resizing.
    :param keep_aspect_ratio: Whether to maintain aspect ratio using letterboxing.
    :param interpolation_method: Interpolation method used for resizing.
    :param padding_color: Padding color is used if the aspect ratio is preserved.
    :return: Resized (and possibly padded) image with shape (H, W, C).
    """
    img = read_plate_image(image_path, image_color_mode=image_color_mode)
    return resize_image(
        img,
        img_height,
        img_width,
        image_color_mode=image_color_mode,
        keep_aspect_ratio=keep_aspect_ratio,
        interpolation_method=interpolation_method,
        padding_color=padding_color,
    )


def preprocess_image(images: np.ndarray) -> np.ndarray:
    """
    Preprocess the image(s), so they're ready to be fed to the model.

    Note: We don't normalize the pixel values between [0, 1] here, because that the model-first
    layer does that.

    :param images: The image(s) contained in a NumPy array.
    :return: A numpy array with shape (N, H, W, C).
    """
    # single sample (H, W, C)
    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)

    if images.ndim != 4:
        raise ValueError("Expected input of shape (N, H, W, C).")

    return images.astype(np.uint8)


def postprocess_output(
    model_output: np.ndarray,
    max_plate_slots: int,
    model_alphabet: str,
    return_confidence: bool = False,
) -> tuple[list[str], np.ndarray] | list[str]:
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
