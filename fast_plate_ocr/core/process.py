"""
Utility functions for processing model input/output.
"""

import os

import cv2
import numpy as np

from fast_plate_ocr.core.types import (
    ImageColorMode,
    ImageInterpolation,
    PaddingColor,
    PathLike,
)

INTERPOLATION_MAP: dict[ImageInterpolation, int] = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos4": cv2.INTER_LANCZOS4,
}
"""Mapping from interpolation method name to OpenCV constant."""


def read_plate_image(
    image_path: PathLike,
    image_color_mode: ImageColorMode = "grayscale",
) -> np.ndarray:
    """
    Reads an image from disk in the requested colour mode.

    Args:
        image_path: Path to the image file.
        image_color_mode: ``"grayscale"`` for single-channel or ``"rgb"`` for three-channel
            colour. Defaults to ``"grayscale"``.

    Returns:
        The image as a NumPy array.
            Grayscale images have shape ``(H, W)``, RGB images have shape ``(H, W, 3)``.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be decoded.
    """
    image_path = str(image_path)

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
    Resizes an in-memory image with optional aspect-ratio preservation and padding.

    Args:
        img: Input image.
        img_height: Target image height.
        img_width: Target image width.
        image_color_mode: Output colour mode, ``"grayscale"`` or ``"rgb"``.
        keep_aspect_ratio: If ``True``, maintain the original aspect ratio using letter-box
            padding. Defaults to ``False``.
        interpolation_method: Interpolation method used for resizing. Defaults to ``"linear"``.
        padding_color: Padding colour (scalar for grayscale, tuple for RGB). Defaults to
            ``(114, 114, 114)``.

    Returns:
        The resized image with shape ``(H, W, C)`` (a channel axis is added for grayscale).

    Raises:
        ValueError: If ``padding_color`` length is not 3 for RGB output.
    """
    # pylint: disable=too-many-locals

    interpolation = INTERPOLATION_MAP[interpolation_method]

    if not keep_aspect_ratio:
        img = cv2.resize(img, (img_width, img_height), interpolation=interpolation)

    else:
        orig_h, orig_w = img.shape[:2]
        # Scale ratio (new / old) - choose the limiting dimension
        r = min(img_height / orig_h, img_width / orig_w)
        # Compute the size of the resized (unpadded) image
        new_unpad_w, new_unpad_h = round(orig_w * r), round(orig_h * r)
        # Resize if necessary
        if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
            img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=interpolation)
        # Padding on each side
        dw, dh = (img_width - new_unpad_w) / 2, (img_height - new_unpad_h) / 2
        top, bottom, left, right = (
            round(dh - 0.1),
            round(dh + 0.1),
            round(dw - 0.1),
            round(dw + 0.1),
        )
        border_color: PaddingColor
        # Ensure padding colour matches channel count
        if image_color_mode == "grayscale":
            if isinstance(padding_color, tuple):
                border_color = int(padding_color[0])
            else:
                border_color = int(padding_color)
        elif image_color_mode == "rgb":
            if isinstance(padding_color, tuple):
                if len(padding_color) != 3:
                    raise ValueError("padding_color must be length-3 for RGB images")
                border_color = tuple(int(c) for c in padding_color)  # type: ignore[assignment]
            else:
                border_color = (int(padding_color),) * 3
        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color,  # type: ignore[arg-type]
        )
    # Add channel axis for gray so output is HxWxC
    if image_color_mode == "grayscale" and img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    return img


def read_and_resize_plate_image(
    image_path: PathLike,
    img_height: int,
    img_width: int,
    image_color_mode: ImageColorMode = "grayscale",
    keep_aspect_ratio: bool = False,
    interpolation_method: ImageInterpolation = "linear",
    padding_color: PaddingColor = (114, 114, 114),
) -> np.ndarray:
    """
    Reads an image from disk and resizes it for model input.

    Args:
        image_path: Path to the image.
        img_height: Desired output height.
        img_width: Desired output width.
        image_color_mode: ``"grayscale"`` or ``"rgb"``. Defaults to ``"grayscale"``.
        keep_aspect_ratio: Whether to preserve aspect ratio via letter-boxing. Defaults to
            ``False``.
        interpolation_method: Interpolation method to use. Defaults to ``"linear"``.
        padding_color: Colour used for padding when aspect ratio is preserved. Defaults to
            ``(114, 114, 114)``.

    Returns:
        The resized (and possibly padded) image with shape ``(H, W, C)``.
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
    Converts image data to the format expected by the model.

    The model itself handles pixel-value normalisation, so this function only ensures the
    batch-dimension and dtype are correct.

    Args:
        images: Image or batch of images with shape ``(H, W, C)`` or ``(N, H, W, C)``.

    Returns:
        A NumPy array with shape ``(N, H, W, C)`` and dtype ``uint8``.

    Raises:
        ValueError: If the input does not have 3 or 4 dimensions.
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
    Decodes model predictions into licence-plate strings.

    Args:
        model_output: Raw output tensor from the model.
        max_plate_slots: Maximum number of character positions.
        model_alphabet: Alphabet used by the model.
        return_confidence: If ``True``, also return per-character confidence scores.
            Defaults to ``False``.

    Returns:
        If ``return_confidence`` is ``False``: a list of decoded plate strings.
            If ``True``: a two-tuple ``(plates, probs)`` where

            * ``plates`` is the list of decoded strings, and
            * ``probs`` is an array of shape ``(N, max_plate_slots)`` with the corresponding
              confidence scores.
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
