"""
Augmentations used for training the OCR model.
"""

import albumentations as A
import cv2

from fast_plate_ocr.core.types import ImageColorMode

BORDER_COLOR_BLACK: tuple[int, int, int] = (0, 0, 0)


def default_train_augmentation(img_color_mode: ImageColorMode) -> A.Compose:
    """
    Default training augmentation pipeline.
    """
    if img_color_mode == "grayscale":
        return A.Compose(
            [
                A.Affine(
                    translate_percent=(-0.02, 0.02),
                    scale=(0.75, 1.10),
                    rotate=(-15, 15),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=BORDER_COLOR_BLACK,
                    shear=(0.0, 0.0),
                    p=0.75,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                A.GaussianBlur(sigma_limit=(0.2, 0.5), p=0.25),
                A.OneOf(
                    [
                        A.CoarseDropout(
                            num_holes_range=(1, 14),
                            hole_height_range=(1, 5),
                            hole_width_range=(1, 5),
                            p=0.2,
                        ),
                        A.PixelDropout(dropout_prob=0.02, p=0.2),
                        A.GridDropout(ratio=0.3, fill="random", p=0.2),
                    ],
                    p=0.7,
                ),
            ]
        )
    if img_color_mode == "rgb":
        return A.Compose(
            [
                A.Affine(
                    translate_percent=(-0.02, 0.02),
                    scale=(0.75, 1.10),
                    rotate=(-15, 15),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=BORDER_COLOR_BLACK,
                    shear=(0.0, 0.0),
                    p=0.75,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
                A.OneOf(
                    [
                        A.HueSaturationValue(
                            hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.7
                        ),
                        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
                    ],
                    p=0.3,
                ),
                A.RandomGamma(gamma_limit=(95, 105), p=0.20),
                A.ToGray(p=0.05),
                A.OneOf(
                    [
                        A.GaussianBlur(sigma_limit=(0.2, 0.5), p=0.5),
                        A.MotionBlur(blur_limit=(3, 3), p=0.5),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
                        A.MultiplicativeNoise(multiplier=(0.98, 1.02), p=0.1),
                        A.ISONoise(intensity=(0.005, 0.02), p=0.1),
                        A.ImageCompression(quality_range=(55, 90), p=0.1),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.CoarseDropout(
                            num_holes_range=(1, 14),
                            hole_height_range=(1, 5),
                            hole_width_range=(1, 5),
                            p=0.2,
                        ),
                        A.PixelDropout(dropout_prob=0.02, p=0.3),
                        A.GridDropout(ratio=0.3, fill="random", p=0.3),
                    ],
                    p=0.5,
                ),
            ]
        )
    raise ValueError(f"Unsupported img_color_mode: {img_color_mode!r}. Expected 'grayscale'/'rgb'.")
