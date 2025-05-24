"""
Augmentations used for training the OCR model.
"""

import albumentations as A
import cv2

BORDER_COLOR_BLACK: tuple[int, int, int] = (0, 0, 0)

TRAIN_AUGMENTATION = A.Compose(
    [
        A.ShiftScaleRotate(
            shift_limit=0.02,
            scale_limit=(-0.3, 0.075),
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            fill=BORDER_COLOR_BLACK,
            p=1,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        A.GaussianBlur(sigma_limit=(0.2, 0.5), p=0.25),
        A.OneOf(
            [
                A.CoarseDropout(
                    num_holes_range=(1, 11),
                    hole_height_range=(1, 5),
                    hole_width_range=(1, 5),
                    p=0.3,
                ),
                A.PixelDropout(dropout_prob=0.02, p=0.2),
                A.GridDropout(ratio=0.3, fill="random", p=0.2),
            ],
            p=0.7,
        ),
    ]
)
"""Training augmentations recipe."""
