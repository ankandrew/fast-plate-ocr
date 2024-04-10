"""
Augmentations used for training the OCR model.
"""

import albumentations as A
import cv2

BORDER_COLOR_BLACK: tuple[int, int, int] = (0, 0, 0)

TRAIN_AUGMENTATION = A.Compose(
    [
        A.ShiftScaleRotate(
            shift_limit=0.06,
            scale_limit=0.1,
            rotate_limit=9,
            border_mode=cv2.BORDER_CONSTANT,
            value=BORDER_COLOR_BLACK,
            p=1,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
        A.MotionBlur(blur_limit=(3, 5), p=0.1),
        A.OneOf(
            [
                A.CoarseDropout(max_holes=10, max_height=4, max_width=4, p=0.3),
                A.PixelDropout(dropout_prob=0.01, p=0.2),
            ],
            p=0.7,
        ),
    ]
)
"""Training augmentations recipe."""
