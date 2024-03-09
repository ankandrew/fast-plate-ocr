import albumentations as A

TRAIN_AUGMENTATION = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=8, p=1),
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1),
        A.Affine(shear=8, p=1),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 7)),
                # TODO: Add more blurs here
            ]
        ),
        A.CoarseDropout(max_holes=3, p=0.85),
        # TODO: Add more augmentations here. Tip: try them first in albumentations/demo.
        A.PixelDropout(dropout_prob=0.01, p=0.25),
        A.ImageCompression(quality_lower=50, quality_upper=70, p=0.10),
        # TODO: Normalize image between [0, 1]
    ]
)
"""Training augmentations recipe."""

VAL_AUGMENTATION = A.Compose(
    [
        # TODO: Normalize image (same as training)
    ]
)
"""Validation augmentations recipe."""
