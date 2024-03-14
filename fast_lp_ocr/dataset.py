"""
Dataset module.
"""

import math

import keras
import keras.src.utils as keras_utils
import numpy as np
import numpy.typing as npt
import pandas as pd

from fast_lp_ocr.config import (
    DEFAULT_IMG_HEIGHT,
    DEFAULT_IMG_WIDTH,
    MAX_PLATE_SLOTS,
    MODEL_ALPHABET,
    PAD_CHAR,
)
from fast_lp_ocr.custom_types import FilePath, PilInterpolation
from fast_lp_ocr.utils import target_transform


class LicensePlatesDataset(keras.utils.PyDataset):
    def __init__(
        self,
        annotations_file: FilePath,
        img_dir: FilePath,
        batch_size: int,
        img_height: int = DEFAULT_IMG_HEIGHT,
        img_width: int = DEFAULT_IMG_WIDTH,
        interpolation: PilInterpolation = "bilinear",
        max_plate_slots: int = MAX_PLATE_SLOTS,
        alphabet: str = MODEL_ALPHABET,
        pad_char: str = PAD_CHAR,
        transform=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.annotations = pd.read_csv(annotations_file)
        assert (
            self.annotations["plate_text"].str.len() <= max_plate_slots
        ).all(), f"Plates are longer than {max_plate_slots}. Change the max_plate_slots parameter"
        self.dataset_size = len(self.annotations.index)
        self.max_plate_slots = max_plate_slots
        self.img_dir = img_dir
        self.img_height = img_height
        self.img_width = img_width
        self.interpolation = interpolation
        self.alphabet = alphabet
        self.pad_char = pad_char
        self.transform = transform
        self.batch_size = batch_size

    def on_epoch_end(self) -> None:
        self.annotations.sample(frac=1).reset_index(drop=True)

    def __len__(self) -> int:
        """
        Return the number of batches in the dataset (rather than the number of samples).
        """
        return math.ceil(self.dataset_size / self.batch_size)

    def __getitem__(self, idx) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Return a complete batch (not a single sample).
        """
        # Return x, y for batch idx.
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, self.dataset_size)
        batch = self.annotations.iloc[low:high]
        batch_image_path, batch_plate_text = batch["image_path"], batch["plate_text"]
        images = [
            keras_utils.img_to_array(
                img=keras_utils.load_img(
                    p,
                    color_mode="grayscale",
                    target_size=(self.img_height, self.img_width),
                    interpolation=self.interpolation,
                ),
                dtype=np.uint8,
            )
            for p in batch_image_path.to_numpy()
        ]
        if self.transform:
            batch_x = np.array([self.transform(image=x)["image"] for x in images])
        else:
            batch_x = np.array(images)
        batch_y = target_transform(
            plate_text=batch_plate_text,
            max_plate_slots=self.max_plate_slots,
            alphabet=self.alphabet,
            pad_char=self.pad_char,
        )
        return batch_x, batch_y
