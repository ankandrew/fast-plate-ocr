"""
Dataset module.
"""

import albumentations as A
import pandas as pd
from torch.utils.data import Dataset

from fast_plate_ocr import utils
from fast_plate_ocr.config import (
    DEFAULT_IMG_HEIGHT,
    DEFAULT_IMG_WIDTH,
    MAX_PLATE_SLOTS,
    MODEL_ALPHABET,
    PAD_CHAR,
)
from fast_plate_ocr.custom_types import FilePath


class LicensePlateDataset(Dataset):
    def __init__(
        self,
        annotations_file: FilePath,
        img_height: int = DEFAULT_IMG_HEIGHT,
        img_width: int = DEFAULT_IMG_WIDTH,
        max_plate_slots: int = MAX_PLATE_SLOTS,
        alphabet: str = MODEL_ALPHABET,
        pad_char: str = PAD_CHAR,
        transform: A.Compose | None = None,
    ):
        self.annotations = pd.read_csv(annotations_file)
        assert (
            self.annotations["plate_text"].str.len() <= max_plate_slots
        ).all(), f"Plates are longer than {max_plate_slots}. Change the max_plate_slots parameter."
        self.img_height = img_height
        self.img_width = img_width
        self.max_plate_slots = max_plate_slots
        self.alphabet = alphabet
        self.pad_char = pad_char
        self.transform = transform

    def __len__(self):
        return len(self.annotations.index)

    def __getitem__(self, idx):
        annotation = self.annotations.iloc[idx]
        x = utils.read_plate_image(
            image_path=annotation.image_path,
            img_height=self.img_height,
            img_width=self.img_width,
        )
        y = utils.target_transform(
            plate_text=annotation.plate_text,
            max_plate_slots=self.max_plate_slots,
            alphabet=self.alphabet,
            pad_char=self.pad_char,
        )
        if self.transform:
            x = self.transform(image=x)["image"]
        return x, y
