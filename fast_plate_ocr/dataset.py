"""
Dataset module.
"""

import albumentations as A
import pandas as pd
from torch.utils.data import Dataset

from fast_plate_ocr import utils
from fast_plate_ocr.config import PlateOCRConfig
from fast_plate_ocr.custom_types import FilePath


class LicensePlateDataset(Dataset):
    def __init__(
        self,
        annotations_file: FilePath,
        config: PlateOCRConfig,
        transform: A.Compose | None = None,
    ):
        self.annotations = pd.read_csv(annotations_file)
        assert (
            self.annotations["plate_text"].str.len() <= config.max_plate_slots
        ).all(), "Plates are longer than max_plate_slots specified param. Change the parameter."
        self.config = config
        self.transform = transform

    def __len__(self):
        return len(self.annotations.index)

    def __getitem__(self, idx):
        annotation = self.annotations.iloc[idx]
        x = utils.read_plate_image(
            image_path=annotation.image_path,
            img_height=self.config.img_height,
            img_width=self.config.img_width,
        )
        y = utils.target_transform(
            plate_text=annotation.plate_text,
            max_plate_slots=self.config.max_plate_slots,
            alphabet=self.config.alphabet,
            pad_char=self.config.pad_char,
        )
        if self.transform:
            x = self.transform(image=x)["image"]
        return x, y
