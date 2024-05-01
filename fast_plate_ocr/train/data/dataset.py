"""
Dataset module.
"""

import os
from os import PathLike

import albumentations as A
import numpy.typing as npt
import pandas as pd
from torch.utils.data import Dataset

from fast_plate_ocr.train.model.config import PlateOCRConfig
from fast_plate_ocr.train.utilities import utils


class LicensePlateDataset(Dataset):
    def __init__(
        self,
        annotations_file: str | PathLike[str],
        config: PlateOCRConfig,
        transform: A.Compose | None = None,
    ) -> None:
        annotations = pd.read_csv(annotations_file)
        annotations["image_path"] = (
            os.path.dirname(os.path.realpath(annotations_file)) + os.sep + annotations["image_path"]
        )
        assert (
            annotations["plate_text"].str.len() <= config.max_plate_slots
        ).all(), "Plates are longer than max_plate_slots specified param. Change the parameter."
        self.annotations = annotations.to_numpy()
        self.config = config
        self.transform = transform

    def __len__(self) -> int:
        return self.annotations.shape[0]

    def __getitem__(self, idx) -> tuple[npt.NDArray, npt.NDArray]:
        image_path, plate_text = self.annotations[idx]
        x = utils.read_plate_image(
            image_path=image_path,
            img_height=self.config.img_height,
            img_width=self.config.img_width,
        )
        y = utils.target_transform(
            plate_text=plate_text,
            max_plate_slots=self.config.max_plate_slots,
            alphabet=self.config.alphabet,
            pad_char=self.config.pad_char,
        )
        if self.transform:
            x = self.transform(image=x)["image"]
        return x, y
