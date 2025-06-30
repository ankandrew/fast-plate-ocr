"""
Dataset module.
"""

import math
import os

import albumentations as A
import numpy as np
import numpy.typing as npt
import pandas as pd
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset

from fast_plate_ocr.core.process import read_and_resize_plate_image
from fast_plate_ocr.train.model.config import PlateOCRConfig
from fast_plate_ocr.train.utilities import utils


class PlateRecognitionPyDataset(PyDataset):
    """
    Custom PyDataset for OCR license plate recognition.
    """

    def __init__(
        self,
        annotations_file: str | os.PathLike,
        plate_config: PlateOCRConfig,
        batch_size: int,
        transform: A.Compose | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Load annotations
        annotations = pd.read_csv(annotations_file, dtype={"plate_text": str})
        annotations["image_path"] = (
            os.path.dirname(os.path.realpath(annotations_file)) + os.sep + annotations["image_path"]
        )
        # Check that plate lengths do not exceed max_plate_slots.
        assert (annotations["plate_text"].str.len() <= plate_config.max_plate_slots).all(), (
            "Plates are longer than max_plate_slots specified param. Change the parameter."
        )
        # Convert the dataframe to a NumPy array
        self.annotations = annotations.to_numpy()

        self.plate_config = plate_config
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Shuffle once at initialization if `shuffle=True`
        self._shuffle_data()

    def __len__(self) -> int:
        return math.ceil(len(self.annotations) / self.batch_size)

    def __getitem__(self, idx: int) -> tuple[npt.NDArray, npt.NDArray]:
        # Determine the idx-es of current batch
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.annotations))
        batch = self.annotations[low:high]

        batch_x = []
        batch_y = []
        for image_path, plate_text in batch:
            # Read and process image
            x = read_and_resize_plate_image(
                image_path=image_path,
                img_height=self.plate_config.img_height,
                img_width=self.plate_config.img_width,
                image_color_mode=self.plate_config.image_color_mode,
                keep_aspect_ratio=self.plate_config.keep_aspect_ratio,
                interpolation_method=self.plate_config.interpolation,
                padding_color=self.plate_config.padding_color,
            )
            # Transform target
            y = utils.target_transform(
                plate_text=plate_text,
                max_plate_slots=self.plate_config.max_plate_slots,
                alphabet=self.plate_config.alphabet,
                pad_char=self.plate_config.pad_char,
            )
            # Apply augmentation if provided
            if self.transform:
                x = self.transform(image=x)["image"]
            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x), np.array(batch_y)

    def _shuffle_data(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.annotations)

    def on_epoch_begin(self) -> None:
        # Optionally shuffle the dataset at the start of each epoch
        self._shuffle_data()
