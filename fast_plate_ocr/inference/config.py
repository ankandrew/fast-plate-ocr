"""
Model config reading/parsing for doing inference.
"""

from dataclasses import dataclass
from os import PathLike

import yaml

from fast_plate_ocr.core.process import ImageColorMode, ImageInterpolation

# pylint: disable=duplicate-code


@dataclass(frozen=True)
class PlateOCRConfig:  # pylint: disable=too-many-instance-attributes
    """
    Plate OCR Config used for inference.

    This dataclass is used to read and parse the config file used for training the OCR model.
    We prefer to keep the inference package with minimal dependencies and avoid using Pydantic here.
    """

    max_plate_slots: int
    """
    Max number of plate slots supported. This represents the number of model classification heads.
    """
    alphabet: str
    """
    All the possible character set for the model output.
    """
    pad_char: str
    """
    Padding character for plates which length is smaller than MAX_PLATE_SLOTS.
    """
    img_height: int
    """
    Image height which is fed to the model.
    """
    img_width: int
    """
    Image width which is fed to the model.
    """
    keep_aspect_ratio: bool = False
    """
    Keep aspect ratio of the input image.
    """
    interpolation: ImageInterpolation = "linear"
    """
    Interpolation method used for resizing the input image.
    """
    image_color_mode: ImageColorMode = "grayscale"
    """
    Input image color mode. Use 'grayscale' for single-channel input or 'rgb' for 3-channel input.
    """
    padding_color: tuple[int, int, int] | int = (114, 114, 114)
    """
    Padding color used when keep_aspect_ratio is True. For grayscale images, this should be a single
    integer and for RGB images, this must be a tuple of three integers.
    """

    @property
    def vocabulary_size(self) -> int:
        return len(self.alphabet)

    @property
    def pad_idx(self) -> int:
        return self.alphabet.index(self.pad_char)

    @property
    def num_channels(self) -> int:
        return 3 if self.image_color_mode == "rgb" else 1

    @classmethod
    def from_yaml(cls, path: str | PathLike[str]) -> "PlateOCRConfig":
        """
        Read and parse a yaml containing the Plate OCR config.
        """
        with open(path, encoding="utf-8") as f_in:
            data = yaml.safe_load(f_in)
        return cls(**data)
