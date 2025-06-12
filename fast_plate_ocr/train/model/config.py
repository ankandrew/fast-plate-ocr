"""
Config values used throughout the code.
"""

from typing import Annotated, TypeAlias

import annotated_types
import yaml
from pydantic import (
    BaseModel,
    PositiveInt,
    StringConstraints,
    computed_field,
    model_validator,
)

from fast_plate_ocr.core.types import ImageColorMode, ImageInterpolation, PathLike

UInt8: TypeAlias = Annotated[int, annotated_types.Ge(0), annotated_types.Le(255)]
"""
An integer in the range [0, 255], used for color channel values.
"""


class PlateOCRConfig(BaseModel, extra="forbid", frozen=True):
    """
    Model License Plate OCR config.
    """

    max_plate_slots: PositiveInt
    """
    Max number of plate slots supported. This represents the number of model classification heads.
    """
    alphabet: str
    """
    All the possible character set for the model output.
    """
    pad_char: Annotated[str, StringConstraints(min_length=1, max_length=1)]
    """
    Padding character for plates which length is smaller than MAX_PLATE_SLOTS.
    """
    img_height: PositiveInt
    """
    Image height which is fed to the model.
    """
    img_width: PositiveInt
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
    padding_color: tuple[UInt8, UInt8, UInt8] | UInt8 = (114, 114, 114)
    """
    Padding color used when keep_aspect_ratio is True. For grayscale images, this should be a single
    integer and for RGB images, this must be a tuple of three integers.
    """

    @computed_field  # type: ignore[misc]
    @property
    def vocabulary_size(self) -> int:
        return len(self.alphabet)

    @computed_field  # type: ignore[misc]
    @property
    def pad_idx(self) -> int:
        return self.alphabet.index(self.pad_char)

    @computed_field  # type: ignore[misc]
    @property
    def num_channels(self) -> int:
        return 3 if self.image_color_mode == "rgb" else 1

    @model_validator(mode="after")
    def check_alphabet_and_pad(self) -> "PlateOCRConfig":
        # `pad_char` must be in alphabet
        if self.pad_char not in self.alphabet:
            raise ValueError("Pad character must be present in model alphabet.")
        # all chars in alphabet must be unique
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("Alphabet must not contain duplicate characters.")
        return self


def load_plate_config_from_yaml(yaml_file_path: PathLike) -> PlateOCRConfig:
    """Read and parse a YAML containing the plate config."""
    with open(yaml_file_path, encoding="utf-8") as f_in:
        yaml_content = yaml.safe_load(f_in)
    config = PlateOCRConfig(**yaml_content)
    return config
