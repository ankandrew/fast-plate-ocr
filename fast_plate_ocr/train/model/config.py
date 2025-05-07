"""
Config values used throughout the code.
"""

from os import PathLike
from typing import Annotated

import yaml
from pydantic import (
    BaseModel,
    PositiveInt,
    StringConstraints,
    computed_field,
    model_validator,
)


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

    @computed_field  # type: ignore[misc]
    @property
    def vocabulary_size(self) -> int:
        return len(self.alphabet)

    @computed_field  # type: ignore[misc]
    @property
    def pad_idx(self) -> int:
        return self.alphabet.index(self.pad_char)

    @model_validator(mode="after")
    def check_pad_in_alphabet(self) -> "PlateOCRConfig":
        if self.pad_char not in self.alphabet:
            raise ValueError("Pad character must be present in model alphabet.")
        return self


def load_config_from_yaml(yaml_file_path: str | PathLike[str]) -> PlateOCRConfig:
    """Read and parse a yaml containing the Plate OCR config."""
    with open(yaml_file_path, encoding="utf-8") as f_in:
        yaml_content = yaml.safe_load(f_in)
    config = PlateOCRConfig(**yaml_content)
    return config
