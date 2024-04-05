"""
Model config reading/parsing for doing inference.
"""

from os import PathLike
from typing import TypedDict

import yaml

# pylint: disable=duplicate-code


class PlateOCRConfig(TypedDict):
    """
    Plate OCR Config used for inference.

    This has the same attributes as the one used in the training Pydantic BaseModel. We use this to
    avoid having Pydantic as a required dependency of the minimal package install.
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


def load_config_from_yaml(yaml_file_path: str | PathLike[str]) -> PlateOCRConfig:
    """
    Read and parse a yaml containing the Plate OCR config.

    Note: This is currently not using Pydantic for parsing/validating to avoid adding it a python
    dependency as part of the minimal installation.
    """
    with open(yaml_file_path, encoding="utf-8") as f_in:
        config: PlateOCRConfig = yaml.safe_load(f_in)
    return config
