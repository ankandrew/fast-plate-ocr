"""
Tests for config module
"""

from pathlib import Path

import pytest

from fast_plate_ocr.inference.config import PlateOCRConfig as PlateOCRConfigInference
from fast_plate_ocr.train.model.config import PlateOCRConfig as PlateOCRConfigTrain
from fast_plate_ocr.train.model.config import load_config_from_yaml
from test import PROJECT_ROOT_DIR


@pytest.mark.parametrize(
    "file_path",
    [f for f in PROJECT_ROOT_DIR.joinpath("config").iterdir() if f.suffix in (".yaml", ".yml")],
)
def test_yaml_configs_are_valid(file_path: Path) -> None:
    load_config_from_yaml(file_path)


@pytest.mark.parametrize(
    "file_path",
    [f for f in PROJECT_ROOT_DIR.joinpath("config").iterdir() if f.suffix in (".yaml", ".yml")],
)
def test_yaml_configs_for_inference_are_valid(file_path: Path) -> None:
    PlateOCRConfigInference.from_yaml(file_path)


@pytest.mark.parametrize(
    "raw_config",
    [
        {
            "max_plate_slots": 7,
            # Pad char not in alphabet, should raise exception
            "alphabet": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "pad_char": "_",
            "img_height": 70,
            "img_width": 140,
        }
    ],
)
def test_invalid_config_raises(raw_config: dict) -> None:
    with pytest.raises(ValueError):
        PlateOCRConfigTrain(**raw_config)


def test_configs_are_consistent():
    typed_dict_fields = set(PlateOCRConfigInference.__annotations__.keys())
    pydantic_fields = set(PlateOCRConfigTrain.model_fields.keys())
    assert typed_dict_fields == pydantic_fields, "Train and inference config have different fields"
