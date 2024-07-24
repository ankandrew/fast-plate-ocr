"""
Tests for config module
"""

from pathlib import Path

import pytest

from fast_plate_ocr.train.model.config import PlateOCRConfig, load_config_from_yaml
from test import PROJECT_ROOT_DIR


@pytest.mark.parametrize(
    "file_path",
    [f for f in PROJECT_ROOT_DIR.joinpath("config").iterdir() if f.suffix in (".yaml", ".yml")],
)
def test_yaml_configs(file_path: Path) -> None:
    load_config_from_yaml(file_path)


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
        PlateOCRConfig(**raw_config)
