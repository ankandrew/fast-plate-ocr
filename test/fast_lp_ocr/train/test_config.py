"""
Tests for config module
"""

from pathlib import Path

import pytest

from fast_plate_ocr.core.types import ImageColorMode
from fast_plate_ocr.inference.config import PlateOCRConfig as PlateOCRConfigInference
from fast_plate_ocr.train.model.config import PlateOCRConfig as PlateOCRConfigTrain
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from test import PROJECT_ROOT_DIR

PROJECT_DEFAULT_CONFIGS = [
    f for f in PROJECT_ROOT_DIR.joinpath("config").iterdir() if f.suffix in (".yaml", ".yml")
]
"""Default OCR model configs present in the project."""


@pytest.mark.parametrize("file_path", PROJECT_DEFAULT_CONFIGS)
def test_yaml_configs_are_valid(file_path: Path) -> None:
    load_plate_config_from_yaml(file_path)


@pytest.mark.parametrize("file_path", PROJECT_DEFAULT_CONFIGS)
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


@pytest.mark.parametrize(
    "alphabet, pad_char, expected_vocab, expected_pad_idx",
    [
        ("ABC", "A", 3, 0),
        ("0123456789", "5", 10, 5),
        ("Z", "Z", 1, 0),
    ],
)
def test_vocabulary_and_pad_idx(
    alphabet: str, pad_char: str, expected_vocab: int, expected_pad_idx: int
) -> None:
    cfg = PlateOCRConfigTrain(
        max_plate_slots=5,
        alphabet=alphabet,
        pad_char=pad_char,
        img_height=32,
        img_width=64,
    )
    assert cfg.vocabulary_size == expected_vocab
    assert cfg.pad_idx == expected_pad_idx


@pytest.mark.parametrize(
    "color_mode, expected_channels",
    [
        ("grayscale", 1),
        ("rgb", 3),
    ],
)
def test_num_channels(color_mode: ImageColorMode, expected_channels: int) -> None:
    cfg = PlateOCRConfigTrain(
        max_plate_slots=5,
        alphabet="ABC",
        pad_char="A",
        img_height=32,
        img_width=64,
        image_color_mode=color_mode,
    )
    assert cfg.num_channels == expected_channels


def test_train_config_defaults_are_correct() -> None:
    cfg = PlateOCRConfigTrain(
        max_plate_slots=3,
        alphabet="XYZ",
        pad_char="X",
        img_height=16,
        img_width=32,
    )

    assert cfg.keep_aspect_ratio is False
    assert cfg.interpolation == "linear"
    assert cfg.image_color_mode == "grayscale"
