"""
Test OCR models module.
"""

from pathlib import Path

import pytest

from fast_plate_ocr.train.model.model_builder import CCTModelConfig
from test import PROJECT_ROOT_DIR

PROJECT_MODELS_CONFIG = [
    f for f in PROJECT_ROOT_DIR.joinpath("models").iterdir() if f.suffix in (".yaml", ".yml")
]
"""Default models configs present in the project."""

CCT_MODEL_CONFIG = [f for f in PROJECT_MODELS_CONFIG if f.name.startswith("cct")]
"""Default CCT models configs present in the project."""


@pytest.mark.parametrize("file_path", CCT_MODEL_CONFIG)
def test_cct_models_yaml_configs_are_valid(file_path: Path) -> None:
    CCTModelConfig.from_yaml(file_path)
