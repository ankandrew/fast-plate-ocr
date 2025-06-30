"""
Test OCR models module.
"""

from pathlib import Path

import pytest

from fast_plate_ocr.train.model.model_schema import load_model_config_from_yaml
from test import MODEL_CONFIG_PATHS


@pytest.mark.parametrize("file_path", MODEL_CONFIG_PATHS)
def test_models_yaml_configs_are_valid(file_path: Path) -> None:
    load_model_config_from_yaml(file_path)
