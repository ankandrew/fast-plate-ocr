"""
Tests for the export script.
"""

import warnings
from pathlib import Path

import pytest
from click.testing import CliRunner

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fast_plate_ocr.cli.export import export as export_cli
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.model.model_builders import build_model
from fast_plate_ocr.train.model.model_schema import load_model_config_from_yaml
from test import MODEL_CONFIG_PATHS, PLATE_CONFIG_DIR

LATIN_VOCAB_PLATE_CONFIG = PLATE_CONFIG_DIR / "default_latin_plate_config.yaml"


def _build_and_save_keras_model(model_cfg_path: Path, plate_cfg_path: Path, save_dir: Path) -> Path:
    model_cfg = load_model_config_from_yaml(model_cfg_path)
    plate_cfg = load_plate_config_from_yaml(plate_cfg_path)

    model = build_model(model_cfg, plate_cfg)
    model_save_path = save_dir / "model.keras"
    model.save(model_save_path)

    return model_save_path


@pytest.mark.parametrize("model_config_path", MODEL_CONFIG_PATHS)
@pytest.mark.parametrize("simplify", [False, True])
@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_export_to_onnx(
    model_config_path: Path,
    simplify: bool,
    dynamic_batch: bool,
    tmp_path: Path,
) -> None:
    plate_config_path = LATIN_VOCAB_PLATE_CONFIG
    runner = CliRunner()
    # Build and save the Keras model
    model_save_path = _build_and_save_keras_model(
        model_config_path, LATIN_VOCAB_PLATE_CONFIG, tmp_path
    )
    # Export with given parameters
    args = [
        "-m",
        str(model_save_path),
        "--plate-config-file",
        str(plate_config_path),
        "--format",
        "onnx",
    ]
    if simplify:
        args.append("--simplify")
    if dynamic_batch:
        args.append("--dynamic-batch")

    result = runner.invoke(export_cli, args)
    assert result.exit_code == 0, result.output
    exported_path = model_save_path.with_suffix(".onnx")
    assert exported_path.exists(), f"Expected exported ONNX file at {exported_path}"


@pytest.mark.parametrize("model_config_path", MODEL_CONFIG_PATHS)
def test_export_to_tflite(
    model_config_path: Path,
    tmp_path: Path,
) -> None:
    plate_config_path = LATIN_VOCAB_PLATE_CONFIG
    runner = CliRunner()
    # Build and save the Keras model
    model_save_path = _build_and_save_keras_model(model_config_path, plate_config_path, tmp_path)
    # Construct CLI arguments for TFLite
    args = [
        "-m",
        str(model_save_path),
        "--plate-config-file",
        str(plate_config_path),
        "--format",
        "tflite",
    ]

    result = runner.invoke(export_cli, args)
    assert result.exit_code == 0, result.output
    exported_path = model_save_path.with_suffix(".tflite")
    assert exported_path.exists(), f"Expected exported TFLite file at {exported_path}"


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("model_config_path", MODEL_CONFIG_PATHS)
def test_export_to_coreml(
    model_config_path: Path,
    tmp_path: Path,
) -> None:
    plate_config_path = LATIN_VOCAB_PLATE_CONFIG
    runner = CliRunner()
    # Build and save the Keras model
    model_save_path = _build_and_save_keras_model(model_config_path, plate_config_path, tmp_path)
    # Construct CLI arguments for CoreML
    args = [
        "-m",
        str(model_save_path),
        "--plate-config-file",
        str(plate_config_path),
        "--format",
        "coreml",
    ]

    result = runner.invoke(export_cli, args)
    assert result.exit_code == 0, result.output
    exported_path = model_save_path.with_suffix(".mlpackage")
    assert exported_path.exists(), f"Expected exported CoreML file at {exported_path}"
