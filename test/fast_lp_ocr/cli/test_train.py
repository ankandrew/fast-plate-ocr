"""
Tests for the train script.
"""

import pathlib
import warnings

import pytest
from click.testing import CliRunner

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fast_plate_ocr.cli.train import train as train_cli


@pytest.mark.filterwarnings("ignore")
def test_train_cli_runs_successfully(
    dummy_dataset: pathlib.Path,
    dummy_plate_config: str,
    dummy_cct_model_config: str,
    tmp_path: pathlib.Path,
) -> None:
    plate_config_yaml = tmp_path / "config.yaml"
    plate_config_yaml.write_text(dummy_plate_config)
    model_config_yaml = tmp_path / "model.yaml"
    model_config_yaml.write_text(dummy_cct_model_config)

    output_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        [
            "--model-config-file",
            str(model_config_yaml),
            "--plate-config-file",
            str(plate_config_yaml),
            "--annotations",
            str(dummy_dataset),
            "--val-annotations",
            str(dummy_dataset),
            "--batch-size",
            "2",
            "--epochs",
            "1",
            "--output-dir",
            str(output_dir),
            "--workers",
            "0",
            "--no-use-multiprocessing",
            "--loss",
            "cce",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_dir.exists(), "Train script did not create output directory"

    sub_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    assert len(sub_dirs) == 1, "Expected exactly one timestamped output directory"
    run_dir = sub_dirs[0]

    assert (run_dir / "model_config.yaml").exists(), "Model config was not saved"
    assert (run_dir / "plate_config.yaml").exists(), "Plate config was not saved"
    assert (run_dir / "train_augmentation.yaml").exists(), "Train augmentation config was not saved"
    assert (run_dir / "hyper_params.json").exists(), "Hyperparameters JSON was not saved"
