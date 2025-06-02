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
def test_train_cli_runs_successfully(dummy_dataset: pathlib.Path, tmp_path: pathlib.Path) -> None:
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        """
        max_plate_slots: 9
        alphabet: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
        pad_char: '_'
        img_height: 64
        img_width: 128
        """
    )

    output_dir = tmp_path / "out"

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        train_cli,
        [
            "--config-file",
            str(config_yaml),
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
