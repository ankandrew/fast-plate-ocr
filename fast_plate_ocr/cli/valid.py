"""
Script for validating trained OCR models.
"""

import pathlib

import click

from fast_plate_ocr.train.data.dataset import PlateRecognitionPyDataset

# Custom metris / losses
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.utilities import utils


@click.command(context_settings={"max_content_width": 120})
@click.option(
    "-m",
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Path to the saved .keras model.",
)
@click.option(
    "--plate-config-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, path_type=pathlib.Path),
    help="Path pointing to the model license plate OCR config.",
)
@click.option(
    "-a",
    "--annotations",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Annotations file used for validation.",
)
@click.option(
    "-b",
    "--batch-size",
    default=1,
    show_default=True,
    type=int,
    help="Batch size.",
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    type=int,
    help="Number of worker threads/processes for parallel data loading via PyDataset.",
)
@click.option(
    "--use-multiprocessing/--no-use-multiprocessing",
    default=False,
    show_default=True,
    help="Whether to use multiprocessing for data loading.",
)
@click.option(
    "--max-queue-size",
    default=10,
    show_default=True,
    type=int,
    help="Maximum number of batches to prefetch for the dataset.",
)
def valid(
    model_path: pathlib.Path,
    plate_config_file: pathlib.Path,
    annotations: pathlib.Path,
    batch_size: int,
    workers: int,
    use_multiprocessing: bool,
    max_queue_size: int,
) -> None:
    """
    Validate the trained OCR model on a labeled set.
    """
    config = load_plate_config_from_yaml(plate_config_file)
    model = utils.load_keras_model(
        model_path, vocab_size=config.vocabulary_size, max_plate_slots=config.max_plate_slots
    )
    val_dataset = PlateRecognitionPyDataset(
        annotations_file=annotations,
        config=config,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size,
    )
    model.evaluate(val_dataset)


if __name__ == "__main__":
    valid()
