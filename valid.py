"""
Script for validating trained OCR models.
"""

import pathlib

import click
from torch.utils.data import DataLoader

from fast_plate_ocr import utils
from fast_plate_ocr.config import MAX_PLATE_SLOTS, MODEL_ALPHABET, PAD_CHAR, VOCABULARY_SIZE

# Custom metris / losses
from fast_plate_ocr.dataset import LicensePlateDataset


@click.command(context_settings={"max_content_width": 140})
@click.option(
    "-m",
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Path to the saved .keras model.",
)
@click.option(
    "-a",
    "--annotations",
    default="assets/benchmark/annotations.csv",
    show_default=True,
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
    "--plate-slots",
    default=MAX_PLATE_SLOTS,
    show_default=True,
    type=int,
    help="Max number of plate slots supported. Plates with less slots will be padded.",
)
@click.option(
    "--alphabet",
    default=MODEL_ALPHABET,
    show_default=True,
    type=str,
    help="Model vocabulary. This must include the padding symbol.",
)
@click.option(
    "--vocab-size",
    default=VOCABULARY_SIZE,
    show_default=True,
    type=int,
    help="Size of the vocabulary. This should match '--alphabet' length.",
)
@click.option(
    "--pad-char",
    default=PAD_CHAR,
    show_default=True,
    type=str,
    help="Padding char for plates with length less than '--plate-slots'.",
)
def valid(
    model_path: pathlib.Path,
    annotations: pathlib.Path,
    batch_size: int,
    plate_slots: int,
    alphabet: str,
    vocab_size: int,
    pad_char: str,
) -> None:
    """Validate a model for a given annotated data."""
    model = utils.load_keras_model(model_path, vocab_size=vocab_size, max_plate_slots=plate_slots)
    val_torch_dataset = LicensePlateDataset(
        annotations_file=annotations,
        max_plate_slots=plate_slots,
        alphabet=alphabet,
        pad_char=pad_char,
    )
    val_dataloader = DataLoader(val_torch_dataset, batch_size=batch_size, shuffle=False)
    model.evaluate(val_dataloader)


if __name__ == "__main__":
    valid()
