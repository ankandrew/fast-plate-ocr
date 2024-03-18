"""
Script for validating trained OCR models.
"""

import pathlib

import click
import keras
from keras.src.activations import softmax
from torch.utils.data import DataLoader

from fast_plate_ocr.config import MAX_PLATE_SLOTS, MODEL_ALPHABET, PAD_CHAR

# Custom metris / losses
from fast_plate_ocr.custom import cat_acc, cce, plate_acc, top_3_k
from fast_plate_ocr.dataset import LicensePlateDataset


@click.command(context_settings={"max_content_width": 140})
@click.option(
    "-m",
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to the saved .keras model.",
)
@click.option(
    "-a",
    "--annotations",
    default="assets/benchmark/annotations.csv",
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
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
    pad_char: str,
) -> None:
    """Validate a model for a given annotated data."""
    custom_objects = {
        "cce": cce,
        "cat_acc": cat_acc,
        "plate_acc": plate_acc,
        "top_3_k": top_3_k,
        "softmax": softmax,
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
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
