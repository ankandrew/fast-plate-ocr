"""
Script for training the License Plate OCR models.
"""

import pathlib
import shutil
from datetime import datetime
from typing import Literal

import albumentations as A
import click
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.src.optimizers import Adam
from torch.utils.data import DataLoader

from fast_plate_ocr.cli.utils import print_params, print_train_details
from fast_plate_ocr.train.data.augmentation import TRAIN_AUGMENTATION
from fast_plate_ocr.train.data.dataset import LicensePlateDataset
from fast_plate_ocr.train.model.config import load_config_from_yaml
from fast_plate_ocr.train.model.custom import (
    cat_acc_metric,
    cce_loss,
    plate_acc_metric,
    top_3_k_metric,
)
from fast_plate_ocr.train.model.models import cnn_ocr_model

# ruff: noqa: PLR0913
# pylint: disable=too-many-arguments,too-many-locals


@click.command(context_settings={"max_content_width": 120})
@click.option(
    "--dense/--no-dense",
    default=True,
    show_default=True,
    help="Whether to use Fully Connected layers in model head or not.",
)
@click.option(
    "--config-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, path_type=pathlib.Path),
    help="Path pointing to the model license plate OCR config.",
)
@click.option(
    "--annotations",
    required=True,
    type=click.Path(exists=True, file_okay=True, path_type=pathlib.Path),
    help="Path pointing to the train annotations CSV file.",
)
@click.option(
    "--val-annotations",
    required=True,
    type=click.Path(exists=True, file_okay=True, path_type=pathlib.Path),
    help="Path pointing to the train validation CSV file.",
)
@click.option(
    "--augmentation-path",
    type=click.Path(exists=True, file_okay=True, path_type=pathlib.Path),
    help="YAML file pointing to the augmentation pipeline saved with Albumentations.save(...)",
)
@click.option(
    "--lr",
    default=1e-3,
    show_default=True,
    type=float,
    help="Initial learning rate to use.",
)
@click.option(
    "--batch-size",
    default=128,
    show_default=True,
    type=int,
    help="Batch size for training.",
)
@click.option(
    "--num-workers",
    default=0,
    show_default=True,
    type=int,
    help="How many subprocesses to load data, used in the torch DataLoader.",
)
@click.option(
    "--output-dir",
    default="./trained_models",
    type=click.Path(dir_okay=True, path_type=pathlib.Path),
    help="Output directory where model will be saved.",
)
@click.option(
    "--epochs",
    default=500,
    show_default=True,
    type=int,
    help="Number of training epochs.",
)
@click.option(
    "--tensorboard",
    "-t",
    is_flag=True,
    help="Whether to use TensorBoard visualization tool.",
)
@click.option(
    "--tensorboard-dir",
    "-l",
    default="tensorboard_logs",
    show_default=True,
    type=click.Path(path_type=pathlib.Path),
    help="The path of the directory where to save the TensorBoard log files.",
)
@click.option(
    "--early-stopping-patience",
    default=100,
    show_default=True,
    type=int,
    help="Stop training when 'val_plate_acc' doesn't improve for X epochs.",
)
@click.option(
    "--reduce-lr-patience",
    default=60,
    show_default=True,
    type=int,
    help="Patience to reduce the learning rate if 'val_plate_acc' doesn't improve within X epochs.",
)
@click.option(
    "--reduce-lr-factor",
    default=0.85,
    show_default=True,
    type=float,
    help="Reduce the learning rate by this factor when 'val_plate_acc' doesn't improve.",
)
@click.option(
    "--activation",
    default="relu",
    show_default=True,
    type=str,
    help="Activation function to use.",
)
@click.option(
    "--pool-layer",
    default="max",
    show_default=True,
    type=click.Choice(["max", "avg"]),
    help="Choose the pooling layer to use.",
)
@print_params(table_title="CLI Training Parameters", c1_title="Parameter", c2_title="Details")
def train(
    dense: bool,
    config_file: pathlib.Path,
    annotations: pathlib.Path,
    val_annotations: pathlib.Path,
    augmentation_path: pathlib.Path | None,
    lr: float,
    batch_size: int,
    num_workers: int,
    output_dir: pathlib.Path,
    epochs: int,
    tensorboard: bool,
    tensorboard_dir: pathlib.Path,
    early_stopping_patience: int,
    reduce_lr_patience: int,
    reduce_lr_factor: float,
    activation: str,
    pool_layer: Literal["max", "avg"],
) -> None:
    """
    Train the License Plate OCR model.
    """
    train_augmentation = (
        A.load(augmentation_path, data_format="yaml") if augmentation_path else TRAIN_AUGMENTATION
    )
    config = load_config_from_yaml(config_file)
    print_train_details(train_augmentation, config.model_dump())
    train_torch_dataset = LicensePlateDataset(
        annotations_file=annotations,
        transform=train_augmentation,
        config=config,
    )
    train_dataloader = DataLoader(
        train_torch_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    if val_annotations:
        val_torch_dataset = LicensePlateDataset(
            annotations_file=val_annotations,
            config=config,
        )
        val_dataloader = DataLoader(
            val_torch_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
    else:
        val_dataloader = None

    # Train
    model = cnn_ocr_model(
        h=config.img_height,
        w=config.img_width,
        dense=dense,
        max_plate_slots=config.max_plate_slots,
        vocabulary_size=config.vocabulary_size,
        activation=activation,
        pool_layer=pool_layer,
    )
    model.compile(
        loss=cce_loss(vocabulary_size=config.vocabulary_size),
        optimizer=Adam(lr),
        metrics=[
            cat_acc_metric(
                max_plate_slots=config.max_plate_slots, vocabulary_size=config.vocabulary_size
            ),
            plate_acc_metric(
                max_plate_slots=config.max_plate_slots, vocabulary_size=config.vocabulary_size
            ),
            top_3_k_metric(vocabulary_size=config.vocabulary_size),
        ],
    )

    output_dir /= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file_path = output_dir / "cnn_ocr-epoch_{epoch:02d}-acc_{val_plate_acc:.3f}.keras"

    # Save params and config used for training
    shutil.copy(config_file, output_dir / "config.yaml")
    A.save(train_augmentation, output_dir / "train_augmentation.yaml", "yaml")

    callbacks = [
        # Reduce the learning rate by 0.5x if 'val_plate_acc' doesn't improve within X epochs
        ReduceLROnPlateau(
            "val_plate_acc",
            patience=reduce_lr_patience,
            factor=reduce_lr_factor,
            min_lr=1e-6,
            verbose=1,
        ),
        # Stop training when 'val_plate_acc' doesn't improve for X epochs
        EarlyStopping(
            monitor="val_plate_acc",
            patience=early_stopping_patience,
            mode="max",
            restore_best_weights=False,
            verbose=1,
        ),
        # We don't use EarlyStopping restore_best_weights=True because it won't restore the best
        # weights when it didn't manage to EarlyStop but finished all epochs
        ModelCheckpoint(
            model_file_path,
            monitor="val_plate_acc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    if tensorboard:
        run_dir = tensorboard_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(TensorBoard(log_dir=run_dir))

    model.fit(train_dataloader, epochs=epochs, validation_data=val_dataloader, callbacks=callbacks)


if __name__ == "__main__":
    train()
