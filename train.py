"""
Script for training the License Plate OCR models.
"""

import os
import pathlib

import click
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from torch.utils.data import DataLoader

from fast_plate_ocr.augmentation import TRAIN_AUGMENTATION
from fast_plate_ocr.config import load_config_from_yaml
from fast_plate_ocr.custom import cat_acc_metric, cce_loss, plate_acc_metric, top_3_k_metric
from fast_plate_ocr.dataset import LicensePlateDataset
from fast_plate_ocr.models import modelo_1m_cpu, modelo_2m

# ruff: noqa: PLR0913
# pylint: disable=too-many-arguments,too-many-locals


@click.command(context_settings={"max_content_width": 140})
@click.option(
    "--model-type",
    type=click.Choice(["1m_cpu", "2m"]),
    help="Type of model to train. See fast_plate_ocr/models.py module.",
)
@click.option(
    "--dense/--no-dense",
    default=True,
    show_default=True,
    help="Whether to use Fully Connected layers in model head or not.",
)
@click.option(
    "--config-file",
    default="./config/arg_plates.yaml",
    show_default=True,
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
    "--output-dir",
    default=None,
    type=str,
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
    default="logs",
    show_default=True,
    type=str,
    help="The path of the directory where to save the TensorBoard log files.",
)
@click.option(
    "--early-stopping-patience",
    default=120,
    show_default=True,
    type=int,
    help="Stop training when 'val_plate_acc' doesn't improve for X epochs.",
)
@click.option(
    "--reduce-lr-patience",
    default=100,
    show_default=True,
    type=int,
    help="Reduce the learning rate by 0.5x if 'val_plate_acc' doesn't improve within X epochs.",
)
def train(
    model_type: str,
    dense: bool,
    config_file: pathlib.Path,
    annotations: pathlib.Path,
    val_annotations: pathlib.Path,
    lr: float,
    batch_size: int,
    output_dir: str,
    epochs: int,
    tensorboard: bool,
    tensorboard_dir: str,
    early_stopping_patience: int,
    reduce_lr_patience: int,
) -> None:
    config = load_config_from_yaml(config_file)
    train_torch_dataset = LicensePlateDataset(
        annotations_file=annotations,
        transform=TRAIN_AUGMENTATION,
        config=config,
    )
    train_dataloader = DataLoader(train_torch_dataset, batch_size=batch_size, shuffle=True)

    if val_annotations:
        val_torch_dataset = LicensePlateDataset(
            annotations_file=val_annotations,
            config=config,
        )
        val_dataloader = DataLoader(val_torch_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_dataloader = None

    # Train
    if model_type == "1m_cpu":
        model = modelo_1m_cpu(
            h=config.img_height,
            w=config.img_width,
            dense=dense,
            max_plate_slots=config.max_plate_slots,
            vocabulary_size=config.vocabulary_size,
        )
    else:
        model = modelo_2m(
            h=config.img_height,
            w=config.img_width,
            dense=dense,
            max_plate_slots=config.max_plate_slots,
            vocabulary_size=config.vocabulary_size,
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

    callbacks = [
        # Reduce the learning rate by 0.5x if 'val_plate_acc' doesn't improve within X epochs
        ReduceLROnPlateau(
            "val_plate_acc",
            verbose=1,
            patience=reduce_lr_patience,
            factor=0.5,
            min_lr=1e-5,
        ),
        # Stop training when 'val_plate_acc' doesn't improve for X epochs
        EarlyStopping(
            monitor="val_plate_acc",
            patience=early_stopping_patience,
            mode="max",
            restore_best_weights=True,
        ),
    ]

    if tensorboard:
        callbacks.append(TensorBoard(log_dir=tensorboard_dir))

    history = model.fit(
        train_dataloader, epochs=epochs, validation_data=val_dataloader, callbacks=callbacks
    )

    best_vpa = max(history.history["val_plate_acc"])
    epochs = len(history.epoch)
    model_name = f"cnn-ocr_{best_vpa:.4}-vpa_epochs-{epochs}"
    # Make dir for trained model
    if output_dir is None:
        model_folder = f"./trained/{model_name}"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        output_path = model_folder
    else:
        output_path = output_dir
    model.save(os.path.join(output_path, f"{model_name}.keras"))


if __name__ == "__main__":
    train()
