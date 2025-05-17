"""
Script for training the License Plate OCR models.
"""

import json
import pathlib
import shutil
from datetime import datetime
from typing import Literal

import albumentations as A
import click
import keras
from keras.src.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    SwapEMAWeights,
    TensorBoard,
    TerminateOnNaN,
)
from keras.src.optimizers import AdamW

from fast_plate_ocr.cli.utils import print_params, print_train_details
from fast_plate_ocr.train.data.augmentation import TRAIN_AUGMENTATION
from fast_plate_ocr.train.data.dataset import PlateRecognitionPyDataset
from fast_plate_ocr.train.model.cct_model import create_cct_model
from fast_plate_ocr.train.model.config import load_config_from_yaml
from fast_plate_ocr.train.model.loss import cce_loss
from fast_plate_ocr.train.model.metric import (
    cat_acc_metric,
    plate_acc_metric,
    plate_len_acc_metric,
    top_3_k_metric,
)

# ruff: noqa: PLR0913
# pylint: disable=too-many-arguments,too-many-locals


EVAL_METRICS: dict[str, Literal["max", "min", "auto"]] = {
    "val_plate_acc": "max",
    "val_cat_acc": "max",
    "val_top_3_k_acc": "max",
    "val_loss": "min",
}
"""Eval metric to monitor."""


@click.command(context_settings={"max_content_width": 120})
@click.option(
    "--config-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Path pointing to the model license plate OCR config.",
)
@click.option(
    "--annotations",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Path pointing to the train annotations CSV file.",
)
@click.option(
    "--val-annotations",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Path pointing to the train validation CSV file.",
)
@click.option(
    "--augmentation-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="YAML file pointing to the augmentation pipeline saved with Albumentations.save(...)",
)
@click.option(
    "--lr",
    default=0.001,
    show_default=True,
    type=float,
    help="Initial learning rate.",
)
@click.option(
    "--final-lr-factor",
    default=1e-2,
    show_default=True,
    type=float,
    help="Final learning rate factor for the cosine decay scheduler. It's the fraction of"
    " the initial learning rate that remains after decay.",
)
@click.option(
    "--warmup-fraction",
    default=0.05,
    show_default=True,
    type=float,
    help="Fraction of total training steps to linearly warm up.",
)
@click.option(
    "--weight-decay",
    default=0.001,
    show_default=True,
    type=float,
    help="Weight decay for the AdamW optimizer.",
)
@click.option(
    "--clipnorm",
    default=1.0,
    show_default=True,
    type=float,
    help="Gradient clipping norm value for the AdamW optimizer.",
)
@click.option(
    "--label-smoothing",
    default=0.05,
    show_default=True,
    type=float,
    help="Amount of label smoothing to apply.",
)
@click.option(
    "--mixed-precision-policy",
    default=None,
    type=click.Choice(["mixed_float16", "mixed_bfloat16", "float32"]),
    help=(
        "Optional mixed precision policy for training. Choose one of: mixed_float16, "
        "mixed_bfloat16, or float32. If not provided, Keras uses its default global policy."
    ),
)
@click.option(
    "--batch-size",
    default=64,
    show_default=True,
    type=int,
    help="Batch size for training.",
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    type=int,
    help="Number of worker threads/processes for parallel data loading.",
)
@click.option(
    "--use-multiprocessing/--no-use-multiprocessing",
    default=False,
    show_default=True,
    help="Use multiprocessing for data loading.",
)
@click.option(
    "--max-queue-size",
    default=10,
    show_default=True,
    type=int,
    help="Maximum queue size for dataset workers.",
)
@click.option(
    "--output-dir",
    default="./trained_models",
    type=click.Path(dir_okay=True, file_okay=False, path_type=pathlib.Path),
    help="Output directory where model will be saved.",
)
@click.option(
    "--epochs",
    default=300,
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
    help="Stop training when the early stopping metric doesn't improve for X epochs.",
)
@click.option(
    "--early-stopping-metric",
    default="val_plate_acc",
    show_default=True,
    type=click.Choice(list(EVAL_METRICS), case_sensitive=False),
    help="Metric to monitor for early stopping.",
)
@click.option(
    "--weights-path",
    type=click.Path(exists=True, file_okay=True, path_type=pathlib.Path),
    help="Path to the pretrained model weights file.",
)
@click.option(
    "--use-ema/--no-use-ema",
    default=True,
    show_default=True,
    help=(
        "Whether to use exponential moving averages in the AdamW optimizer. "
        "Defaults to True; use --no-use-ema to disable."
    ),
)
@click.option(
    "--wd-ignore",
    default="bias,layer_norm",
    show_default=True,
    type=str,
    help="Comma‑separated list of variable‑name substrings to exclude from weight decay.",
)
@print_params(table_title="CLI Training Parameters", c1_title="Parameter", c2_title="Details")
def train(
    config_file: pathlib.Path,
    annotations: pathlib.Path,
    val_annotations: pathlib.Path,
    augmentation_path: pathlib.Path | None,
    lr: float,
    final_lr_factor: float,
    warmup_fraction: float,
    weight_decay: float,
    clipnorm: float,
    label_smoothing: float,
    mixed_precision_policy: str | None,
    batch_size: int,
    workers: int,
    use_multiprocessing: bool,
    max_queue_size: int,
    output_dir: pathlib.Path,
    epochs: int,
    tensorboard: bool,
    tensorboard_dir: pathlib.Path,
    early_stopping_patience: int,
    early_stopping_metric: str,
    weights_path: pathlib.Path | None,
    use_ema: bool,
    wd_ignore: str,
) -> None:
    """
    Train the License Plate OCR model.
    """
    if mixed_precision_policy is not None:
        keras.mixed_precision.set_global_policy(mixed_precision_policy)

    train_augmentation = (
        A.load(augmentation_path, data_format="yaml") if augmentation_path else TRAIN_AUGMENTATION
    )
    config = load_config_from_yaml(config_file)
    print_train_details(train_augmentation, config.model_dump())

    train_dataset = PlateRecognitionPyDataset(
        annotations_file=annotations,
        transform=train_augmentation,
        config=config,
        batch_size=batch_size,
        shuffle=True,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size,
    )

    val_dataset = PlateRecognitionPyDataset(
        annotations_file=val_annotations,
        config=config,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size,
    )

    # Train
    model = create_cct_model(
        input_shape=(config.img_height, config.img_width, 1),
        max_plate_slots=config.max_plate_slots,
        vocabulary_size=config.vocabulary_size,
        transformer_layers=3,
        conv_layers=3,
        num_output_channels=[32, 64, 96],
        num_heads=3,
        projection_dim=96,
        transformer_units=[96, 96],
        positional_emb=True,
    )

    if weights_path:
        model.load_weights(weights_path, skip_mismatch=True)

    total_steps = epochs * len(train_dataset)
    warmup_steps = int(warmup_fraction * total_steps)

    cosine_decay = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0 if warmup_steps > 0 else lr,
        decay_steps=total_steps,
        alpha=final_lr_factor,
        warmup_steps=warmup_steps,
        warmup_target=lr if warmup_steps > 0 else None,
    )

    optimizer = AdamW(cosine_decay, weight_decay=weight_decay, clipnorm=clipnorm, use_ema=use_ema)
    optimizer.exclude_from_weight_decay(
        var_names=[name.strip() for name in wd_ignore.split(",") if name.strip()]
    )

    model.compile(
        loss=cce_loss(vocabulary_size=config.vocabulary_size, label_smoothing=label_smoothing),
        jit_compile=False,
        optimizer=optimizer,
        metrics=[
            cat_acc_metric(
                max_plate_slots=config.max_plate_slots, vocabulary_size=config.vocabulary_size
            ),
            plate_acc_metric(
                max_plate_slots=config.max_plate_slots, vocabulary_size=config.vocabulary_size
            ),
            top_3_k_metric(vocabulary_size=config.vocabulary_size),
            plate_len_acc_metric(
                max_plate_slots=config.max_plate_slots,
                vocabulary_size=config.vocabulary_size,
                pad_token_index=config.pad_idx,
            ),
        ],
    )

    output_dir /= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save params and config used for training
    shutil.copy(config_file, output_dir / "config.yaml")
    A.save(train_augmentation, output_dir / "train_augmentation.yaml", "yaml")
    with open(output_dir / "hyper_params.json", "w", encoding="utf-8") as f_out:
        json.dump(
            {k: v for k, v in locals().items() if k in click.get_current_context().params},
            f_out,
            indent=4,
            default=str,
        )

    callbacks = [
        # Stop training when early_stopping_metric doesn't improve for X epochs
        EarlyStopping(
            monitor=early_stopping_metric,
            patience=early_stopping_patience,
            mode=EVAL_METRICS[early_stopping_metric],
            restore_best_weights=False,
            verbose=1,
        ),
        # To save model checkpoint with EMA weights, we need to place this before `ModelCheckpoint`
        *([SwapEMAWeights(swap_on_epoch=True)] if use_ema else []),
        # We don't use EarlyStopping restore_best_weights=True because it won't restore the best
        # weights when it didn't manage to EarlyStop but finished all epochs
        ModelCheckpoint(output_dir / "last.keras", save_weights_only=False, save_best_only=False),
        ModelCheckpoint(
            output_dir / "best.keras",
            monitor=early_stopping_metric,
            mode=EVAL_METRICS[early_stopping_metric],
            save_best_only=True,
            verbose=1,
        ),
        TerminateOnNaN(),
    ]

    if tensorboard:
        run_dir = tensorboard_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(TensorBoard(log_dir=run_dir))

    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)


if __name__ == "__main__":
    train()
