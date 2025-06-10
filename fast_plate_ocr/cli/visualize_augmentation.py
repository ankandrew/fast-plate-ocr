"""
Script to visualize the augmented plates used during training.
"""

import pathlib
import random
from math import ceil

import albumentations as A
import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from fast_plate_ocr.train.data.augmentation import (
    default_augmentation,
)
from fast_plate_ocr.train.model.config import PlateOCRConfig, load_plate_config_from_yaml
from fast_plate_ocr.train.utilities import utils


def _set_seed(seed: int | None) -> None:
    """Set random seed for reproducing augmentations."""
    if seed:
        random.seed(seed)
        np.random.seed(seed)


def load_images(
    img_dir: pathlib.Path,
    num_images: int,
    shuffle: bool,
    plate_config: PlateOCRConfig,
    augmentation: A.Compose,
) -> tuple[list[npt.NDArray[np.uint8]], list[npt.NDArray[np.uint8]]]:
    images = list(
        utils.load_images_from_folder(
            img_dir,
            width=plate_config.img_width,
            height=plate_config.img_height,
            image_color_mode=plate_config.image_color_mode,
            keep_aspect_ratio=plate_config.keep_aspect_ratio,
            interpolation_method=plate_config.interpolation,
            padding_color=plate_config.padding_color,
            shuffle=shuffle,
            limit=num_images,
        )
    )
    augmented_images = [augmentation(image=i)["image"] for i in images]
    return images, augmented_images


def display_images(
    images: list[npt.NDArray[np.uint8]],
    augmented_images: list[npt.NDArray[np.uint8]],
    columns: int,
    rows: int,
    show_original: bool,
) -> None:
    num_images = len(images)
    total_plots = rows * columns
    num_pages = ceil(num_images / total_plots)
    for page in range(num_pages):
        _, axs = plt.subplots(rows, columns, figsize=(8, 8))
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            idx = page * total_plots + i
            if idx < num_images:
                if show_original:
                    img_to_show = np.concatenate((images[idx], augmented_images[idx]), axis=1)
                else:
                    img_to_show = augmented_images[idx]
                ax.imshow(img_to_show, cmap="gray")
                ax.axis("off")
            else:
                ax.axis("off")
        plt.tight_layout()
        plt.show()


# ruff: noqa: PLR0913
# pylint: disable=too-many-arguments,too-many-locals


@click.command(context_settings={"max_content_width": 120})
@click.option(
    "--img-dir",
    "-d",
    required=True,
    type=click.Path(exists=True, dir_okay=True, path_type=pathlib.Path),
    help="Path to the images that will be augmented and visualized.",
)
@click.option(
    "--plate-config-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, path_type=pathlib.Path),
    help="Path pointing to the model license plate OCR config.",
)
@click.option(
    "--num-images",
    "-n",
    type=int,
    default=250,
    show_default=True,
    help="Maximum number of images to visualize.",
)
@click.option(
    "--augmentation-path",
    type=click.Path(exists=True, file_okay=True, path_type=pathlib.Path),
    help="YAML file pointing to the augmentation pipeline saved with Albumentations.save(...)",
)
@click.option(
    "--shuffle",
    "-s",
    is_flag=True,
    default=False,
    help="Whether to shuffle the images before plotting them.",
)
@click.option(
    "--columns",
    "-c",
    type=int,
    default=3,
    show_default=True,
    help="Number of columns in the grid layout for displaying images.",
)
@click.option(
    "--rows",
    "-r",
    type=int,
    default=4,
    show_default=True,
    help="Number of rows in the grid layout for displaying images.",
)
@click.option(
    "--show-original",
    "-o",
    is_flag=True,
    help="Show the original image along with the augmented one.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed for reproducing augmentations.",
)
def visualize_augmentation(
    img_dir: pathlib.Path,
    plate_config_file: pathlib.Path,
    num_images: int,
    augmentation_path: pathlib.Path | None,
    shuffle: bool,
    columns: int,
    rows: int,
    seed: int | None,
    show_original: bool,
) -> None:
    """
    Visualize augmentation pipeline applied to raw images.
    """
    _set_seed(seed)
    config = load_plate_config_from_yaml(plate_config_file)
    aug = (
        A.load(augmentation_path, data_format="yaml")
        if augmentation_path
        else default_augmentation(config.image_color_mode)
    )
    aug.set_random_seed(seed)
    images, augmented_images = load_images(img_dir, num_images, shuffle, config, aug)
    display_images(images, augmented_images, columns, rows, show_original)


if __name__ == "__main__":
    # pylint: disable = no-value-for-parameter
    visualize_augmentation()
