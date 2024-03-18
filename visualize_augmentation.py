"""
Script for visualize the augmented plates used during training.
"""

import pathlib
import random
from math import ceil

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from fast_plate_ocr import utils
from fast_plate_ocr.augmentation import TRAIN_AUGMENTATION
from fast_plate_ocr.config import DEFAULT_IMG_HEIGHT, DEFAULT_IMG_WIDTH


def _set_seed(seed: int | None) -> None:
    """Set random seed for reproducing augmentations."""
    if seed:
        random.seed(seed)
        np.random.seed(seed)


def load_images(
    img_dir: pathlib.Path,
    num_images: int,
    shuffle: bool,
    height: int,
    width: int,
) -> tuple[list[npt.NDArray[np.uint8]], list[npt.NDArray[np.uint8]]]:
    images = utils.load_images_from_folder(
        img_dir, height=height, width=width, shuffle=shuffle, limit=num_images
    )
    augmented_images = [TRAIN_AUGMENTATION(image=i)["image"] for i in images]
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


@click.command()
@click.option(
    "--img-dir",
    "-d",
    type=click.Path(exists=True, dir_okay=True, path_type=pathlib.Path),
    default="assets/benchmark/imgs",
    help="Path to the images that will be augmented and visualized.",
)
@click.option(
    "--num-images",
    "-n",
    type=int,
    default=1_000,
    show_default=True,
    help="Maximum number of images to visualize.",
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
    "--height",
    "-h",
    type=int,
    default=DEFAULT_IMG_HEIGHT,
    show_default=True,
    help="Height to which the images will be resize.",
)
@click.option(
    "--width",
    "-w",
    type=int,
    default=DEFAULT_IMG_WIDTH,
    show_default=True,
    help="Width to which the images will be resize.",
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
    num_images: int,
    shuffle: bool,
    columns: int,
    rows: int,
    height: int,
    width: int,
    seed: int | None,
    show_original: bool,
) -> None:
    _set_seed(seed)
    images, augmented_images = load_images(img_dir, num_images, shuffle, height, width)
    display_images(images, augmented_images, columns, rows, show_original)


if __name__ == "__main__":
    # pylint: disable = no-value-for-parameter
    visualize_augmentation()
