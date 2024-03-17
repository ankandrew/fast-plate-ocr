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

from augmentation import TRAIN_AUGMENTATION
from fast_plate_ocr.config import DEFAULT_IMG_HEIGHT, DEFAULT_IMG_WIDTH
from fast_plate_ocr.utils import read_plate_image


def load_images(
    img_dir: pathlib.Path,
    num_images: int,
    shuffle_img: bool,
    height: int,
    width: int,
) -> list[npt.NDArray[np.uint8]]:
    img_paths = sorted(f for f in img_dir.iterdir() if f.is_file())
    img_paths = img_paths[:num_images]
    if shuffle_img:
        random.shuffle(img_paths)
    img_data = [
        TRAIN_AUGMENTATION(
            image=read_plate_image(image_path=str(img), img_height=height, img_width=width)
        )["image"]
        for img in img_paths
    ]
    return img_data


def display_images(images: list[npt.NDArray[np.uint8]], columns: int, rows: int) -> None:
    num_images = len(images)
    total_plots = rows * columns
    num_pages = ceil(num_images / total_plots)
    for page in range(num_pages):
        _, axs = plt.subplots(rows, columns, figsize=(8, 8))
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            idx = page * total_plots + i
            if idx < num_images:
                ax.imshow(images[idx], cmap="gray")
                ax.axis("off")
            else:
                ax.axis("off")
        plt.tight_layout()
        plt.show()


@click.command()
@click.option(
    "--img-dir",
    "-d",
    type=click.Path(exists=True, dir_okay=True, path_type=pathlib.Path),
    default="assets/benchmark/imgs",
    help="Path to the images that will be augmented and visualized",
)
@click.option(
    "--num-images",
    "-n",
    type=int,
    default=1_000,
    show_default=True,
    help="Maximum number of images to visualize",
)
@click.option(
    "--shuffle_img",
    is_flag=True,
    default=False,
    help="Whether to shuffle the images before plotting them",
)
@click.option(
    "--columns",
    "-c",
    type=int,
    default=4,
    show_default=True,
    help="Number of columns in the grid layout for displaying images",
)
@click.option(
    "--rows",
    "-r",
    type=int,
    default=4,
    show_default=True,
    help="Number of rows in the grid layout for displaying images",
)
@click.option(
    "--height",
    "-h",
    type=int,
    default=DEFAULT_IMG_HEIGHT,
    show_default=True,
    help="Height to which the images will be resize",
)
@click.option(
    "--width",
    "-w",
    type=int,
    default=DEFAULT_IMG_WIDTH,
    show_default=True,
    help="Width to which the images will be resize",
)
def visualize_augmentation(
    img_dir: pathlib.Path,
    num_images: int,
    shuffle_img: bool,
    columns: int,
    rows: int,
    height: int,
    width: int,
) -> None:
    augmented_images = load_images(img_dir, num_images, shuffle_img, height, width)
    display_images(augmented_images, columns, rows)


if __name__ == "__main__":
    # pylint: disable = no-value-for-parameter
    visualize_augmentation()
