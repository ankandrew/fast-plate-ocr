"""
Script for displaying an image with the OCR model predictions.
"""

import logging
import pathlib

import click
import cv2
import keras
import numpy as np

from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.utilities import utils
from fast_plate_ocr.train.utilities.utils import postprocess_model_output

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


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
    "-d",
    "--img-dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=pathlib.Path),
    help="Directory containing the images to make predictions from.",
)
@click.option(
    "-l",
    "--low-conf-thresh",
    type=float,
    default=0.35,
    show_default=True,
    help="Threshold for displaying low confidence characters.",
)
@click.option(
    "-f",
    "--filter-conf",
    type=float,
    help="Display plates that any of the plate characters are below this number.",
)
def visualize_predictions(
    model_path: pathlib.Path,
    plate_config_file: pathlib.Path,
    img_dir: pathlib.Path,
    low_conf_thresh: float,
    filter_conf: float | None,
):
    """
    Visualize OCR model predictions on unlabeled data.
    """
    plate_config = load_plate_config_from_yaml(plate_config_file)
    model = utils.load_keras_model(model_path, plate_config)
    images = utils.load_images_from_folder(
        img_dir,
        width=plate_config.img_width,
        height=plate_config.img_height,
        image_color_mode=plate_config.image_color_mode,
        keep_aspect_ratio=plate_config.keep_aspect_ratio,
        interpolation_method=plate_config.interpolation,
        padding_color=plate_config.padding_color,
    )
    for image in images:
        x = np.expand_dims(image, 0)
        prediction = model(x, training=False)
        prediction = keras.ops.stop_gradient(prediction).numpy()
        plate, probs = postprocess_model_output(
            prediction=prediction,
            alphabet=plate_config.alphabet,
            max_plate_slots=plate_config.max_plate_slots,
            vocab_size=plate_config.vocabulary_size,
        )
        if not filter_conf or (filter_conf and np.any(probs < filter_conf)):
            utils.display_predictions(
                image=image, plate=plate, probs=probs, low_conf_thresh=low_conf_thresh
            )
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_predictions()
