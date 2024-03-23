"""
Script for displaying an image with the OCR model predictions.
"""

import logging
import pathlib
from contextlib import nullcontext

import click
import cv2
import keras
import numpy as np
import numpy.typing as npt

from fast_plate_ocr import utils
from fast_plate_ocr.config import (
    DEFAULT_IMG_HEIGHT,
    DEFAULT_IMG_WIDTH,
    MAX_PLATE_SLOTS,
    MODEL_ALPHABET,
    VOCABULARY_SIZE,
)

logging.basicConfig(level=logging.INFO)


def check_low_conf(probs, thresh=0.3):
    """
    Add position of chars. that are < thresh
    """
    return [i for i, prob in enumerate(probs) if prob < thresh]


def display_predictions(
    image: npt.NDArray,
    prediction: npt.NDArray,
    alphabet: str = MODEL_ALPHABET,
    plate_slots: int = MAX_PLATE_SLOTS,
    vocab_size: int = VOCABULARY_SIZE,
) -> None:
    """
    Display plate and corresponding prediction.
    """
    plate, probs = utils.postprocess_model_output(
        prediction=prediction,
        alphabet=alphabet,
        max_plate_slots=plate_slots,
        vocab_size=vocab_size,
    )
    plate_str = "".join(plate)
    logging.info("Plate: %s", plate_str)
    logging.info("Confidence: %s", probs)
    image_to_show = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    # Converting to BGR for color text
    image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_GRAY2RGB)
    # Average probabilities
    avg_prob = np.mean(probs) * 100
    cv2.putText(
        image_to_show,
        f"{plate_str}  {avg_prob:.{2}f}%",
        org=(5, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 0),
        lineType=1,
        thickness=6,
    )
    cv2.putText(
        image_to_show,
        f"{plate_str}  {avg_prob:.{2}f}%",
        org=(5, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        lineType=1,
        thickness=2,
    )
    # Display character with low confidence
    low_conf_chars = "Low conf. on: " + " ".join(
        [plate[i] for i in check_low_conf(probs, thresh=0.15)]
    )
    cv2.putText(
        image_to_show,
        low_conf_chars,
        org=(5, 200),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=(0, 0, 220),
        lineType=1,
        thickness=2,
    )
    cv2.imshow("plates", image_to_show)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        return


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
    "-d",
    "--img-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=pathlib.Path),
    default="assets/benchmark/imgs",
    show_default=True,
    help="Directory containing the images to make predictions from.",
)
@click.option(
    "-t",
    "--time",
    default=True,
    is_flag=True,
    help="Log time taken to run predictions.",
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
def visualize_predictions(
    model_path: pathlib.Path,
    img_dir: pathlib.Path,
    height: int,
    width: int,
    plate_slots: int,
    alphabet: str,
    vocab_size: int,
    time: bool,
):
    model = utils.load_keras_model(model_path, vocab_size=vocab_size, max_plate_slots=plate_slots)
    images = utils.load_images_from_folder(img_dir, width=width, height=height)
    for image in images:
        with utils.log_time_taken("Prediction time") if time else nullcontext():
            x = np.expand_dims(image, 0)
            prediction = model(x, training=False)
            prediction = keras.ops.stop_gradient(prediction).numpy()
        display_predictions(
            image=image,
            prediction=prediction,
            alphabet=alphabet,
            plate_slots=plate_slots,
            vocab_size=vocab_size,
        )
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_predictions()
