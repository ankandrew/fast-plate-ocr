"""
Script for converting Keras models to ONNX format.
"""

import logging
import pathlib
import shutil
from tempfile import NamedTemporaryFile

import click
import numpy as np
import onnx
import onnxruntime as rt
import onnxsim
import tensorflow as tf
import tf2onnx
from tf2onnx import constants as tf2onnx_constants

from fast_plate_ocr.common.utils import log_time_taken
from fast_plate_ocr.train.model.config import load_config_from_yaml
from fast_plate_ocr.train.utilities.utils import load_keras_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


# pylint: disable=too-many-arguments,too-many-locals


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
    "--output-path",
    required=True,
    type=str,
    help="Output name for ONNX model.",
)
@click.option(
    "--simplify/--no-simplify",
    default=False,
    show_default=True,
    help="Simplify ONNX model using onnxsim.",
)
@click.option(
    "--config-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, path_type=pathlib.Path),
    help="Path pointing to the model license plate OCR config.",
)
@click.option(
    "--opset",
    default=16,
    type=click.IntRange(max=max(tf2onnx_constants.OPSET_TO_IR_VERSION)),
    show_default=True,
    help="Opset version for ONNX.",
)
def export_onnx(
    model_path: pathlib.Path,
    output_path: str,
    simplify: bool,
    config_file: pathlib.Path,
    opset: int,
) -> None:
    """
    Export Keras models to ONNX format.
    """
    config = load_config_from_yaml(config_file)
    model = load_keras_model(
        model_path,
        vocab_size=config.vocabulary_size,
        max_plate_slots=config.max_plate_slots,
    )
    spec = (tf.TensorSpec((None, config.img_height, config.img_width, 1), tf.uint8, name="input"),)
    # Convert from Keras to ONNX using tf2onnx library
    with NamedTemporaryFile(suffix=".onnx") as tmp:
        tmp_onnx = tmp.name
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=opset,
            output_path=tmp_onnx,
        )
        if simplify:
            logging.info("Simplifying ONNX model ...")
            model_simp, check = onnxsim.simplify(onnx.load(tmp_onnx))
            assert check, "Simplified ONNX model could not be validated!"
            onnx.save(model_simp, output_path)
        else:
            shutil.copy(tmp_onnx, output_path)
    output_names = [n.name for n in model_proto.graph.output]
    x = np.random.randint(0, 256, size=(1, config.img_height, config.img_width, 1), dtype=np.uint8)
    # Run dummy inference and log time taken
    m = rt.InferenceSession(output_path)
    with log_time_taken("ONNX inference took:"):
        onnx_pred = m.run(output_names, {"input": x})
    # Check if ONNX and keras have the same results
    if not np.allclose(model.predict(x, verbose=0), onnx_pred[0], rtol=1e-5, atol=1e-5):
        logging.warning("ONNX model output was not close to Keras model for the given tolerance!")
    logging.info("Model converted to ONNX! Saved at %s", output_path)


if __name__ == "__main__":
    export_onnx()
