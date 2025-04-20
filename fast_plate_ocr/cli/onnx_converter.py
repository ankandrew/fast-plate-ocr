"""
Script for converting Keras models to ONNX format.
"""

import logging
import pathlib
import shutil
from tempfile import NamedTemporaryFile

import click
import keras
import numpy as np
import onnx
import onnxruntime as rt
import onnxsim

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
    "--simplify/--no-simplify",
    default=False,
    show_default=True,
    help="Simplify ONNX model using onnxsim.",
)
@click.option(
    "--config-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Path pointing to the model license plate OCR config.",
)
@click.option(
    "--save-dir",
    required=False,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    help="Custom dir to save the ONNX model. By default same directory from the model is used.",
)
def export_onnx(
    model_path: pathlib.Path,
    simplify: bool,
    config_file: pathlib.Path,
    save_dir: pathlib.Path,
) -> None:
    """
    Export Keras models to ONNX format.
    """
    onnx_output_file = model_path.with_suffix(".onnx")
    onnx_output_file = onnx_output_file if save_dir is None else save_dir / onnx_output_file.name

    if onnx_output_file.exists():
        logging.info("Overwriting existing ONNX file at %s", onnx_output_file)
        onnx_output_file.unlink()

    config = load_config_from_yaml(config_file)
    model = load_keras_model(
        model_path,
        vocab_size=config.vocabulary_size,
        max_plate_slots=config.max_plate_slots,
    )
    spec = [
        keras.InputSpec(
            name="input", shape=(None, config.img_height, config.img_width, 1), dtype="uint8"
        )
    ]
    # Convert from Keras to ONNX
    with NamedTemporaryFile(suffix=".onnx") as tmp:
        tmp_onnx = tmp.name
        model.export(tmp_onnx, format="onnx", verbose=False, input_signature=spec)
        if simplify:
            logging.info("Simplifying ONNX model ...")
            model_simp, check = onnxsim.simplify(onnx.load(tmp_onnx))
            assert check, "Simplified ONNX model could not be validated!"
            onnx.save(model_simp, onnx_output_file)
        else:
            shutil.copy(tmp_onnx, onnx_output_file)
    output_names = [o.name for o in onnx.load(onnx_output_file).graph.output]
    x = np.random.randint(0, 256, size=(1, config.img_height, config.img_width, 1), dtype=np.uint8)
    # Run dummy inference and log time taken
    m = rt.InferenceSession(onnx_output_file)
    with log_time_taken("ONNX inference took:"):
        onnx_pred = m.run(output_names, {"input": x})
    # Check if ONNX and keras have the same results
    if not np.allclose(model.predict(x, verbose=0), onnx_pred[0], rtol=1e-5, atol=1e-5):
        logging.warning("ONNX model output was not close to Keras model for the given tolerance!")
    logging.info("Model converted to ONNX! Saved at %s", onnx_output_file)


if __name__ == "__main__":
    export_onnx()
