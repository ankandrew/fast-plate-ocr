"""
Script for exporting the trained Keras models to other formats.
"""

import logging
import pathlib
import shutil
from tempfile import NamedTemporaryFile, TemporaryDirectory

import click
import keras
import numpy as np
from numpy.typing import DTypeLike

from fast_plate_ocr.cli.utils import requires
from fast_plate_ocr.core.types import TensorDataFormat
from fast_plate_ocr.core.utils import log_time_taken
from fast_plate_ocr.train.model.config import (
    PlateOCRConfig,
    load_plate_config_from_yaml,
)
from fast_plate_ocr.train.utilities.utils import load_keras_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


# pylint: disable=too-many-arguments,too-many-locals,import-outside-toplevel


def _dummy_input(b: int, h: int, w: int, n_c: int, dtype: DTypeLike = np.uint8) -> np.ndarray:
    """Random tensor in [0, 255] shaped (b, h, w, 1)."""
    return np.random.randint(0, 256, size=(b, h, w, n_c)).astype(dtype)


def _validate_prediction(
    keras_model: keras.Model,
    exported_predict,
    x: np.ndarray,
    target: str,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """Compare Keras and exported backend on a single forward pass."""
    keras_out = keras_model.predict(x, verbose=0)
    exported_out = exported_predict(x)
    if not np.allclose(keras_out, exported_out, rtol=rtol, atol=atol):
        logging.warning("%s output deviates from Keras beyond tolerance.", target.upper())
    else:
        logging.info("%s output matches Keras ✔", target.upper())


def _make_output_path(
    model_path: pathlib.Path, save_dir: pathlib.Path | None, new_ext: str
) -> pathlib.Path:
    """
    Build an output filename next to the model or inside --save-dir.

    Note: If the file already exists we delete it.

    :param model_path: Path to the model file.
    :param save_dir: Directory to save the exported model.
    :param new_ext: Extension to append to the model filename.
    :return: Path to the output file.
    """
    out_file = model_path.with_suffix(new_ext)
    if save_dir is not None:
        out_file = save_dir / out_file.name

    if out_file.exists():
        logging.info("Overwriting existing %s", out_file)
        if out_file.is_dir():
            shutil.rmtree(out_file)
        else:
            out_file.unlink()

    return out_file


def _prepare_model_for_onnx_export(
    model: keras.Model,
    plate_config: PlateOCRConfig,
    dynamic_batch: bool,
    input_dtype: str,
    data_format: TensorDataFormat,
):
    """
    Prepare a Keras model for ONNX export by adjusting input layout if needed.

    The model is only wrapped when 'channels_first' (NxCxHxW) format is requested, by inserting a
    Permute layer to convert NxCxHxW to NxHxWxC (the model's expected input).
    """
    if data_format == "channels_first":
        # NxCxHxW -> NxHxWxC
        inp_shape = (
            plate_config.num_channels,
            plate_config.img_height,
            plate_config.img_width,
        )
        x_in = keras.Input(shape=inp_shape, dtype=input_dtype, name="input_nchw")
        x_out = model(keras.layers.Permute((2, 3, 1))(x_in))
        export_model = keras.Model(x_in, x_out, name=f"{model.name}_nchw")
    else:
        # Default is channels last (NxHxWxC), keep the original graph
        inp_shape = (
            plate_config.img_height,
            plate_config.img_width,
            plate_config.num_channels,
        )
        export_model = model

    batch_dim = None if dynamic_batch else 1
    spec_shape = (batch_dim, *inp_shape)
    dummy_input = np.random.randint(0, 256, size=(1, *inp_shape)).astype(input_dtype)
    return export_model, spec_shape, dummy_input


@requires("onnx", "onnxruntime", "onnxslim")
def export_onnx(
    model: keras.Model,
    plate_config: PlateOCRConfig,
    out_file: pathlib.Path,
    simplify: bool,
    dynamic_batch: bool,
    skip_validation: bool = False,
    onnx_input_dtype: str = "uint8",
    onnx_data_format: TensorDataFormat = "channels_last",
) -> None:
    import onnxruntime as rt

    export_model, spec_shape, dummy_input = _prepare_model_for_onnx_export(
        model, plate_config, dynamic_batch, onnx_input_dtype, onnx_data_format
    )
    spec = [keras.InputSpec(name="input", shape=spec_shape, dtype=onnx_input_dtype)]

    with NamedTemporaryFile(suffix=".onnx") as tmp:
        export_model.export(tmp.name, format="onnx", verbose=False, input_signature=spec)

        if simplify:
            import onnx
            import onnxslim

            logging.info("Simplifying ONNX ...")
            model_simp = onnxslim.slim(onnx.load(tmp.name))
            onnx.save(model_simp, out_file)
        else:
            shutil.copy(tmp.name, out_file)

    # Load the newly converted ONNX model
    sess = rt.InferenceSession(out_file)
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    def _predict(x: np.ndarray):
        return sess.run(output_names, {input_name: x})[0]

    if skip_validation:
        logging.info("Skipping ONNX validation.")
    else:
        _validate_prediction(export_model, _predict, dummy_input, "ONNX")

    with log_time_taken("ONNX inference time"):
        _predict(dummy_input)

    logging.info("Saved ONNX model to %s", out_file)


@requires("tensorflow")
def export_tflite(
    model: keras.Model,
    plate_config: PlateOCRConfig,
    out_file: pathlib.Path,
    skip_validation: bool = False,
) -> None:
    import tensorflow as tf

    with TemporaryDirectory() as tmp_dir:
        model.export(tmp_dir, format="tf_saved_model")

        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_bytes = converter.convert()
        out_file.write_bytes(tflite_bytes)

    if skip_validation:
        logging.info("Skipping TFLite validation.")
        logging.info("Saved TFLite model to %s", out_file)
        return

    class _TFLiteRunner:
        def __init__(self, path):
            self.interp = tf.lite.Interpreter(str(path))
            self.interp.allocate_tensors()
            self.inp = self.interp.get_input_details()[0]["index"]
            self.out = self.interp.get_output_details()[0]["index"]

        def __call__(self, x: np.ndarray):
            self.interp.set_tensor(self.inp, x)
            self.interp.invoke()
            return self.interp.get_tensor(self.out)

    tfl_runner = _TFLiteRunner(out_file)
    _validate_prediction(
        model,
        tfl_runner,
        _dummy_input(
            1,
            plate_config.img_height,
            plate_config.img_width,
            plate_config.num_channels,
            np.float32,
        ),
        "TFLite",
        atol=5e-3,
        rtol=5e-3,
    )
    logging.info("Saved TFLite model to %s", out_file)


@requires("coremltools", "tensorflow")
def export_coreml(
    model: keras.Model,
    plate_config: PlateOCRConfig,
    out_file: pathlib.Path,
    skip_validation: bool = False,
) -> None:
    import coremltools as ct
    import tensorflow as tf

    with TemporaryDirectory() as tmp_dir:
        model.export(tmp_dir, format="tf_saved_model")
        loaded = tf.saved_model.load(tmp_dir)
        func = loaded.signatures["serving_default"]

        ct_inputs = [
            ct.TensorType(
                shape=(
                    1,
                    plate_config.img_height,
                    plate_config.img_width,
                    plate_config.num_channels,
                ),
                dtype=np.float32,
            )
        ]
        mlmodel = ct.convert(
            [func],
            source="tensorflow",
            convert_to="mlprogram",
            inputs=ct_inputs,
        )
        mlmodel.save(str(out_file))

    if skip_validation:
        logging.info("Skipping CoreML validation.")
        return

    mlmodel = ct.models.MLModel(str(out_file))

    spec = mlmodel.get_spec()
    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name

    def _predict(x: np.ndarray):
        return mlmodel.predict({input_name: x})[output_name]

    _validate_prediction(
        model,
        _predict,
        _dummy_input(
            1,
            plate_config.img_height,
            plate_config.img_width,
            plate_config.num_channels,
            np.float32,
        ),
        "CoreML",
    )
    logging.info("Saved CoreML model to %s", out_file)


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
    "-f",
    "--format",
    "export_format",
    type=click.Choice(["onnx", "tflite", "coreml"], case_sensitive=False),
    default="onnx",
    show_default=True,
    help="Target export format.",
)
@click.option(
    "--simplify/--no-simplify",
    default=True,
    show_default=True,
    help="Simplify ONNX model using onnxslim (only applies when format is ONNX).",
)
@click.option(
    "--plate-config-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Path to the model OCR config YAML.",
)
@click.option(
    "--save-dir",
    required=False,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    help="Directory to save the exported model. Defaults to model's directory.",
)
@click.option(
    "--dynamic-batch/--no-dynamic-batch",
    default=True,
    show_default=True,
    help="Enable dynamic batch size (only applies to ONNX format).",
)
@click.option(
    "--skip-validation/--no-skip-validation",
    default=False,
    show_default=True,
    help="Skip the post-export inference validation step.",
)
@click.option(
    "--onnx-input-dtype",
    type=click.Choice(["uint8", "float32"], case_sensitive=False),
    default="uint8",
    show_default=True,
    help="Data type of the ONNX model input.",
)
@click.option(
    "--onnx-data-format",
    type=click.Choice(["channels_last", "channels_first"], case_sensitive=False),
    default="channels_last",
    show_default=True,
    help=(
        "Data format of the input tensor. It can be either "
        "'channels_last' (NHWC) or 'channels_first' (NCHW)."
    ),
)
def export(  # noqa: PLR0913
    model_path: pathlib.Path,
    export_format: str,
    simplify: bool,
    plate_config_file: pathlib.Path,
    save_dir: pathlib.Path,
    dynamic_batch: bool,
    skip_validation: bool,
    onnx_input_dtype: str,
    onnx_data_format: TensorDataFormat,
) -> None:
    """
    Export Keras models to other formats.
    """

    plate_config = load_plate_config_from_yaml(plate_config_file)
    model = load_keras_model(model_path, plate_config)

    if export_format == "onnx":
        out_file = _make_output_path(model_path, save_dir, ".onnx")
        export_onnx(
            model=model,
            plate_config=plate_config,
            out_file=out_file,
            simplify=simplify,
            dynamic_batch=dynamic_batch,
            skip_validation=skip_validation,
            onnx_input_dtype=onnx_input_dtype,
            onnx_data_format=onnx_data_format,
        )
    elif export_format == "tflite":
        out_file = _make_output_path(model_path, save_dir, ".tflite")
        # TFLite doesn't seem to support dynamic batch size
        # See: https://ai.google.dev/edge/litert/inference#run-inference
        export_tflite(
            model=model,
            plate_config=plate_config,
            out_file=out_file,
        )
    elif export_format == "coreml":
        out_file = _make_output_path(model_path, save_dir, ".mlpackage")
        export_coreml(
            model=model,
            plate_config=plate_config,
            out_file=out_file,
            skip_validation=skip_validation,
        )


if __name__ == "__main__":
    export()
