"""
Script for exporting the trained Keras models to other formats.
"""

import logging
import pathlib
import platform
import shutil
from tempfile import TemporaryDirectory

import click
import keras
import numpy as np
from numpy.typing import DTypeLike

from fast_plate_ocr.cli.utils import requires
from fast_plate_ocr.core.types import TensorDataFormat
from fast_plate_ocr.core.utils import log_time_taken
from fast_plate_ocr.train.model.config import (
    PlateConfig,
    load_plate_config_from_yaml,
)
from fast_plate_ocr.train.utilities.utils import load_keras_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


# ruff: noqa: PLC0415
# pylint: disable=too-many-arguments,too-many-locals,import-outside-toplevel


def _dummy_input(b: int, h: int, w: int, n_c: int, dtype: DTypeLike = np.uint8) -> np.ndarray:
    """Random tensor in [0, 255] shaped (b, h, w, 1)."""
    return np.random.randint(0, 256, size=(b, h, w, n_c)).astype(dtype)


def _normalize_outputs(
    outputs,
    names: list[str],
):
    """Convert Keras/ONNX outputs into a name-mapped dictionary for comparison."""
    if isinstance(outputs, dict):
        return {name: outputs[name] for name in names}
    if isinstance(outputs, (list, tuple)):
        return dict(zip(names, outputs, strict=False))
    if len(names) != 1:
        raise ValueError("Expected multiple outputs but received a single tensor.")
    return {names[0]: outputs}


def _validate_prediction(
    keras_model: keras.Model,
    exported_predict,
    x: np.ndarray,
    target: str,
    output_names: list[str],
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """Compare Keras and exported backend on a single forward pass."""
    keras_out = _normalize_outputs(keras_model.predict(x, verbose=0), output_names)
    exported_out = exported_predict(x)
    for name in output_names:
        if not np.allclose(keras_out[name], exported_out[name], rtol=rtol, atol=atol):
            logging.warning("%s output '%s' deviates from Keras beyond tolerance.", target.upper(), name)
        else:
            logging.info("%s output '%s' matches Keras ✔", target.upper(), name)


def _get_output_names(model: keras.Model) -> list[str]:
    if isinstance(model.output, dict):
        return list(model.output.keys())
    if hasattr(model, "output_names") and model.output_names:
        return list(model.output_names)
    return [t.name.split(":")[0] for t in model.outputs]


def _make_output_path(model_path: pathlib.Path, save_dir: pathlib.Path | None, new_ext: str) -> pathlib.Path:
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
    plate_config: PlateConfig,
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
        inp_shape = (plate_config.num_channels, plate_config.img_height, plate_config.img_width)
        x_in = keras.Input(shape=inp_shape, dtype=input_dtype, name="input_nchw")
        x_out = model(keras.layers.Permute((2, 3, 1))(x_in))
        export_model = keras.Model(x_in, x_out, name=f"{model.name}_nchw")
    else:
        # Default is channels-last (NxHxWxC), keep the original graph
        inp_shape = (plate_config.img_height, plate_config.img_width, plate_config.num_channels)
        export_model = model

    batch_dim = None if dynamic_batch else 1
    spec_shape = (batch_dim, *inp_shape)
    dummy_input = np.random.randint(0, 256, size=(1, *inp_shape)).astype(input_dtype)
    return export_model, spec_shape, dummy_input


@requires("onnx", "onnxruntime", "onnxslim")
def export_onnx(
    model: keras.Model,
    plate_config: PlateConfig,
    out_file: pathlib.Path,
    simplify: bool,
    dynamic_batch: bool,
    skip_validation: bool = False,
    onnx_input_dtype: str = "uint8",
    onnx_data_format: TensorDataFormat = "channels_last",
    opset_version: int | None = None,
) -> None:
    import onnxruntime as rt

    export_model, spec_shape, dummy_input = _prepare_model_for_onnx_export(
        model,
        plate_config,
        dynamic_batch,
        onnx_input_dtype,
        onnx_data_format,
    )
    spec = [keras.InputSpec(name="input", shape=spec_shape, dtype=onnx_input_dtype)]

    with TemporaryDirectory() as tmp_dir:
        tmp_onnx = pathlib.Path(tmp_dir) / "model.onnx"
        export_model.export(
            str(tmp_onnx),
            format="onnx",
            verbose=False,
            input_signature=spec,
            opset_version=opset_version,
        )

        if simplify:
            import onnx
            import onnxslim

            logging.info("Simplifying ONNX ...")
            model_simp = onnxslim.slim(onnx.load(tmp_onnx))
            onnx.save(model_simp, out_file)
        else:
            shutil.copy(tmp_onnx, out_file)

    # Load the newly converted ONNX model
    sess = rt.InferenceSession(out_file)
    input_name = sess.get_inputs()[0].name
    onnx_output_names = [o.name for o in sess.get_outputs()]

    if not isinstance(export_model.output, dict):
        raise ValueError("Exporter expects dict outputs (e.g., {'plate': ..., 'region': ...}). ")

    keras_keys = list(export_model.output.keys())

    if len(keras_keys) != len(onnx_output_names):
        raise ValueError(
            f"Mismatch: Keras dict outputs ({len(keras_keys)}) vs ONNX outputs ({len(onnx_output_names)})."
        )

    def _predict(x: np.ndarray):
        values = sess.run(onnx_output_names, {input_name: x})
        return dict(zip(keras_keys, values, strict=False))

    if skip_validation:
        logging.info("Skipping ONNX validation.")
    else:
        _validate_prediction(
            export_model,
            _predict,
            dummy_input,
            "ONNX",
            output_names=keras_keys,
        )

    with log_time_taken("ONNX inference time"):
        _predict(dummy_input)

    logging.info("Saved ONNX model to %s", out_file)


@requires("tensorflow")
def export_tflite(
    model: keras.Model, plate_config: PlateConfig, out_file: pathlib.Path, skip_validation: bool = False
) -> None:
    import tensorflow as tf

    with TemporaryDirectory() as tmp_dir:
        model.export(tmp_dir, format="tf_saved_model")

        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        out_file.write_bytes(converter.convert())

    if skip_validation:
        logging.info("Skipping TFLite validation.")
        logging.info("Saved TFLite model to %s", out_file)
        return

    from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter

    output_names = _get_output_names(model)

    class _TFLiteRunner:
        def __init__(self, path: pathlib.Path, out_names: list[str]):
            self.interp = LiteRTInterpreter(model_path=str(path))
            self.interp.allocate_tensors()
            sigs = self.interp.get_signature_list()
            self.sig = next(iter(sigs))
            self.runner = self.interp.get_signature_runner(self.sig)

            sig_meta = sigs[self.sig]
            self.inp_name = next(iter(sig_meta["inputs"]))
            self.out_names = out_names

        def __call__(self, x: np.ndarray):
            y = self.runner(**{self.inp_name: x})
            return {k: y[k] for k in self.out_names}

    tfl_runner = _TFLiteRunner(out_file, output_names)

    _validate_prediction(
        model,
        tfl_runner,
        _dummy_input(1, plate_config.img_height, plate_config.img_width, plate_config.num_channels, np.float32),
        "TFLite",
        output_names=output_names,
        atol=5e-3,
        rtol=5e-3,
    )

    logging.info("Saved TFLite model to %s", out_file)


@requires("coremltools", "tensorflow")
def export_coreml(
    model: keras.Model,
    plate_config: PlateConfig,
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
        mlmodel = ct.convert([func], source="tensorflow", convert_to="mlprogram", inputs=ct_inputs)
        mlmodel.save(str(out_file))

    if skip_validation:
        logging.info("Skipping CoreML validation.")
        logging.info("Saved CoreML model to %s", out_file)
        return

    if platform.system() != "Darwin":
        logging.info("Skipping CoreML validation outside macOS.")
        logging.info("Saved CoreML model to %s", out_file)
        return

    mlmodel = ct.models.MLModel(str(out_file))

    # Keras keys (dict outputs): ["plate"] or ["plate", "region"]
    output_names = _get_output_names(model)

    spec = mlmodel.get_spec()
    input_name = spec.description.input[0].name

    # CoreML output keys in a stable, declared order
    coreml_keys = [o.name for o in spec.description.output]

    if len(coreml_keys) != len(output_names):
        raise ValueError(f"CoreML outputs {coreml_keys} != expected {output_names}")

    coreml_to_keras = dict(zip(coreml_keys, output_names, strict=False))

    def _predict(x: np.ndarray):
        y = mlmodel.predict({input_name: x})
        return {coreml_to_keras[k]: y[k] for k in coreml_keys}

    dummy = _dummy_input(1, plate_config.img_height, plate_config.img_width, plate_config.num_channels, np.float32)

    _validate_prediction(
        model,
        _predict,
        dummy,
        "CoreML",
        output_names=output_names,
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
    "--onnx-opset-version",
    required=False,
    default=None,
    type=int,
    help="ONNX opset version to use (only applies when format is ONNX).",
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
    help="Data format of the input tensor. It can be either 'channels_last' (NHWC) or 'channels_first' (NCHW).",
)
def export(  # noqa: PLR0913
    model_path: pathlib.Path,
    export_format: str,
    simplify: bool,
    onnx_opset_version: int | None,
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
            opset_version=onnx_opset_version,
        )
    elif export_format == "tflite":
        out_file = _make_output_path(model_path, save_dir, ".tflite")
        # TODO: From Keras 3.13.0 we can use TFLite exporter directly in model.export(...)
        export_tflite(
            model=model,
            plate_config=plate_config,
            out_file=out_file,
            skip_validation=skip_validation,
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
