"""
Utilities function used for doing inference with the OCR models.
"""

import logging
import pathlib
import shutil
import urllib.request
from http import HTTPStatus
from typing import Literal

from tqdm.asyncio import tqdm

from fast_plate_ocr.core.utils import safe_write

BASE_URL: str = "https://github.com/ankandrew/cnn-ocr-lp/releases/download"
OcrModel = Literal[
    "cct-s-v1-global-model",
    "cct-xs-v1-global-model",
    "argentinian-plates-cnn-model",
    "argentinian-plates-cnn-synth-model",
    "european-plates-mobile-vit-v2-model",
    "global-plates-mobile-vit-v2-model",
]
"""Available OCR models for doing inference."""


AVAILABLE_ONNX_MODELS: dict[OcrModel, tuple[str, str]] = {
    "cct-s-v1-global-model": (
        f"{BASE_URL}/arg-plates/cct_s_v1_global.onnx",
        f"{BASE_URL}/arg-plates/cct_s_v1_global_plate_config.yaml",
    ),
    "cct-xs-v1-global-model": (
        f"{BASE_URL}/arg-plates/cct_xs_v1_global.onnx",
        f"{BASE_URL}/arg-plates/cct_xs_v1_global_plate_config.yaml",
    ),
    "argentinian-plates-cnn-model": (
        f"{BASE_URL}/arg-plates/arg_cnn_ocr.onnx",
        f"{BASE_URL}/arg-plates/arg_cnn_ocr_config.yaml",
    ),
    "argentinian-plates-cnn-synth-model": (
        f"{BASE_URL}/arg-plates/arg_cnn_ocr_synth.onnx",
        f"{BASE_URL}/arg-plates/arg_cnn_ocr_config.yaml",
    ),
    "european-plates-mobile-vit-v2-model": (
        f"{BASE_URL}/arg-plates/european_mobile_vit_v2_ocr.onnx",
        f"{BASE_URL}/arg-plates/european_mobile_vit_v2_ocr_config.yaml",
    ),
    "global-plates-mobile-vit-v2-model": (
        f"{BASE_URL}/arg-plates/global_mobile_vit_v2_ocr.onnx",
        f"{BASE_URL}/arg-plates/global_mobile_vit_v2_ocr_config.yaml",
    ),
}
"""Dictionary of available OCR models and their URLs."""

MODEL_CACHE_DIR: pathlib.Path = pathlib.Path.home() / ".cache" / "fast-plate-ocr"
"""Default location where models will be stored."""


def _download_with_progress(url: str, filename: pathlib.Path) -> None:
    """
    Download utility function with progress bar.

    :param url: URL of the model to download.
    :param filename: Where to save the OCR model.
    """
    with urllib.request.urlopen(url) as response, safe_write(filename, mode="wb") as out_file:
        if response.getcode() != HTTPStatus.OK:
            raise ValueError(f"Failed to download file from {url}. Status code: {response.status}")

        file_size = int(response.headers.get("Content-Length", 0))
        desc = f"Downloading {filename.name}"

        with tqdm.wrapattr(out_file, "write", total=file_size, desc=desc) as f_out:
            shutil.copyfileobj(response, f_out)


def download_model(
    model_name: OcrModel,
    save_directory: pathlib.Path | None = None,
    force_download: bool = False,
) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Download an OCR model and the config to a given directory.

    Args:
        model_name: Which model to download.
        save_directory: Directory to save the OCR model. It should point to a folder.
            If not supplied, this will point to '~/.cache/<model_name>'.
        force_download: Force and download the model if it already exists in
            `save_directory`.

    Returns:
        A tuple consisting of (model_downloaded_path, config_downloaded_path).
    """
    if model_name not in AVAILABLE_ONNX_MODELS:
        available_models = ", ".join(AVAILABLE_ONNX_MODELS.keys())
        raise ValueError(f"Unknown model {model_name}. Use one of [{available_models}]")

    if save_directory is None:
        save_directory = MODEL_CACHE_DIR / model_name
    elif save_directory.is_file():
        raise ValueError(f"Expected a directory, but got {save_directory}")

    save_directory.mkdir(parents=True, exist_ok=True)

    model_url, plate_config_url = AVAILABLE_ONNX_MODELS[model_name]
    model_filename = save_directory / model_url.split("/")[-1]
    plate_config_filename = save_directory / plate_config_url.split("/")[-1]

    if not force_download and model_filename.is_file() and plate_config_filename.is_file():
        logging.info(
            "Skipping download of '%s' model, already exists at %s",
            model_name,
            save_directory,
        )
        return model_filename, plate_config_filename

    # Download the model if not present or if we want to force the download
    if force_download or not model_filename.is_file():
        logging.info("Downloading model to %s", model_filename)
        _download_with_progress(url=model_url, filename=model_filename)

    # Same for the config
    if force_download or not plate_config_filename.is_file():
        logging.info("Downloading config to %s", plate_config_filename)
        _download_with_progress(url=plate_config_url, filename=plate_config_filename)

    return model_filename, plate_config_filename
