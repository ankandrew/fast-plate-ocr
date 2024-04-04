"""
Utilities function used for doing inference with the OCR models.
"""

import logging
import pathlib
import shutil
import urllib.request
from http import HTTPStatus

from tqdm.asyncio import tqdm

BASE_URL: str = "https://github.com/ankandrew/cnn-ocr-lp/releases/download"

AVAILABLE_ONNX_MODELS: dict[str, str] = {
    "argentinian-plates-cnn-model": f"{BASE_URL}/untagged-2d82907046a6a4abe03f/arg-cnn-ocr.onnx"
}
"""Available ONNX models for doing inference."""


def _download_with_progress(url: str, filename: pathlib.Path) -> None:
    """
    Download utility function with progress bar.

    :param url: URL of the model to download.
    :param filename: Where to save the OCR model.
    """
    with urllib.request.urlopen(url) as response, open(filename, "wb") as out_file:
        if response.getcode() != HTTPStatus.OK:
            raise ValueError(f"Failed to download file from {url}. Status code: {response.status}")

        file_size = int(response.headers.get("Content-Length", 0))
        desc = "(Unknown total file size)" if file_size == 0 else ""

        with tqdm.wrapattr(out_file, "write", total=file_size, desc=desc) as f_out:
            shutil.copyfileobj(response, f_out)


def download_model(
    model_name: str,
    save_directory: pathlib.Path,
    skip_if_exists: bool = True,
    create_dir: bool = True,
) -> pathlib.Path:
    """
    Download an OCR model to a given directory.

    :param model_name: Which model to download. Current available models are
     ('argentinian-plates-cnn-model', ).
    :param save_directory: Directory to save the OCR model. It should point to a folder.
    :param skip_if_exists: Don't download model if it already exists in `save_directory`.
    :param create_dir: Create directory recursively if parents don't exist.
    :return: Path to the downloaded model.
    """
    if model_name not in AVAILABLE_ONNX_MODELS:
        raise ValueError(f"Unknown model {model_name}. Use one of {AVAILABLE_ONNX_MODELS.keys()}")

    model_url = AVAILABLE_ONNX_MODELS[model_name]
    filename = save_directory / model_url.split("/")[-1]

    if skip_if_exists and filename.is_file():
        logging.info(
            "Skipping download of %s model, already exists at %s",
            model_name,
            save_directory,
        )
        return filename

    if save_directory.is_file():
        raise ValueError(f"Expected a directory, but got {save_directory}")

    save_directory.parent.mkdir(parents=create_dir, exist_ok=True)

    logging.info("Downloading model to %s", filename)
    _download_with_progress(url=model_url, filename=filename)
