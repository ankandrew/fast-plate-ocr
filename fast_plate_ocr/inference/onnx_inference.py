"""
ONNX inference module.
"""

import logging
from contextlib import nullcontext

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

import fast_plate_ocr.inference.config
from fast_plate_ocr.common.utils import log_time_taken
from fast_plate_ocr.inference import hub
from fast_plate_ocr.inference.process import postprocess_output, preprocess_image, read_plate_image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def _load_image_from_source(source: str | list[str] | npt.NDArray) -> npt.NDArray:
    """
    Loads an image from a given source.

    :param source: Path to the input image file or numpy array representing the image.
    :return: Numpy array representing the input image.
    """
    if isinstance(source, str):
        return read_plate_image(source)

    if isinstance(source, list) and isinstance(source[0], str):
        return np.array([read_plate_image(i) for i in source])

    if isinstance(source, np.ndarray):
        if source.ndim > 3:
            raise ValueError("Expected source to be of shape (H, W, 1) or (H, W) or (1, H, W, 1)")
        source = source.squeeze()
        return source

    raise ValueError("Unsupported input type. Only file path or numpy array is supported.")


class FastPlateOCR:
    """
    ONNX inference class for performing license plates OCR.
    """

    def __init__(self, ocr_model: str, use_gpu: bool = False, log_time: bool = False):
        """
        The current OCR models available are:

        - 'argentinian-plates-cnn-model': OCR for Argentinian license plates.

        :param ocr_model: Name of the OCR model to use.
        :param use_gpu: Flag indicating whether to use GPU backend.
        """
        self.logger = logging.getLogger(__name__)
        self.log_time = log_time

        if use_gpu:
            self.providers = ["CUDAExecutionProvider"]
            self.device = "GPU"
        else:
            self.providers = ["CPUExecutionProvider"]
            self.device = "CPU"

        model_path, config_path = hub.download_model(model_name=ocr_model)
        self.config = fast_plate_ocr.inference.config.load_config_from_yaml(config_path)
        self.model = ort.InferenceSession(model_path, providers=self.providers)
        self.logger.info("Using ONNX Runtime with %s device.", self.device)

    def run(
        self,
        source: str | list[str] | npt.NDArray,
        return_confidence: bool = False,
    ) -> tuple[list[str], npt.NDArray] | list[str]:
        """
        Runs inference on an image.

        :param source: Path to the input image file or numpy array representing the image.
        :param return_confidence: Whether to return confidence scores along with plate predictions.
        :return: Decoded license plate characters as a list.
        """
        x = _load_image_from_source(source)
        with log_time_taken("Pre-process") if self.log_time else nullcontext():
            x = preprocess_image(x, self.config["img_height"], self.config["img_width"])
        with log_time_taken("Model run") if self.log_time else nullcontext():
            y: list[npt.NDArray] = self.model.run(None, {"input": x})
        with log_time_taken("Post-process") if self.log_time else nullcontext():
            return postprocess_output(
                y[0],
                self.config["max_plate_slots"],
                self.config["alphabet"],
                return_confidence=return_confidence,
            )
