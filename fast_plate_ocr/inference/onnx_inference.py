"""
ONNX inference module.
"""

import logging
from contextlib import nullcontext

import numpy.typing as npt
import onnxruntime as ort

import fast_plate_ocr.inference.config
from fast_plate_ocr.common.utils import log_time_taken
from fast_plate_ocr.inference import hub
from fast_plate_ocr.inference.process import postprocess_output, preprocess_image, read_plate_image

logging.basicConfig(level=logging.INFO)


class ONNXInference:
    """
    ONNX inference class for performing license plates OCR.
    """

    def __init__(self, ocr_model: str, use_gpu: bool = False, log_time: bool = False):
        """
        Initialize ONNXInference.

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
        image_path: str,
        return_confidence: bool = False,
    ) -> tuple[list[str], npt.NDArray] | list[str]:
        """
        Runs inference on an image.

        :param image_path: Path to the input image file.
        :param return_confidence: Whether to return confidence scores along with plate predictions.
        :return: Decoded license plate characters as a list.
        """
        x = read_plate_image(image_path)
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
