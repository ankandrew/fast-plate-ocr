"""
ONNX inference module.
"""

import logging
import os
import pathlib
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fast_plate_ocr.common.utils import measure_time
from fast_plate_ocr.inference import hub
from fast_plate_ocr.inference.config import load_config_from_yaml
from fast_plate_ocr.inference.hub import OcrModel
from fast_plate_ocr.inference.process import postprocess_output, preprocess_image, read_plate_image


def _load_image_from_source(
    source: str | list[str] | npt.NDArray | list[npt.NDArray],
) -> npt.NDArray | list[npt.NDArray]:
    """
    Loads an image from a given source.

    :param source: Path to the input image file, list of paths, or numpy array representing one or
     multiple images.
    :return: Numpy array representing the input image(s) or a list of numpy arrays.
    """
    if isinstance(source, str):
        # Shape returned (H, W)
        return read_plate_image(source)

    if isinstance(source, list):
        # Are image paths
        if all(isinstance(s, str) for s in source):
            # List returned with array item of shape (H, W)
            return [read_plate_image(i) for i in source]  # type: ignore[arg-type]
        # Are list of numpy arrays
        if all(isinstance(a, np.ndarray) for a in source):
            # List returned with array item of shape (H, W)
            return source  # type: ignore[return-value]
        raise ValueError("Expected source to be a list of `str` or `np.ndarray`!")

    if isinstance(source, np.ndarray):
        # Squeeze grayscale channel dimension if supplied
        source = source.squeeze()
        if source.ndim != 2:
            raise ValueError("Expected source array to be of shape (H, W) or (H, W, 1).")
        # Shape returned (H, W)
        return source

    raise ValueError("Unsupported input type. Only file path or numpy array is supported.")


class ONNXPlateRecognizer:
    """
    ONNX inference class for performing license plates OCR.
    """

    def __init__(
        self,
        hub_ocr_model: OcrModel | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions | None = None,
        model_path: str | os.PathLike[str] | None = None,
        config_path: str | os.PathLike[str] | None = None,
        force_download: bool = False,
    ) -> None:
        """
        Initializes the ONNXPlateRecognizer with the specified OCR model and inference device.

        The current OCR models available from the HUB are:

        - `argentinian-plates-cnn-model`: OCR for Argentinian license plates. Uses fully conv
            architecture.
        - `argentinian-plates-cnn-synth-model`: OCR for Argentinian license plates trained with
            synthetic and real data. Uses fully conv architecture.
        - `european-plates-mobile-vit-v2-model`: OCR for European license plates. Uses MobileVIT-2
            for the backbone.

        Args:
            hub_ocr_model: Name of the OCR model to use from the HUB.
            device: Device type for inference. Should be one of ('cpu', 'cuda', 'auto'). If
                'auto' mode, the device will be deduced from
                `onnxruntime.get_available_providers()`.
            providers: Optional sequence of providers in order of decreasing precedence. If not
                specified, all available providers are used based on the device argument.
            sess_options: Advanced session options for ONNX Runtime.
            model_path: Path to ONNX model file to use (In case you want to use a custom one).
            config_path: Path to config file to use (In case you want to use a custom one).
            force_download: Force and download the model, even if it already exists.
        Returns:
            None.
        """
        self.logger = logging.getLogger(__name__)

        if providers is not None:
            self.providers = providers
            self.logger.info("Using custom providers: %s", providers)
        else:
            if device == "cuda":
                self.providers = ["CUDAExecutionProvider"]
            elif device == "cpu":
                self.providers = ["CPUExecutionProvider"]
            elif device == "auto":
                self.providers = ort.get_available_providers()
            else:
                raise ValueError(
                    f"Device should be one of ('cpu', 'cuda', 'auto'). Got '{device}'."
                )

            self.logger.info("Using device '%s' with providers: %s", device, self.providers)

        if model_path and config_path:
            model_path = pathlib.Path(model_path)
            config_path = pathlib.Path(config_path)
            if not model_path.exists() or not config_path.exists():
                raise FileNotFoundError("Missing model/config file!")
            self.model_name = model_path.stem
        elif hub_ocr_model:
            self.model_name = hub_ocr_model
            model_path, config_path = hub.download_model(
                model_name=hub_ocr_model, force_download=force_download
            )
        else:
            raise ValueError(
                "Either provide a model from the HUB or a custom model_path and config_path"
            )

        self.config = load_config_from_yaml(config_path)
        self.model = ort.InferenceSession(
            model_path, providers=self.providers, sess_options=sess_options
        )
        self.logger.info("Using ONNX Runtime with %s.", self.providers)

    def benchmark(self, n_iter: int = 10_000, include_processing: bool = False) -> None:
        """
        Benchmark time taken to run the OCR model. This reports the average inference time and the
        throughput in plates per second.

        Args:
            n_iter: The number of iterations to run the benchmark. This determines how many times
                the inference will be executed to compute the average performance metrics.
            include_processing: Indicates whether the benchmark should include preprocessing and
                postprocessing times in the measurement.
        """
        cum_time = 0.0
        x = np.random.randint(
            0, 256, size=(1, self.config["img_height"], self.config["img_width"], 1), dtype=np.uint8
        )
        for _ in range(n_iter):
            with measure_time() as time_taken:
                if include_processing:
                    self.run(x)
                else:
                    self.model.run(None, {"input": x})
            cum_time += time_taken()

        avg_time = (cum_time / n_iter) if n_iter > 0 else 0.0
        avg_pps = (1_000 / avg_time) if n_iter > 0 else 0.0

        console = Console()
        model_info = Panel(
            Text(f"Model: {self.model_name}\nProviders: {self.providers}", style="bold green"),
            title="Model Information",
            border_style="bright_blue",
            expand=False,
        )
        console.print(model_info)
        table = Table(title=f"Benchmark for '{self.model_name}' Model", border_style="bright_blue")
        table.add_column("Metric", justify="center", style="cyan", no_wrap=True)
        table.add_column("Value", justify="center", style="magenta")
        table.add_row("Number of Iterations", str(n_iter))
        table.add_row("Average Time (ms)", f"{avg_time:.4f}")
        table.add_row("Plates Per Second (PPS)", f"{avg_pps:.4f}")
        console.print(table)

    def run(
        self,
        source: str | list[str] | npt.NDArray | list[npt.NDArray],
        return_confidence: bool = False,
    ) -> tuple[list[str], npt.NDArray] | list[str]:
        """
        Performs OCR to recognize license plate characters from an image or a list of images.

        Args:
            source: The path(s) to the image(s), a numpy array representing an image or a list
                of NumPy arrays. If a numpy array is provided, it is expected to already be in
                grayscale format, with shape `(H, W) `or `(H, W, 1)`. A list of numpy arrays with
                different image sizes may also be provided.
            return_confidence: Whether to return confidence scores along with plate predictions.

        Returns:
            A list of plates for each input image. If `return_confidence` is True, a numpy
                array is returned with the shape `(N, plate_slots)`, where N is the batch size and
                each plate slot is the confidence for the recognized license plate character.
        """
        x = _load_image_from_source(source)
        # Preprocess
        x = preprocess_image(x, self.config["img_height"], self.config["img_width"])
        # Run model
        y: list[npt.NDArray] = self.model.run(None, {"input": x})
        # Postprocess model output
        return postprocess_output(
            y[0],
            self.config["max_plate_slots"],
            self.config["alphabet"],
            return_confidence=return_confidence,
        )
