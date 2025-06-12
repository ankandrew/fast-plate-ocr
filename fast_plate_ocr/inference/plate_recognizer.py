"""
ONNX inference module.
"""

import logging
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

from fast_plate_ocr.core.process import (
    postprocess_output,
    preprocess_image,
    read_and_resize_plate_image,
    resize_image,
)
from fast_plate_ocr.core.types import BatchArray, BatchOrImgLike, ImgLike, PathLike
from fast_plate_ocr.core.utils import measure_time
from fast_plate_ocr.inference import hub
from fast_plate_ocr.inference.config import PlateOCRConfig
from fast_plate_ocr.inference.hub import OcrModel


def _frame_from(item: ImgLike, cfg: PlateOCRConfig) -> BatchArray:
    """
    Converts a single image-like input into a normalized (H, W, C) NumPy array ready for model
    inference. It handles both file paths and in-memory images. If input is a file path, the image
    is read and resized using the configuration provided. If it's a NumPy array, it is validated and
    resized accordingly.
    """
    # If it's a path, read and resize
    if isinstance(item, (str | pathlib.PurePath)):
        return read_and_resize_plate_image(
            item,
            img_height=cfg.img_height,
            img_width=cfg.img_width,
            image_color_mode=cfg.image_color_mode,
            keep_aspect_ratio=cfg.keep_aspect_ratio,
            interpolation_method=cfg.interpolation,
            padding_color=cfg.padding_color,
        )

    # Otherwise it must be a numpy array
    if not isinstance(item, np.ndarray):
        raise TypeError(f"Unsupported element type: {type(item)}")

    # If it has (N, H, W, C) shape we assume it's ready for inference
    if item.ndim == 4:
        return item

    # If it's a single frame resize accordingly
    return resize_image(
        item,
        cfg.img_height,
        cfg.img_width,
        image_color_mode=cfg.image_color_mode,
        keep_aspect_ratio=cfg.keep_aspect_ratio,
        interpolation_method=cfg.interpolation,
        padding_color=cfg.padding_color,
    )


def _load_image_from_source(source: BatchOrImgLike, cfg: PlateOCRConfig) -> BatchArray:
    """
    Converts an image input or batch of inputs into a 4-D NumPy array (N, H, W, C).

    This utility supports a wide range of input formats, including single images or batches, file
    paths or NumPy arrays. It ensures the result is always a model-ready batch.

    Supported input formats:
    - Single path (`str` or `PathLike`) -> image is read and resized
    - List or tuple of paths -> each image is read and resized
    - Single 2D or 3D NumPy array -> resized and wrapped in a batch
    - List or tuple of NumPy arrays -> each image is resized and batched
    - Single 4D NumPy array with shape (N, H, W, C) -> returned as is

    Args:
        source: A single image or batch of images in path or NumPy array format.
        cfg: The configuration object that defines image preprocessing parameters.

    Returns:
        A 4D NumPy array of shape (N, H, W, C), dtype uint8, ready for model inference.
    """
    if isinstance(source, np.ndarray) and source.ndim == 4:
        return source

    items: Sequence[ImgLike] = (
        source
        if isinstance(source, Sequence)
        and not isinstance(source, (str | pathlib.PurePath | np.ndarray))
        else [source]
    )

    frames: list[BatchArray] = [
        frame
        for item in items
        for frame in (
            _frame_from(item, cfg)  # type: ignore[attr-defined]
            if isinstance(item, np.ndarray) and item.ndim == 4
            else [_frame_from(item, cfg)]
        )
    ]

    return np.stack(frames, axis=0, dtype=np.uint8)


class LicensePlateRecognizer:
    """
    ONNX inference class for performing license plates OCR.
    """

    def __init__(
        self,
        hub_ocr_model: OcrModel | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions | None = None,
        model_path: PathLike | None = None,
        config_path: PathLike | None = None,
        force_download: bool = False,
    ) -> None:
        """
        Initializes the `LicensePlateRecognizer` with the specified OCR model and inference device.

        The current OCR models available from the HUB are:

        - `argentinian-plates-cnn-model`: OCR for Argentinian license plates. Uses fully conv
            architecture.
        - `argentinian-plates-cnn-synth-model`: OCR for Argentinian license plates trained with
            synthetic and real data. Uses fully conv architecture.
        - `european-plates-mobile-vit-v2-model`: OCR for European license plates. Uses MobileVIT-2
            for the backbone.
        - `global-plates-mobile-vit-v2-model`: OCR for Global license plates (+65 countries).
            Uses MobileVIT-2 for the backbone.

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

        self.config = PlateOCRConfig.from_yaml(config_path)
        self.model = ort.InferenceSession(
            model_path, providers=self.providers, sess_options=sess_options
        )
        self.logger.info("Using ONNX Runtime with %s.", self.providers)

    def benchmark(
        self,
        n_iter: int = 10_000,
        batch_size: int = 1,
        include_processing: bool = False,
        warmup: int = 50,
    ) -> None:
        """
        Run an inference benchmark and pretty print the results.

        It reports the following metrics:

        * **Average latency per batch** (milliseconds)
        * **Throughput** in *plates / second* (PPS), i.e., how many plates the pipeline can process
          per second at the chosen ``batch_size``.

        Args:
            n_iter: The number of iterations to run the benchmark. This determines how many times
                the inference will be executed to compute the average performance metrics.
            batch_size : Batch size to use for the benchmark.
            include_processing: Indicates whether the benchmark should include preprocessing and
                postprocessing times in the measurement.
            warmup: Number of warmup iterations to run before the benchmark.
        """
        x = np.random.randint(
            0,
            256,
            size=(
                batch_size,
                self.config.img_height,
                self.config.img_width,
                self.config.num_channels,
            ),
            dtype=np.uint8,
        )

        # Warm-up
        for _ in range(warmup):
            if include_processing:
                self.run(x)
            else:
                self.model.run(None, {"input": x})

        # Timed loop
        cum_time = 0.0
        for _ in range(n_iter):
            with measure_time() as time_taken:
                if include_processing:
                    self.run(x)
                else:
                    self.model.run(None, {"input": x})
            cum_time += time_taken()

        avg_time_ms = cum_time / n_iter if n_iter else 0.0
        pps = (1_000 / avg_time_ms) * batch_size if n_iter else 0.0

        console = Console()
        model_info = Panel(
            Text(f"Model: {self.model_name}\nProviders: {self.providers}", style="bold green"),
            title="Model Information",
            border_style="bright_blue",
            expand=False,
        )
        console.print(model_info)
        table = Table(title=f"Benchmark for '{self.model_name}'", border_style="bright_blue")
        table.add_column("Metric", justify="center", style="cyan", no_wrap=True)
        table.add_column("Value", justify="center", style="magenta")

        table.add_row("Batch size", str(batch_size))
        table.add_row("Warm-up iters", str(warmup))
        table.add_row("Timed iterations", str(n_iter))
        table.add_row("Average Time / batch (ms)", f"{avg_time_ms:.4f}")
        table.add_row("Plates per Second (PPS)", f"{pps:.4f}")
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
        x = _load_image_from_source(source, self.config)
        # Preprocess
        x = preprocess_image(x)
        # Run model
        y: list[npt.NDArray] = self.model.run(None, {"input": x})
        # Postprocess model output
        return postprocess_output(
            y[0],
            self.config.max_plate_slots,
            self.config.alphabet,
            return_confidence=return_confidence,
        )
