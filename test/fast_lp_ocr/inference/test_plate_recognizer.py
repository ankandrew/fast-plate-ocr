"""
Tests for ONNX inference module.
"""

from collections.abc import Iterator

import cv2
import numpy.typing as npt
import pytest

from fast_plate_ocr import LicensePlateRecognizer
from test.assets import ASSETS_DIR


@pytest.fixture(scope="module", name="onnx_model")
def onnx_model_fixture() -> Iterator[LicensePlateRecognizer]:
    yield LicensePlateRecognizer("argentinian-plates-cnn-model", device="cpu")


@pytest.mark.parametrize(
    "input_image, expected_plate_count",
    [
        # Single image path (str)
        (str(ASSETS_DIR / "test_plate_1.png"), 1),
        # Single image path (Path)
        (ASSETS_DIR / "test_plate_1.png", 1),
        # Multiple Image paths
        (
            [str(ASSETS_DIR / "test_plate_1.png"), str(ASSETS_DIR / "test_plate_2.png")],
            2,
        ),
        # Multiple Image paths (Path)
        ([ASSETS_DIR / "test_plate_1.png", ASSETS_DIR / "test_plate_2.png"], 2),
        # NumPy array with single image
        (cv2.imread(str(ASSETS_DIR / "test_plate_1.png"), cv2.IMREAD_GRAYSCALE), 1),
        # NumPy array with batch images
        (
            [
                cv2.imread(str(ASSETS_DIR / "test_plate_1.png"), cv2.IMREAD_GRAYSCALE),
                cv2.imread(str(ASSETS_DIR / "test_plate_2.png"), cv2.IMREAD_GRAYSCALE),
            ],
            2,
        ),
    ],
)
def test_result_from_different_image_sources(
    input_image: str | list[str] | npt.NDArray,
    expected_plate_count: int,
    onnx_model: LicensePlateRecognizer,
) -> None:
    actual_plate_count = len(onnx_model.run(input_image))
    assert actual_plate_count == expected_plate_count
