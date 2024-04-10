"""
Tests for inference process module.
"""

import numpy as np
import numpy.typing as npt
import pytest

from fast_plate_ocr.inference.process import postprocess_output


@pytest.mark.parametrize(
    "model_output, max_plate_slots, model_alphabet, expected_plates",
    [
        (
            np.array(
                [
                    [[0.5, 0.4, 0.1], [0.2, 0.6, 0.2], [0.1, 0.4, 0.5]],
                    [[0.1, 0.1, 0.8], [0.2, 0.2, 0.6], [0.1, 0.4, 0.5]],
                ],
                dtype=np.float32,
            ),
            3,
            "ABC",
            ["ABC", "CCC"],
        ),
        (
            np.array(
                [[[0.1, 0.4, 0.5], [0.6, 0.2, 0.2], [0.1, 0.5, 0.4]]],
                dtype=np.float32,
            ),
            3,
            "ABC",
            ["CAB"],
        ),
    ],
)
def test_postprocess_output(
    model_output: npt.NDArray,
    max_plate_slots: int,
    model_alphabet: str,
    expected_plates: list[str],
) -> None:
    actual_plate = postprocess_output(model_output, max_plate_slots, model_alphabet)
    assert actual_plate == expected_plates
