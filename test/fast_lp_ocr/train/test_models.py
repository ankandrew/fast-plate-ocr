"""
Test OCR models module.
"""

import pytest
from keras import Input

from fast_plate_ocr.train.model import models


@pytest.mark.parametrize(
    "max_plates_slots, vocabulary_size, expected_hidden_units", [(7, 37, 7 * 37)]
)
def test_head(max_plates_slots: int, vocabulary_size: int, expected_hidden_units: int) -> None:
    x = Input((70, 140, 1))
    out_tensor = models.head(x, max_plates_slots, vocabulary_size)
    actual_hidden_units = out_tensor.shape[-1]
    assert actual_hidden_units == expected_hidden_units


@pytest.mark.parametrize(
    "max_plates_slots, vocabulary_size, expected_hidden_units", [(7, 37, 7 * 37)]
)
def test_head_no_fc(
    max_plates_slots: int, vocabulary_size: int, expected_hidden_units: int
) -> None:
    x = Input((70, 140, 1))
    out_tensor = models.head_no_fc(x, max_plates_slots, vocabulary_size)
    actual_hidden_units = out_tensor.shape[1] * out_tensor.shape[2]
    assert actual_hidden_units == expected_hidden_units
