"""
Test the custom metric/losses module.
"""

# ruff: noqa: E402
# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports
# fmt: off
from fast_plate_ocr.utils import set_pytorch_backend

set_pytorch_backend()
# fmt: on

import pytest
import torch

from fast_plate_ocr.custom import cat_acc_metric


@pytest.mark.parametrize(
    "y_true, y_pred, expected_accuracy",
    [
        (torch.tensor([[[1, 0]] * 6]), torch.tensor([[[0.9, 0.1]] * 6]), 1.0),
    ],
)
def test_cat_acc(y_true: torch.Tensor, y_pred: torch.Tensor, expected_accuracy: float):
    actual_accuracy = cat_acc_metric(2, 1)(y_true, y_pred)
    assert actual_accuracy == expected_accuracy
