"""
Test the custom metric/losses module.
"""

import pytest
import tensorflow as tf

from fast_lp_ocr.custom import cat_acc


@pytest.mark.parametrize(
    "y_true, y_pred, expected_accuracy",
    [
        (tf.constant([[[1, 0]] * 6]), tf.constant([[[0.9, 0.1]] * 6]), 1.0),
    ],
)
def test_cat_acc(y_true: tf.Tensor, y_pred: tf.Tensor, expected_accuracy: float):
    actual_accuracy = cat_acc(y_true, y_pred, -1, 1)
    assert actual_accuracy == expected_accuracy
