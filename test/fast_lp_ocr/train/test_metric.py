"""
Test the custom metric/losses module.
"""

import numpy as np
import pytest

from fast_plate_ocr.train.model.metric import (
    cat_acc_metric,
    plate_acc_metric,
    plate_len_acc_metric,
)


@pytest.mark.parametrize(
    "y_true, y_pred, expected_accuracy",
    [
        (np.array([[[1, 0]] * 6]), np.array([[[0.9, 0.1]] * 6]), 1.0),
    ],
)
def test_cat_acc(y_true: np.ndarray, y_pred: np.ndarray, expected_accuracy: float) -> None:
    actual_accuracy = cat_acc_metric(2, 1)(y_true, y_pred)
    assert actual_accuracy == expected_accuracy


@pytest.mark.parametrize(
    "y_true, y_pred, expected_accuracy",
    [
        (
            np.array(
                [
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    [
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                ]
            ),
            # First batch slice plate was recognized completely correct but second one wasn't
            # So 50% of plates were recognized correctly
            0.5,
        ),
    ],
)
def test_plate_accuracy(y_true: np.ndarray, y_pred: np.ndarray, expected_accuracy: float) -> None:
    actual_accuracy = plate_acc_metric(y_true.shape[1], y_true.shape[2])(y_true, y_pred)
    assert actual_accuracy == expected_accuracy


@pytest.mark.parametrize(
    "y_true, y_pred, max_slots, vocab_size, pad_idx, expected_acc",
    [
        # All lengths match, accuracy should be 1.0
        (
            np.array(
                [
                    [
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                    ],
                    [
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                    ],
                ],
            ),
            np.array(
                [
                    [
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                    ],
                    [
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                    ],
                ],
            ),
            4,
            3,
            0,
            1.0,
        ),
        # 2) One match, one mismatch, accuracy should be 0.5
        (
            np.array(
                [
                    [
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                    ],
                    [
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                    ],
                ],
            ),
            np.array(
                [
                    [
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                    ],
                    [
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                    ],
                ],
            ),
            4,
            3,
            0,
            0.5,
        ),
    ],
)
def test_plate_len_acc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_slots: int,
    vocab_size: int,
    pad_idx: int,
    expected_acc: float,
) -> None:
    metric_fn = plate_len_acc_metric(max_slots, vocab_size, pad_idx)
    actual = metric_fn(y_true, y_pred)
    assert actual == expected_acc
