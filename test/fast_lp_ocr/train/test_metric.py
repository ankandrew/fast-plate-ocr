"""
Test the custom metric/losses module.
"""

import numpy as np
import pytest

from fast_plate_ocr.train.model.metric import (
    cat_acc_metric,
    plate_acc_metric,
    plate_len_acc_metric,
    top_3_k_metric,
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
    "y_true, y_pred, vocab_size, expected_acc",
    [
        # Both true labels are within the top-3 predictions, so accuracy should be 1.0
        (
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                ],
            ),
            np.array(
                [
                    [0.30, 0.20, 0.10, 0.35, 0.05],
                    [0.10, 0.05, 0.15, 0.20, 0.50],
                ],
            ),
            5,
            1.0,
        ),
        # One true label is in the top-3 predictions, the other one not, so accuracy should be 0.5
        (
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                ],
            ),
            np.array(
                [
                    [0.40, 0.30, 0.20, 0.10, 0.00],
                    [0.05, 0.15, 0.10, 0.50, 0.20],
                ],
            ),
            5,
            0.5,
        ),
        # Neither true label is in the top-3 predictions, so accuracy should be 0.0
        (
            np.array(
                [
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                ],
            ),
            np.array(
                [
                    [0.50, 0.05, 0.02, 0.25, 0.18],
                    [0.60, 0.20, 0.05, 0.10, 0.15],
                ],
            ),
            5,
            0.0,
        ),
    ],
)
def test_top_3_k(
    y_true: np.ndarray, y_pred: np.ndarray, vocab_size: int, expected_acc: float
) -> None:
    metric_fn = top_3_k_metric(vocab_size)
    actual = metric_fn(y_true, y_pred)
    assert actual == expected_acc


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
        # One match, one mismatch, accuracy should be 0.5
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
