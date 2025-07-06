"""
Evaluation metrics for license plate recognition models.
"""

from keras import metrics, ops


def cat_acc_metric(max_plate_slots: int, vocabulary_size: int):
    """
    Categorical accuracy metric.
    """

    def cat_acc(y_true, y_pred):
        """
        This is simply the CategoricalAccuracy for multi-class label problems. Example if the
        correct label is ABC123 and ABC133 is predicted, it will not give a precision of 0% like
        plate_acc (not completely classified correctly), but 83.3% (5/6).
        """
        y_true = ops.reshape(y_true, newshape=(-1, max_plate_slots, vocabulary_size))
        y_pred = ops.reshape(y_pred, newshape=(-1, max_plate_slots, vocabulary_size))
        return ops.mean(metrics.categorical_accuracy(y_true, y_pred))

    return cat_acc


def plate_acc_metric(max_plate_slots: int, vocabulary_size: int):
    """
    Plate accuracy metric.
    """

    def plate_acc(y_true, y_pred):
        """
        Compute how many plates were correctly classified. For a single plate, if ground truth is
        'ABC 123', and the prediction is 'ABC 123', then this would give a score of 1. If the
        prediction was ABD 123, it would score 0.
        """
        y_true = ops.reshape(y_true, newshape=(-1, max_plate_slots, vocabulary_size))
        y_pred = ops.reshape(y_pred, newshape=(-1, max_plate_slots, vocabulary_size))
        y_pred = ops.cast(y_pred, dtype="float32")
        et = ops.equal(ops.argmax(y_true, axis=-1), ops.argmax(y_pred, axis=-1))
        return ops.mean(ops.cast(ops.all(et, axis=-1, keepdims=False), dtype="float32"))

    return plate_acc


def top_3_k_metric(vocabulary_size: int):
    """
    Top 3 K categorical accuracy metric.
    """

    def top_3_k(y_true, y_pred):
        """
        Calculates how often the true character is found in the 3 predictions with the highest
        probability.
        """
        # Reshape into 2-d
        y_true = ops.reshape(y_true, newshape=(-1, vocabulary_size))
        y_pred = ops.reshape(y_pred, newshape=(-1, vocabulary_size))
        y_pred = ops.cast(y_pred, dtype="float32")
        return ops.mean(metrics.top_k_categorical_accuracy(y_true, y_pred, k=3))

    return top_3_k


def plate_len_acc_metric(
    max_plate_slots: int,
    vocabulary_size: int,
    pad_token_index: int,
):
    """
    Plate-length accuracy metric.
    """

    def plate_len_acc(y_true, y_pred):
        """
        Proportion of plates whose predicted length matches the ground-truth length exactly.
        """
        y_true = ops.reshape(y_true, (-1, max_plate_slots, vocabulary_size))
        y_pred = ops.reshape(ops.cast(y_pred, "float32"), (-1, max_plate_slots, vocabulary_size))
        true_idx = ops.argmax(y_true, axis=-1)
        pred_idx = ops.argmax(y_pred, axis=-1)
        true_len = ops.sum(ops.cast(ops.not_equal(true_idx, pad_token_index), "int32"), axis=-1)
        pred_len = ops.sum(ops.cast(ops.not_equal(pred_idx, pad_token_index), "int32"), axis=-1)
        return ops.mean(ops.cast(ops.equal(true_len, pred_len), dtype="float32"))

    return plate_len_acc
