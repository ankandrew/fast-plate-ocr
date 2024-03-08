"""
Custom cce, plate_acc and acc for plate recognition using CNN
"""

from keras import KerasTensor, losses, metrics, ops


def cat_acc(y_true: KerasTensor, y_pred: KerasTensor):
    y_true = ops.reshape(y_true, newshape=(-1, 7, 37))
    y_pred = ops.reshape(y_pred, newshape=(-1, 7, 37))
    return ops.mean(metrics.categorical_accuracy(y_true, y_pred))


def plate_acc(y_true: KerasTensor, y_pred: KerasTensor):
    """
    How many plates were correctly classified
    If Ground Truth is ABC 123
    Then prediction ABC 123 would score 1
    else ABD 123 would score 0
    Avg these results (1 + 0) / 2 -> Gives .5 accuracy
    (Half of the plates were completely corrected classified)
    """
    y_true = ops.reshape(y_true, newshape=(-1, 7, 37))
    y_pred = ops.reshape(y_pred, newshape=(-1, 7, 37))
    et = ops.equal(ops.argmax(y_true), ops.argmax(y_pred))
    return ops.mean(ops.cast(ops.all(et, axis=-1, keepdims=False), dtype="float32"))


def top_3_k(y_true: KerasTensor, y_pred: KerasTensor):
    # Reshape into 2-d
    y_true = ops.reshape(y_true, (-1, 37))
    y_pred = ops.reshape(y_pred, (-1, 37))
    return ops.mean(metrics.top_k_categorical_accuracy(y_true, y_pred, k=3))


# Custom loss
def cce(y_true: KerasTensor, y_pred: KerasTensor):
    y_true = ops.reshape(y_true, newshape=(-1, 37))
    y_pred = ops.reshape(y_pred, newshape=(-1, 37))
    return ops.mean(
        losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.2)
    )
