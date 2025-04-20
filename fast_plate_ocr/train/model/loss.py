"""
Loss functions for training license plate recognition models.
"""

import keras
from keras.api import losses, ops


@keras.saving.register_keras_serializable(package="fast_plate_ocr")
def cce_loss(vocabulary_size: int, label_smoothing: float = 0.1):
    """
    Categorical cross-entropy loss.
    """

    def cce(y_true, y_pred):
        """
        Computes the categorical cross-entropy loss.
        """
        y_true = ops.reshape(y_true, newshape=(-1, vocabulary_size))
        y_pred = ops.reshape(y_pred, newshape=(-1, vocabulary_size))
        return ops.mean(
            losses.categorical_crossentropy(
                y_true, y_pred, from_logits=False, label_smoothing=label_smoothing
            )
        )

    return cce
