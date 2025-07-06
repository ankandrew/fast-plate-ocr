"""
Loss functions for training license plate recognition models.
"""

from keras import losses, ops


def cce_loss(vocabulary_size: int, label_smoothing: float = 0.01):
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


def focal_cce_loss(
    vocabulary_size: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
    label_smoothing: float = 0.01,
):
    """
    Categorical focal cross-entropy loss.
    """

    def cce(y_true, y_pred):
        """
        Computes the focal categorical cross-entropy loss.
        """
        y_true = ops.reshape(y_true, newshape=(-1, vocabulary_size))
        y_pred = ops.reshape(y_pred, newshape=(-1, vocabulary_size))
        return ops.mean(
            losses.categorical_focal_crossentropy(
                y_true,
                y_pred,
                alpha=alpha,
                gamma=gamma,
                from_logits=False,
                label_smoothing=label_smoothing,
            )
        )

    return cce
