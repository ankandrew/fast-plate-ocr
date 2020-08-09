'''
Custom cce, plate_acc and acc for plate recognition using CNN
'''
from tensorflow.keras import backend as K
import tensorflow as tf


# Custom Metrics


def cat_acc(y_true, y_pred):
    y_true = K.reshape(y_true, shape=(-1, 7, 37))
    y_pred = K.reshape(y_pred, shape=(-1, 7, 37))
    return K.mean(tf.keras.metrics.categorical_accuracy(y_true, y_pred))


def plate_acc(y_true, y_pred):
    '''
    How many plates were correctly classified
    If Ground Truth is ABC 123
    Then prediction ABC 123 would score 1
    else ABD 123 would score 0
    Avg these results (1 + 0) / 2 -> Gives .5 accuracy
    (Half of the plates were completely corrected classified)
    '''
    y_true = K.reshape(y_true, shape=(-1, 7, 37))
    y_pred = K.reshape(y_pred, shape=(-1, 7, 37))
    et = K.equal(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(
        K.cast(K.all(et, axis=-1, keepdims=False), dtype='float32')
    )


def top_3_k(y_true, y_pred):
    # Reshape into 2-d
    y_true = K.reshape(y_true, (-1, 37))
    y_pred = K.reshape(y_pred, (-1, 37))
    return K.mean(
        tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
    )

# Custom loss


def cce(y_true, y_pred):
    y_true = K.reshape(y_true, shape=(-1, 37))
    y_pred = K.reshape(y_pred, shape=(-1, 37))

    return K.mean(
        tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=False, label_smoothing=0.2
        )
    )
