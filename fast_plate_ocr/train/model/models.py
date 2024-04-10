"""
Model definitions for the FastLP OCR.
"""

from keras.activations import softmax
from keras.layers import (
    Activation,
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
    Rescaling,
    Reshape,
    Softmax,
)
from keras.models import Model

from fast_plate_ocr.train.model.layer_blocks import block_bn, block_no_activation


def cnn_ocr_model(
    h: int,
    w: int,
    max_plate_slots: int,
    vocabulary_size: int,
    dense: bool = True,
) -> Model:
    """
    OCR model implemented with just CNN layers.
    """
    input_tensor = Input((h, w, 1))
    x = Rescaling(1.0 / 255)(input_tensor)
    # Backbone
    x, _ = block_bn(x, k=3, n_c=32, s=2, padding="same")
    x, _ = block_bn(x, k=3, n_c=64, s=1, padding="same")
    x, _ = block_bn(x, k=1, n_c=64, s=1, padding="same")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    x, _ = block_bn(x, k=3, n_c=64, s=1, padding="same")
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding="same")
    x, _ = block_bn(x, k=1, n_c=128, s=1, padding="same")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding="same")
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding="same")
    x, _ = block_bn(x, k=1, n_c=256, s=1, padding="same")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    x, _ = block_bn(x, k=3, n_c=256, s=1, padding="same")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    x, _ = block_bn(x, k=1, n_c=512, s=1, padding="same")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    x, _ = block_bn(x, k=1, n_c=1024, s=1, padding="same")
    x = (
        head(x, max_plate_slots, vocabulary_size)
        if dense
        else head_no_fc(x, max_plate_slots, vocabulary_size)
    )
    return Model(inputs=input_tensor, outputs=x)


def head(x, max_plate_slots: int, vocabulary_size: int):
    """
    Model's head with Fully Connected (FC) layers.
    """
    x = GlobalAveragePooling2D()(x)
    # dropout for more robust learning
    x = Dropout(0.5)(x)
    dense_outputs = [
        Activation(softmax)(Dense(units=vocabulary_size)(x)) for _ in range(max_plate_slots)
    ]
    # concat all the dense outputs
    x = Concatenate()(dense_outputs)
    return x


def head_no_fc(x, max_plate_slots: int, vocabulary_size: int):
    """
    Model's head without Fully Connected (FC) layers.
    """
    x = block_no_activation(x, k=1, n_c=max_plate_slots * vocabulary_size, s=1, padding="same")
    x = GlobalAveragePooling2D()(x)
    x = Reshape((max_plate_slots, vocabulary_size, 1))(x)
    x = Softmax(axis=-2)(x)
    return x
