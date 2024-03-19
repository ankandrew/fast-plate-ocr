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

from fast_plate_ocr.config import MAX_PLATE_SLOTS, VOCABULARY_SIZE
from fast_plate_ocr.layer_blocks import block_bn, block_bn_sep_conv_l2, block_no_activation


def modelo_2m(
    h: int,
    w: int,
    dense: bool = True,
    max_plate_slots: int = MAX_PLATE_SLOTS,
    vocabulary_size: int = VOCABULARY_SIZE,
) -> Model:
    """
    2M parameter model that uses normal Convolutional layers (not Depthwise Convolutional layers).
    """
    input_tensor = Input((h, w, 1))
    x = Rescaling(1.0 / 255)(input_tensor)
    # Backbone
    x, _ = block_bn(x)
    x, _ = block_bn(x, k=3, n_c=32, s=1, padding="same")
    x, _ = block_bn(x, k=3, n_c=32, s=1, padding="same")
    x, _ = block_bn(x, k=1, n_c=64, s=1, padding="same")
    x = MaxPool2D(pool_size=(3, 3), strides=(3, 3), padding="same")(x)
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


def modelo_1m_cpu(
    h: int,
    w: int,
    dense: bool = True,
    max_plate_slots: int = MAX_PLATE_SLOTS,
    vocabulary_size: int = VOCABULARY_SIZE,
) -> Model:
    """
    1.2M parameter model that uses Depthwise Convolutional layers, more suitable for low-end devices
    """
    input_tensor = Input((h, w, 1))
    x = Rescaling(1.0 / 255)(input_tensor)
    x, _ = block_bn(x, k=3, n_c=32, s=1, padding="same")
    x, _ = block_bn(x, k=3, n_c=64, s=1, padding="same")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    x, _ = block_bn(x, k=3, n_c=64, s=1, padding="same")
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding="same")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    x, _ = block_bn(x, k=1, n_c=128, s=1, padding="same")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    x, _ = block_bn_sep_conv_l2(x, k=3, n_c=128, s=1, padding="same", depth_multiplier=1)
    x, _ = block_bn(x, k=1, n_c=256, s=1, padding="same")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    x, _ = block_bn_sep_conv_l2(x, k=3, n_c=256, s=1, padding="same", depth_multiplier=1)
    x, _ = block_bn_sep_conv_l2(x, k=1, n_c=512, s=1, padding="same", depth_multiplier=1)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    x, _ = block_bn(x, k=1, n_c=1024, s=1, padding="same")
    x = (
        head(x, max_plate_slots, vocabulary_size)
        if dense
        else head_no_fc(x, max_plate_slots, vocabulary_size)
    )
    return Model(inputs=input_tensor, outputs=x)


def head(x, max_plate_slots: int = MAX_PLATE_SLOTS, vocabulary_size: int = VOCABULARY_SIZE):
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


def head_no_fc(x, max_plate_slots: int = MAX_PLATE_SLOTS, vocabulary_size: int = VOCABULARY_SIZE):
    """
    Model's head without Fully Connected (FC) layers.
    """
    x = block_no_activation(x, k=1, n_c=max_plate_slots * vocabulary_size, s=1, padding="same")
    x = GlobalAveragePooling2D()(x)
    x = Reshape((max_plate_slots, vocabulary_size, 1))(x)
    x = Softmax(axis=-2)(x)
    return x
