"""
Model definitions for the FastLP OCR.
"""

from typing import Literal

import kimm
from keras.activations import softmax
from keras.layers import (
    Activation,
    Add,
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Reshape,
    Softmax,
)
from keras.models import Model
from keras.src.layers import Rescaling

from fast_plate_ocr.train.model.layer_blocks import (
    block_average_conv_down,
    block_bn,
    block_max_conv_down,
    block_no_activation,
)


def cnn_ocr_model(
    h: int,
    w: int,
    max_plate_slots: int,
    vocabulary_size: int,
    dense: bool = True,
    activation: str = "relu",
    pool_layer: Literal["avg", "max"] = "max",
) -> Model:
    input_tensor = Input((h, w, 1))  # Define the input tensor
    backbone = kimm.models.MobileViTV2W050(
        input_tensor=input_tensor,
        include_top=False,
        weights=None,
    )
    backbone_output = backbone.output
    x = (
        head(backbone_output, max_plate_slots, vocabulary_size)
        if dense
        else head_no_fc(backbone_output, max_plate_slots, vocabulary_size)
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


def transformer_encoder(inputs, num_heads, key_dim, mlp_dim, dropout=0.1):
    # Layer normalization 1
    x = LayerNormalization(epsilon=1e-6)(inputs)
    # Multi-head attention
    x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    # Skip connection
    x = Add()([x, inputs])

    # Layer normalization 2
    y = LayerNormalization(epsilon=1e-6)(x)
    # MLP
    y = Dense(mlp_dim, activation="relu")(y)
    y = Dropout(dropout)(y)
    y = Dense(inputs.shape[-1])(y)
    # Skip connection
    outputs = Add()([y, x])
    return outputs


def cnn_transformer_ocr_model(
    h: int,
    w: int,
    max_plate_slots: int,
    vocabulary_size: int,
    num_transformer_blocks: int = 8,
    num_heads: int = 4,
    mlp_dim: int = 128,
    activation: str = "relu",
    pool_layer: Literal["avg", "max"] = "max",
) -> Model:
    input_tensor = Input((h, w, 1))
    x = Rescaling(1.0 / 255)(input_tensor)
    # Pooling-Conv layer
    if pool_layer == "avg":
        block_pool_conv = block_average_conv_down
    elif pool_layer == "max":
        block_pool_conv = block_max_conv_down
    # Backbone
    x = block_pool_conv(x, n_c=32, padding="same", activation=activation)
    x, _ = block_bn(x, k=3, n_c=64, s=1, padding="same", activation=activation)
    x, _ = block_bn(x, k=1, n_c=64, s=1, padding="same", activation=activation)
    x = block_pool_conv(x, n_c=64, padding="same", activation=activation)
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding="same", activation=activation)
    x, _ = block_bn(x, k=1, n_c=128, s=1, padding="same", activation=activation)
    x = block_pool_conv(x, n_c=128, padding="same", activation=activation)
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding="same", activation=activation)

    # Flatten spatial dimensions to sequence
    x = Reshape((-1, x.shape[-1]))(x)  # Shape: (batch_size, seq_length, channels)

    # Transformer Encoder Blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(
            x,
            num_heads=num_heads,
            key_dim=x.shape[-1] // num_heads,
            mlp_dim=mlp_dim,
            dropout=0.1,
        )

    # Global Pooling
    x = GlobalAveragePooling1D()(x)

    # Output Layers
    x = Dropout(0.5)(x)
    dense_outputs = [
        Activation(softmax)(Dense(units=vocabulary_size)(x)) for _ in range(max_plate_slots)
    ]
    x = Concatenate()(dense_outputs)

    return Model(inputs=input_tensor, outputs=x)
