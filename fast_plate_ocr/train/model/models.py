"""
Model definitions for the FastLP OCR.
"""

from typing import Literal

from keras import layers, ops
from keras.activations import gelu, softmax
from keras.layers import (
    Activation,
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Rescaling,
    Reshape,
    Softmax,
)
from keras.models import Model

from fast_plate_ocr.train.model.layer_blocks import (
    block_average_conv_down,
    block_bn,
    block_max_conv_down,
    block_no_activation,
)


def vitstr_tiny(
    max_plate_slots,
    vocabulary_size,
    input_shape=(96, 96, 1),  # License plate image input (grayscale)
    patch_size=16,
    embed_dim=192,
    depth=12,
    num_heads=3,
    mlp_ratio=4,
    dropout_rate=0.1,
):
    # Input layer for grayscale image
    inputs = layers.Input(shape=input_shape)
    # Rescaling input to [0, 1]
    x = layers.Rescaling(1.0 / 255)(inputs)
    # Patching
    num_patches = (input_shape[0] // patch_size) ** 2
    patches = layers.Conv2D(embed_dim, patch_size, strides=patch_size)(x)
    patches = layers.Reshape((num_patches, embed_dim))(patches)
    # Learnable position embeddings
    positions = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(
        ops.arange(start=0, stop=num_patches, step=1)
    )
    # Adding position embeddings
    encoded_patches = patches + positions
    # Transformer blocks
    for _ in range(depth):
        # Layer normalization
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )(x1, x1)
        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        mlp_hidden_dim = embed_dim * mlp_ratio
        mlp_output = layers.Dense(mlp_hidden_dim, activation=gelu)(x3)
        mlp_output = layers.Dropout(dropout_rate)(mlp_output)
        mlp_output = layers.Dense(embed_dim)(mlp_output)
        # Skip connection 2
        encoded_patches = layers.Add()([mlp_output, x2])
    # Global Average Pooling
    x = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    # Output head (MLP)
    x = layers.Dense(embed_dim, activation=gelu)(x)
    outputs = [layers.Dense(vocabulary_size, activation=softmax)(x) for _ in range(max_plate_slots)]
    # Concatenate the outputs for the sequence of characters
    outputs = layers.Concatenate()(outputs)
    # Create model
    model = Model(inputs, outputs)
    return model


def cnn_ocr_model(
    h: int,
    w: int,
    max_plate_slots: int,
    vocabulary_size: int,
    dense: bool = True,
    activation: str = "relu",
    pool_layer: Literal["avg", "max"] = "max",
) -> Model:
    """
    OCR model implemented with just CNN layers (v2).
    """
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
    x, _ = block_bn(x, k=1, n_c=256, s=1, padding="same", activation=activation)
    x = block_pool_conv(x, n_c=256, padding="same", activation=activation)
    x, _ = block_bn(x, k=1, n_c=512, s=1, padding="same", activation=activation)
    x, _ = block_bn(x, k=1, n_c=1024, s=1, padding="same", activation=activation)
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
