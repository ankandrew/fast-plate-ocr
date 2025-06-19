"""
Layer blocks used in the OCR model.
"""

from collections.abc import Sequence

import keras
import numpy as np
from keras import ops

# pylint: disable=too-many-ancestors,abstract-method,attribute-defined-outside-init,arguments-differ
# pylint: disable=useless-parent-delegation


class AddCoords(keras.layers.Layer):
    """Add coords to a tensor, modified from paper: https://arxiv.org/abs/1807.03247"""

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def build(self, input_shape):
        # Assuming input_shape is (batch, height, width, channels)
        self.x_dim = input_shape[1]
        self.y_dim = input_shape[2]

    def call(self, input_tensor):
        """
        input_tensor: (batch, x_dim, y_dim, c)
        """
        batch_size_tensor = ops.shape(input_tensor)[0]
        xx_ones = ops.ones([batch_size_tensor, self.x_dim])
        xx_ones = ops.expand_dims(xx_ones, -1)
        xx_range = ops.tile(ops.expand_dims(ops.arange(self.y_dim), 0), [batch_size_tensor, 1])
        xx_range = ops.expand_dims(xx_range, 1)
        xx_channel = ops.matmul(xx_ones, xx_range)
        xx_channel = ops.expand_dims(xx_channel, -1)
        yy_ones = ops.ones([batch_size_tensor, self.y_dim])
        yy_ones = ops.expand_dims(yy_ones, 1)
        yy_range = ops.tile(ops.expand_dims(ops.arange(self.x_dim), 0), [batch_size_tensor, 1])

        yy_range = ops.expand_dims(yy_range, -1)
        yy_channel = ops.matmul(yy_range, yy_ones)
        yy_channel = ops.expand_dims(yy_channel, -1)
        xx_channel = ops.cast(xx_channel, "float32") / (self.x_dim - 1)
        yy_channel = ops.cast(yy_channel, "float32") / (self.y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        ret = ops.concatenate([input_tensor, xx_channel, yy_channel], axis=-1)
        if self.with_r:
            rr = ops.sqrt(ops.square(xx_channel) + ops.square(yy_channel))
            ret = ops.concatenate([ret, rr], axis=-1)
        return ret


@keras.saving.register_keras_serializable(package="fast_plate_ocr")
class CoordConv2D(keras.layers.Layer):
    """CoordConv2D layer as in the paper, modified from paper: https://arxiv.org/abs/1807.03247"""

    def __init__(self, with_r: bool = False, **conv_kwargs):
        super().__init__()
        self.with_r = with_r
        self.conv_kwargs = conv_kwargs.copy()
        self.addcoords = AddCoords(with_r=with_r)
        self.conv = keras.layers.Conv2D(**conv_kwargs)

    def call(self, inputs):
        x = self.addcoords(inputs)
        return self.conv(x)

    def get_config(self):
        config = super().get_config()
        config.update({"with_r": self.with_r, **self.conv_kwargs})
        return config


def _build_binomial_filter(filter_size: int) -> np.ndarray:
    """Builds and returns the normalized binomial filter according to `filter_size`."""
    if filter_size == 1:
        binomial_filter = np.array([1.0])
    elif filter_size == 2:
        binomial_filter = np.array([1.0, 1.0])
    elif filter_size == 3:
        binomial_filter = np.array([1.0, 2.0, 1.0])
    elif filter_size == 4:
        binomial_filter = np.array([1.0, 3.0, 3.0, 1.0])
    elif filter_size == 5:
        binomial_filter = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    elif filter_size == 6:
        binomial_filter = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
    elif filter_size == 7:
        binomial_filter = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
    else:
        raise ValueError(f"Filter size not supported, got {filter_size}")

    binomial_filter = binomial_filter[:, np.newaxis] * binomial_filter[np.newaxis, :]
    binomial_filter = binomial_filter / np.sum(binomial_filter)

    return binomial_filter


@keras.saving.register_keras_serializable(package="fast_plate_ocr")
class MaxBlurPooling2D(keras.layers.Layer):
    def __init__(self, pool_size: int = 2, filter_size: int = 3, padding: str = "same", **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.filter_size = filter_size
        self.padding = padding

        super().__init__(**kwargs)

    def build(self, input_shape):
        binomial_filter = _build_binomial_filter(filter_size=self.filter_size)
        binomial_filter = np.repeat(binomial_filter, input_shape[3])
        # Maybe this should be channel first/last agnostic
        binomial_filter = np.reshape(
            binomial_filter, (self.filter_size, self.filter_size, input_shape[3], 1)
        )
        blur_init = keras.initializers.constant(binomial_filter)

        self.blur_kernel = self.add_weight(
            name="blur_kernel",
            shape=(self.filter_size, self.filter_size, input_shape[3], 1),
            initializer=blur_init,
            trainable=False,
        )

        super().build(input_shape)

    def call(self, x):
        x = ops.max_pool(
            x,
            (self.pool_size, self.pool_size),
            strides=(1, 1),
            padding=self.padding,
        )
        x = ops.depthwise_conv(
            x, self.blur_kernel, padding=self.padding, strides=(self.pool_size, self.pool_size)
        )

        return x

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            int(np.ceil(input_shape[1] / 2)),
            int(np.ceil(input_shape[2] / 2)),
            input_shape[3],
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pool_size": self.pool_size,
                "filter_size": self.filter_size,
                "padding": self.padding,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="fast_plate_ocr")
class SqueezeExcite(keras.layers.Layer):
    """
    Applies squeeze and excitation to input feature maps as seen in https://arxiv.org/abs/1709.01507

    Note: this was taken from https://keras.io/examples/vision/patch_convnet.
    """

    def __init__(self, ratio: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

    def build(self, input_shape):
        filters = input_shape[-1]
        self.squeeze = keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.reduction = keras.layers.Dense(
            units=int(filters // self.ratio),
            activation="relu",
            use_bias=False,
        )
        self.excite = keras.layers.Dense(units=filters, activation="sigmoid", use_bias=False)
        self.multiply = keras.layers.Multiply()

    def call(self, x):
        shortcut = x
        x = self.squeeze(x)
        x = self.reduction(x)
        x = self.excite(x)
        x = self.multiply([shortcut, x])
        return x


@keras.utils.register_keras_serializable(package="fast_plate_ocr")
class DyT(keras.layers.Layer):
    """
    Dynamic Tanh (DyT) is an element-wise operation as a drop-in replacement for normalization
    layers in Transformers.

    Paper: https://arxiv.org/abs/2503.10622.
    """

    def __init__(self, alpha_init_value: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha_init_value = alpha_init_value

    def build(self, input_shape):
        channels = int(input_shape[-1])

        # scalar alpha
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=keras.initializers.Constant(self.alpha_init_value),
            trainable=True,
        )

        self.weight = self.add_weight(
            name="weight",
            shape=(channels,),
            initializer="ones",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(channels,),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x):
        x = keras.ops.tanh(self.alpha * x)
        return x * self.weight + self.bias

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha_init_value": self.alpha_init_value})
        return cfg


def build_norm_layer(norm_type) -> keras.layers.Layer:
    if norm_type == "layer_norm":
        return keras.layers.LayerNormalization(epsilon=1e-5)
    if norm_type == "rms_norm":
        return keras.layers.RMSNormalization(epsilon=1e-5)
    if norm_type == "dyt":
        return DyT(alpha_init_value=0.5)
    raise ValueError(f"Unknown norm_type {norm_type}")


@keras.saving.register_keras_serializable(package="fast_plate_ocr")
class PositionEmbedding(keras.layers.Layer):
    def __init__(
        self,
        sequence_length,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError("`sequence_length` must be an Integer, received `None`.")
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, start_index=0):
        shape = keras.ops.shape(inputs)
        feature_length = shape[-1]
        sequence_length = shape[-2]
        # trim to match the length of the input sequence, which might be less than the
        # sequence_length of the layer.
        position_embeddings = keras.ops.convert_to_tensor(self.position_embeddings)
        position_embeddings = keras.ops.slice(
            position_embeddings,
            (start_index, 0),
            (sequence_length, feature_length),
        )
        return keras.ops.broadcast_to(position_embeddings, shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config


@keras.saving.register_keras_serializable(package="fast_plate_ocr")
class TokenReducer(keras.layers.Layer):
    def __init__(self, num_tokens, projection_dim, num_heads=2, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)

    def build(self, input_shape):
        self.query_tokens = self.add_weight(
            shape=(1, self.num_tokens, self.projection_dim),
            initializer="random_normal",
            trainable=True,
            name="query_tokens",
        )
        # input_shape is assumed to be (batch_size, seq_length, projection_dim)
        seq_length = input_shape[1]
        if seq_length is None:
            raise ValueError("Input sequence length must be defined (not None).")
        self.attn.build(
            query_shape=(1, self.num_tokens, self.projection_dim),
            value_shape=(1, seq_length, self.projection_dim),
        )
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.num_tokens, self.projection_dim

    def call(self, inputs):
        """
        inputs: Tensor of shape (batch_size, seq_length, projection_dim)
        returns: Tensor of shape (batch_size, num_tokens, projection_dim)
        """
        batch_size = keras.ops.shape(inputs)[0]
        # Tile the learned query tokens for each example in the batch.
        query_tokens = keras.ops.tile(self.query_tokens, [batch_size, 1, 1])
        # Perform cross-attention where the queries are the learned tokens and keys/values are the
        # input tokens.
        reduced_tokens = self.attn(query=query_tokens, key=inputs, value=inputs)
        return reduced_tokens

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "num_tokens": self.num_tokens,
                "projection_dim": self.projection_dim,
                "num_heads": self.num_heads,
            }
        )
        return cfg


@keras.saving.register_keras_serializable(package="fast_plate_ocr")
class StochasticDepth(keras.layers.Layer):
    def __init__(self, drop_prob: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (keras.ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + keras.random.uniform(
                shape, 0, 1, seed=self.seed_generator, dtype=x.dtype
            )
            random_tensor = keras.ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_prob": self.drop_prob})
        return cfg


@keras.saving.register_keras_serializable(package="fast_plate_ocr")
class MLP(keras.layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate: float = 0.1,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_units = list(hidden_units)
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.dense_layers = []
        self.dropout_layers = []
        for units in self.hidden_units:
            self.dense_layers.append(keras.layers.Dense(units, activation=self.activation))
            self.dropout_layers.append(keras.layers.Dropout(self.dropout_rate))

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        for dense, drop in zip(self.dense_layers, self.dropout_layers, strict=True):
            x = dense(x)
            x = drop(x, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "hidden_units": self.hidden_units,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
            }
        )
        return cfg


@keras.saving.register_keras_serializable(package="fast_plate_ocr")
class VocabularyProjection(keras.layers.Layer):
    def __init__(self, vocabulary_size: int, dropout_rate: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.dropout_rate = dropout_rate
        self.dropout = (
            keras.layers.Dropout(self.dropout_rate) if self.dropout_rate is not None else None
        )
        self.classifier = keras.layers.Dense(self.vocabulary_size, activation="softmax")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=None):
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return self.classifier(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"vocabulary_size": self.vocabulary_size, "dropout_rate": self.dropout_rate})
        return cfg


@keras.saving.register_keras_serializable(package="fast_plate_ocr")
class TransformerBlock(keras.layers.Layer):
    def __init__(
        self,
        projection_dim: int,
        num_heads: int,
        mlp_units: Sequence[int],
        attention_dropout: float,
        mlp_dropout: float,
        drop_path_rate: float,
        norm_type: str | None = "layer_norm",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_type = norm_type
        self.norm1 = build_norm_layer(norm_type)
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=attention_dropout
        )
        self.drop1 = StochasticDepth(drop_path_rate)
        self.norm2 = build_norm_layer(norm_type)
        self.mlp = MLP(hidden_units=mlp_units, dropout_rate=mlp_dropout)
        self.drop2 = StochasticDepth(drop_path_rate)

    def build(self, input_shape) -> None:
        super().build(input_shape)

    def call(self, x, training=None):
        # 1. MHA + residual
        y = self.norm1(x)
        y = self.attn(y, y)
        y = self.drop1(y, training=training)
        x = keras.layers.Add()([x, y])

        # 2. MLP + residual
        y = self.norm2(x)
        y = self.mlp(y, training=training)
        y = self.drop2(y, training=training)
        return keras.layers.Add()([x, y])

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "projection_dim": self.attn.key_dim,
                "num_heads": self.attn.num_heads,
                "mlp_units": self.mlp.hidden_units,
                "mlp_dropout": self.mlp.dropout_rate,
                "attention_dropout": self.attn.dropout,
                "drop_path_rate": self.drop1.drop_prob,
                "norm_type": self.norm_type,
            }
        )
        return cfg
