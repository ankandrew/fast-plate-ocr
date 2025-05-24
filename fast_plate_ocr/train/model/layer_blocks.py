"""
Layer blocks used in the OCR model.
"""

import keras
import numpy as np
from keras import ops, regularizers
from keras.src.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    SeparableConv2D,
)


def block_no_bn(i, k=3, n_c=64, s=1, padding="same", activation: str = "relu"):
    x1 = Conv2D(
        kernel_size=k,
        filters=n_c,
        strides=s,
        padding=padding,
        kernel_regularizer=regularizers.l2(0.01),
        use_bias=False,
    )(i)
    x2 = Activation(activation)(x1)
    return x2, x1


def block_no_activation(i, k=3, n_c=64, s=1, padding="same"):
    x = Conv2D(
        kernel_size=k,
        filters=n_c,
        strides=s,
        padding=padding,
        kernel_regularizer=regularizers.l2(0.01),
        use_bias=False,
    )(i)
    x = BatchNormalization()(x)
    return x


def block_bn(i, k=3, n_c=64, s=1, padding="same", activation: str = "relu"):
    x1 = Conv2D(
        kernel_size=k,
        filters=n_c,
        strides=s,
        padding=padding,
        kernel_regularizer=regularizers.l2(0.01),
        use_bias=False,
    )(i)
    x2 = BatchNormalization()(x1)
    x2 = Activation(activation)(x2)
    return x2, x1


def block_bn_no_l2(i, k=3, n_c=64, s=1, padding="same", activation: str = "relu"):
    x1 = Conv2D(kernel_size=k, filters=n_c, strides=s, padding=padding, use_bias=False)(i)
    x2 = BatchNormalization()(x1)
    x2 = Activation(activation)(x2)
    return x2, x1


def block_bn_sep_conv_l2(
    i, k=3, n_c=64, s=1, padding="same", depth_multiplier=1, activation: str = "relu"
):
    l2_kernel_reg = regularizers.l2(0.01)
    x1 = SeparableConv2D(
        kernel_size=k,
        filters=n_c,
        depth_multiplier=depth_multiplier,
        strides=s,
        padding=padding,
        use_bias=False,
        depthwise_regularizer=l2_kernel_reg,
        pointwise_regularizer=l2_kernel_reg,
    )(i)
    x2 = BatchNormalization()(x1)
    x2 = Activation(activation)(x2)
    return x2, x1


def block_bn_relu6(i, k=3, n_c=64, s=1, padding="same", activation: str = "relu6"):
    x1 = Conv2D(
        kernel_size=k,
        filters=n_c,
        strides=s,
        padding=padding,
        kernel_regularizer=regularizers.l2(0.01),
        use_bias=False,
    )(i)
    x2 = BatchNormalization()(x1)
    x2 = Activation(activation)(x2)
    return x2, x1


def block_bn_relu6_no_l2(i, k=3, n_c=64, s=1, padding="same", activation: str = "relu6"):
    x1 = Conv2D(kernel_size=k, filters=n_c, strides=s, padding=padding, use_bias=False)(i)
    x2 = BatchNormalization()(x1)
    x2 = Activation(activation)(x2)
    return x2, x1


def block_average_conv_down(x, n_c, padding="same", activation: str = "relu"):
    x = AveragePooling2D(pool_size=2, strides=1, padding="valid")(x)
    x = Conv2D(
        filters=n_c,
        kernel_size=3,
        strides=2,
        padding=padding,
        kernel_regularizer=regularizers.l2(0.01),
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def block_max_conv_down(x, n_c, padding="same", activation: str = "relu"):
    x = MaxPooling2D(pool_size=2, strides=1, padding="valid")(x)
    x = Conv2D(
        filters=n_c,
        kernel_size=3,
        strides=2,
        padding=padding,
        kernel_regularizer=regularizers.l2(0.01),
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


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
    def __init__(self, pool_size: int = 2, filter_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.filter_size = filter_size

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
            padding="same",
        )
        x = ops.depthwise_conv(
            x, self.blur_kernel, padding="same", strides=(self.pool_size, self.pool_size)
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
        config.update({"pool_size": self.pool_size, "filter_size": self.filter_size})
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
            units=filters // self.ratio,
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


@keras.utils.register_keras_serializable(package="custom_layers")
class DyT(keras.layers.Layer):
    """
    Dynamic Tanh (DyT), is an element-wise operation as a drop-in replacement for normalization
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
