"""
Layer blocks used in the OCR model.
"""

from keras import regularizers
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
