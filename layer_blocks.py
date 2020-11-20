import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, SeparableConv2D
# print(tf.__version__)


def block_no_bn(i, k=3, n_c=64, s=1, padding='same'):
    x1 = Conv2D(kernel_size=k, filters=n_c, strides=s, padding=padding,
                kernel_regularizer=regularizers.l2(0.01), use_bias=False)(i)
    x2 = Activation(relu)(x1)
    return x2, x1


def block_no_activation(i, k=3, n_c=64, s=1, padding='same'):
    x = Conv2D(kernel_size=k, filters=n_c, strides=s, padding=padding,
               kernel_regularizer=regularizers.l2(0.01), use_bias=False)(i)
    x = BatchNormalization()(x)
    return x


def block_bn(i, k=3, n_c=64, s=1, padding='same'):
    x1 = Conv2D(kernel_size=k, filters=n_c, strides=s, padding=padding,
                kernel_regularizer=regularizers.l2(0.01), use_bias=False)(i)
    x2 = BatchNormalization()(x1)
    x2 = Activation(relu)(x2)
    return x2, x1


def block_bn_no_l2(i, k=3, n_c=64, s=1, padding='same'):
    x1 = Conv2D(kernel_size=k, filters=n_c, strides=s,
                padding=padding, use_bias=False)(i)
    x2 = BatchNormalization()(x1)
    x2 = Activation(relu)(x2)
    return x2, x1


# Para CPU
def block_bn_sep_conv_l2(i, k=3, n_c=64, s=1, padding='same', depth_multiplier=1):
    l2_kernel_reg = regularizers.l2(0.01)
    x1 = SeparableConv2D(kernel_size=k, filters=n_c, depth_multiplier=depth_multiplier,
                         strides=s, padding=padding, use_bias=False,
                         depthwise_regularizer=l2_kernel_reg,
                         pointwise_regularizer=l2_kernel_reg)(i)
    x2 = BatchNormalization()(x1)
    x2 = Activation(relu)(x2)
    return x2, x1


# For Edge TPU
# o bien usar tf.keras.layers.ReLU(6.0)
def relu6(x):
    return K.relu(x, max_value=6)


def block_bn_relu6(i, k=3, n_c=64, s=1, padding='same'):
    x1 = Conv2D(kernel_size=k, filters=n_c, strides=s, padding=padding,
                kernel_regularizer=regularizers.l2(0.01), use_bias=False)(i)
    x2 = BatchNormalization()(x1)
    x2 = Activation(relu6)(x2)
    return x2, x1


def block_bn_relu6_no_l2(i, k=3, n_c=64, s=1, padding='same'):
    x1 = Conv2D(kernel_size=k, filters=n_c, strides=s, padding=padding,
                use_bias=False)(i)
    x2 = BatchNormalization()(x1)
    x2 = Activation(relu6)(x2)
    return x2, x1
