import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Add
# print(tf.__version__)


def block_no_bn(i, k=3, n_c=64, s=1, padding='same'):
    x1 = Conv2D(kernel_size=k, filters=n_c, strides=s, padding=padding,
                kernel_regularizer=regularizers.l2(0.01), use_bias=False)(i)
    x2 = Activation(relu)(x1)
    return x2, x1


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


# SIN USAR (se puede experimentar mas)

# def tiny_res_block(x, n_output):
#     # If channels last
#     upscale = K.int_shape(x)[-1] != n_output
#     h = Conv2D(kernel_size=3, filters=n_output, strides=1,
#                padding='same', kernel_regularizer=regularizers.l2(0.01), use_bias=False)(x)
#     h = BatchNormalization()(h)
#     h = Activation(relu)(h)
#     # try k =1
#     h = Conv2D(kernel_size=3, filters=n_output, strides=1,
#                padding='same', kernel_regularizer=regularizers.l2(0.01), use_bias=False)(h)
#     if upscale:
#         f = Conv2D(kernel_size=1, filters=n_output,
#                    strides=1, padding='same')(x)
#     else:
#         f = x
#     x = Add()([f, h])
#     x = BatchNormalization()(x)
#     return Activation(relu)(x)


# def tiny_res_block_1x1(x, n_output):
#     # If channels last
#     upscale = K.int_shape(x)[-1] != n_output
#     h = Conv2D(kernel_size=3, filters=n_output, strides=1,
#                padding='same', kernel_regularizer=regularizers.l2(0.01), use_bias=False)(x)
#     h = BatchNormalization()(h)
#     h = Activation(relu)(h)
#     # try k =1
#     h = Conv2D(kernel_size=1, filters=n_output, strides=1,
#                padding='same', kernel_regularizer=regularizers.l2(0.01), use_bias=False)(h)
#     if upscale:
#         f = Conv2D(kernel_size=1, filters=n_output,
#                    strides=1, padding='same')(x)
#     else:
#         f = x
#     x = Add()([f, h])
#     x = BatchNormalization()(x)
#     return Activation(relu)(x)

# def sam(x):
#     '''
#     YOLO v4 SAM-ish
#     '''
#     # If channels last
#     n_c = K.int_shape(x)[-1]
#     h = Conv2D(kernel_size=1, filters=n_c, strides=1, padding='same')(x)
#     h = BatchNormalization()(h)
#     gate = Activation(sigmoid)(h)
#     return Multiply()([x, gate])


# def se(x):
#     '''
#     Pseudo - Squeeze-and-Excitation
#     '''
#     # If channels last
#     n_c = K.int_shape(x)[-1]
#     h = Conv2D(kernel_size=1, filters=n_c, strides=1, padding='valid')(x)
#     h = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(h)
#     h = Conv2D(kernel_size=1, filters=n_c, strides=3, padding='valid')(h)
#     h = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(h)
#     h = GlobalAveragePooling2D()(h)
#     h = BatchNormalization()(h)
#     gate = Activation(sigmoid)(h)
#     return Multiply()([x, gate])

# def block_bn_db(i, k=3, n_c=64, s=1, padding='same', db_size=5):
#     '''
#     DropBlock + Conv + BN + Act
#     '''
#     x1 = DropBlock2D(block_size=db_size, keep_prob=0.9)(i)
#     x1 = Conv2D(kernel_size=k, filters=n_c, strides=s,
#                 padding=padding, use_bias=False)(x1)
#     x1 = BatchNormalization()(x1)
#     x1 = Activation(relu)(x1)
#     return x1, i
