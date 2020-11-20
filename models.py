from layer_blocks import block_bn, block_bn_sep_conv_l2, block_no_activation
from tensorflow.keras.activations import softmax
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation, Concatenate, Dense, Dropout,
    GlobalAveragePooling2D, Input, MaxPool2D,
    Lambda, Reshape
)


def modelo_2m(h, w, dense=True):
    '''
    Modelo de 2 millones de parametros
    '''
    input_tensor = Input((h, w, 1))
    # Backbone
    x, _ = block_bn(input_tensor)
    x, _ = block_bn(x, k=3, n_c=32, s=1, padding='same')
    x, _ = block_bn(x, k=3, n_c=32, s=1, padding='same')
    x, _ = block_bn(x, k=1, n_c=64, s=1, padding='same')
    x = MaxPool2D(pool_size=(3, 3), strides=(3, 3), padding='same')(x)
    x, _ = block_bn(x, k=3, n_c=64, s=1, padding='same')
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding='same')
    x, _ = block_bn(x, k=1, n_c=128, s=1, padding='same')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding='same')
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding='same')
    x, _ = block_bn(x, k=1, n_c=256, s=1, padding='same')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x, _ = block_bn(x, k=3, n_c=256, s=1, padding='same')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x, _ = block_bn(x, k=1, n_c=512, s=1, padding='same')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x, _ = block_bn(x, k=1, n_c=1024, s=1, padding='same')
    if dense:
        x = head(x)
    else:
        x = head_no_fc(x)
    return Model(inputs=input_tensor, outputs=x)


def modelo_1m_cpu(h, w, dense=True):
    '''
    Modelo de 1.2 M params
    Reemplaza Conv2D por SeparableConv2d
    para que ejecute mas rapido en CPUs
    '''
    input_tensor = Input((h, w, 1))
    x, _ = block_bn(input_tensor, k=3, n_c=32, s=1, padding='same')
    x, _ = block_bn(x, k=3, n_c=64, s=1, padding='same')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x, _ = block_bn(x, k=3, n_c=64, s=1, padding='same')
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding='same')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x, _ = block_bn(x, k=1, n_c=128, s=1, padding='same')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x, _ = block_bn_sep_conv_l2(
        x, k=3, n_c=128, s=1, padding='same', depth_multiplier=1)
    x, _ = block_bn(x, k=1, n_c=256, s=1, padding='same')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x, _ = block_bn_sep_conv_l2(
        x, k=3, n_c=256, s=1, padding='same', depth_multiplier=1)
    x, _ = block_bn_sep_conv_l2(
        x, k=1, n_c=512, s=1, padding='same', depth_multiplier=1)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x, _ = block_bn(x, k=1, n_c=1024, s=1, padding='same')
    if dense:
        x = head(x)
    else:
        x = head_no_fc(x)
    return Model(inputs=input_tensor, outputs=x)


def head(x):
    '''
    Se encarga de la parte de clasificacion
    de caracteres e incluye Fully Connected Layers
    '''
    x = GlobalAveragePooling2D()(x)
    # dropout for more robust learning
    x = Dropout(0.5)(x)
    x1 = Dense(units=37)(x)
    x2 = Dense(units=37)(x)
    x3 = Dense(units=37)(x)
    x4 = Dense(units=37)(x)
    x5 = Dense(units=37)(x)
    x6 = Dense(units=37)(x)
    x7 = Dense(units=37)(x)
    # Softmax act.
    x1 = Activation(softmax)(x1)
    x2 = Activation(softmax)(x2)
    x3 = Activation(softmax)(x3)
    x4 = Activation(softmax)(x4)
    x5 = Activation(softmax)(x5)
    x6 = Activation(softmax)(x6)
    x7 = Activation(softmax)(x7)
    x = Concatenate()([x1, x2, x3, x4, x5, x6, x7])
    return x


def head_no_fc(x):
    '''
    No incluye fully connected, logra un
    +2.5~ en val_place_acc
    Sin los FC, evitamos un poco mas el overfitting
    '''
    x = block_no_activation(x, k=1, n_c=259, s=1, padding='same')
    x = GlobalAveragePooling2D()(x)
    x = Reshape((7, 37, 1))(x)
    return Lambda(lambda x: softmax(x, axis=-2))(x)
