from typing import Literal, TypeAlias

import keras
import yaml
from keras.src.layers import RMSNormalization
from pydantic import BaseModel, PositiveFloat, PositiveInt

from fast_plate_ocr.train.model.layer_blocks import (
    CoordConv2D,
    DyT,
    MaxBlurPooling2D,
    SqueezeExcite,
)

ActivationStr: TypeAlias = Literal[
    "celu",
    "elu",
    "exponential",
    "gelu",
    "glu",
    "hard_shrink",
    "hard_sigmoid",
    "hard_silu",
    "hard_tanh",
    "leaky_relu",
    "linear",
    "log_sigmoid",
    "log_softmax",
    "mish",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "soft_shrink",
    "softmax",
    "softplus",
    "softsign",
    "sparse_plus",
    "sparsemax",
    "squareplus",
    "tanh",
    "tanh_shrink",
    "threshold",
]


WeightInitializationStr: TypeAlias = Literal[
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "lecun_normal",
    "lecun_uniform",
    "ones",
    "random_normal",
    "random_uniform",
    "truncated_normal",
    "variance_scaling",
    "zeros",
]

PaddingTypeStr: TypeAlias = Literal["valid", "same"]
PositiveIntTuple: TypeAlias = PositiveInt | tuple[PositiveInt, PositiveInt]


class _Conv2D(BaseModel):
    filters: PositiveInt
    kernel_size: PositiveIntTuple
    strides: PositiveIntTuple = 1
    padding: PaddingTypeStr = "same"
    activation: ActivationStr = "relu"
    use_bias: bool = True
    kernel_initializer: WeightInitializationStr = "he_normal"
    bias_initializer: WeightInitializationStr = "zeros"

    def to_keras_layer(self):
        return keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )


class _CoordConv2D(_Conv2D):
    with_r: bool = False

    def to_keras_layer(self) -> keras.layers.Layer:
        conv_args = self.model_dump(exclude={"with_r"})
        return CoordConv2D(with_r=self.with_r, **conv_args)


class _DepthwiseConv2D(BaseModel):
    kernel_size: PositiveIntTuple
    strides: PositiveIntTuple = 1
    padding: PaddingTypeStr = "same"
    depth_multiplier: PositiveInt = 1
    activation: ActivationStr = "relu"
    use_bias: bool = True
    depthwise_initializer: WeightInitializationStr = "he_normal"
    bias_initializer: WeightInitializationStr = "zeros"

    def to_keras_layer(self) -> keras.layers.Layer:
        return keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            depth_multiplier=self.depth_multiplier,
            activation=self.activation,
            use_bias=self.use_bias,
            depthwise_initializer=self.depthwise_initializer,
            bias_initializer=self.bias_initializer,
        )


class _MaxBlurPooling2D(BaseModel):
    pool_size: PositiveInt = 2
    filter_size: PositiveInt = 3

    def to_keras_layer(self) -> keras.layers.Layer:
        return MaxBlurPooling2D(pool_size=self.pool_size, filter_size=self.filter_size)


class _MaxPooling2D(BaseModel):
    pool_size: PositiveIntTuple = 2
    strides: PositiveInt | None = None
    padding: PaddingTypeStr = "valid"

    def to_keras_layer(self) -> keras.layers.Layer:
        return keras.layers.MaxPooling2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
        )


class _AveragePooling2D(BaseModel):
    pool_size: PositiveIntTuple = 2
    strides: PositiveInt | None = None
    padding: PaddingTypeStr = "valid"

    def to_keras_layer(self) -> keras.layers.Layer:
        return keras.layers.AveragePooling2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
        )


class _SqueezeExcite(BaseModel):
    ratio: PositiveFloat = 1.0

    def to_keras_layer(self) -> keras.layers.Layer:
        return SqueezeExcite(ratio=self.ratio)


# TODO: Add residual & repeat blocks?


class _BatchNormalization(BaseModel):
    momentum: PositiveFloat = 0.99
    epsilon: PositiveFloat = 1e-3
    center: bool = True
    scale: bool = True

    def to_keras_layer(self) -> keras.layers.Layer:
        return keras.layers.BatchNormalization(
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
        )


class _LayerNorm(BaseModel):
    epsilon: PositiveFloat = 1e-3

    def to_keras_layer(self) -> keras.layers.Layer:
        return keras.layers.LayerNormalization(epsilon=self.epsilon)


class _RMSNorm(BaseModel):
    epsilon: PositiveFloat = 1e-6

    def to_keras_layer(self) -> keras.layers.Layer:
        return RMSNormalization(epsilon=self.epsilon)


class _DyT(BaseModel):
    alpha_init_value: PositiveFloat = 0.5

    def to_keras_layer(self) -> keras.layers.Layer:
        return DyT(alpha_init_value=self.alpha_init_value)


# Example YAML configuration
yaml_content = """
layers:
  - filters: 32
    kernel_size: 3
    strides: 1
  - filters: 64
    kernel_size: 3
    strides: 2
"""

# Parse YAML and create Conv2D instances
config = yaml.safe_load(yaml_content)
conv_blocks = [_CoordConv2D(**layer) for layer in config["layers"]]

# Map each configuration to a Keras layer
keras_layers = [conv_block.to_keras_layer() for conv_block in conv_blocks]

print(keras_layers)
