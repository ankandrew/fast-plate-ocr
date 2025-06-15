"""
Schema definitions for validating supported model architectures and layer blocks.
"""

from typing import Annotated, Literal, TypeAlias

import keras
import yaml
from keras.src.layers import RMSNormalization
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator

from fast_plate_ocr.core.types import PathLike
from fast_plate_ocr.train.model.layers import (
    MLP,
    CoordConv2D,
    DyT,
    MaxBlurPooling2D,
    SqueezeExcite,
)

UnitFloat: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]
"""A float that must be in range of [0, 1]."""
PaddingTypeStr: TypeAlias = Literal["valid", "same"]
"""Padding modes supported by Keras convolution and pooling layers."""
PositiveIntTuple: TypeAlias = PositiveInt | tuple[PositiveInt, PositiveInt]
"""A single positive integer or a tuple of two positive integers, usually used for sizes/strides."""
NormalizationStr: TypeAlias = Literal["layer_norm", "rms_norm", "dyt"]
"""Available normalization layers."""

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
"""Supported Keras activation functions."""

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
"""Keras weight initialization strategies."""


class _Rescaling(BaseModel):
    scale: float = 1.0 / 255
    offset: float = 0.0

    def to_keras_layer(self):
        return keras.layers.Rescaling(self.scale, self.offset)


class _Activation(BaseModel):
    layer: Literal["Activation"]
    activation: ActivationStr

    def to_keras_layer(self) -> keras.layers.Layer:
        return keras.layers.Activation(self.activation)


class _Conv2DBase(BaseModel):
    filters: PositiveInt
    kernel_size: PositiveIntTuple
    strides: PositiveIntTuple = 1
    padding: PaddingTypeStr = "same"
    activation: ActivationStr = "relu"
    use_bias: bool = True
    kernel_initializer: WeightInitializationStr = "he_normal"
    bias_initializer: WeightInitializationStr = "zeros"


class _Conv2D(_Conv2DBase):
    layer: Literal["Conv2D"]

    def to_keras_layer(self) -> keras.layers.Layer:
        params = self.model_dump(exclude={"layer"})
        return keras.layers.Conv2D(**params)


class _CoordConv2D(_Conv2DBase):
    layer: Literal["CoordConv2D"]
    with_r: bool = False

    def to_keras_layer(self) -> keras.layers.Layer:
        params = self.model_dump(exclude={"layer", "with_r"})
        return CoordConv2D(with_r=self.with_r, **params)


class _DepthwiseConv2D(BaseModel):
    layer: Literal["DepthwiseConv2D"]
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


class _SeparableConv2D(BaseModel):
    layer: Literal["SeparableConv2D"]
    filters: PositiveInt
    kernel_size: PositiveIntTuple
    strides: PositiveIntTuple = 1
    padding: PaddingTypeStr = "same"
    depth_multiplier: PositiveInt = 1
    activation: ActivationStr = "relu"
    use_bias: bool = True
    depthwise_initializer: WeightInitializationStr = "he_normal"
    pointwise_initializer: WeightInitializationStr = "glorot_uniform"
    bias_initializer: WeightInitializationStr = "zeros"

    def to_keras_layer(self) -> keras.layers.Layer:
        return keras.layers.SeparableConv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            depth_multiplier=self.depth_multiplier,
            activation=self.activation,
            use_bias=self.use_bias,
            depthwise_initializer=self.depthwise_initializer,
            pointwise_initializer=self.pointwise_initializer,
            bias_initializer=self.bias_initializer,
        )


class _MLP(BaseModel):
    layer: Literal["MLP"]
    hidden_units: list[PositiveInt]
    dropout_rate: UnitFloat = 0.1
    activation: ActivationStr = "gelu"

    def to_keras_layer(self) -> keras.layers.Layer:
        return MLP(
            hidden_units=self.hidden_units,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
        )


class _MaxBlurPooling2D(BaseModel):
    layer: Literal["MaxBlurPooling2D"]
    pool_size: PositiveInt = 2
    filter_size: PositiveInt = 3
    padding: PaddingTypeStr = "same"

    def to_keras_layer(self) -> keras.layers.Layer:
        return MaxBlurPooling2D(
            pool_size=self.pool_size, filter_size=self.filter_size, padding=self.padding
        )


class _MaxPooling2D(BaseModel):
    layer: Literal["MaxPooling2D"]
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
    layer: Literal["AveragePooling2D"]
    pool_size: PositiveIntTuple = 2
    strides: PositiveInt | None = None
    padding: PaddingTypeStr = "valid"

    def to_keras_layer(self) -> keras.layers.Layer:
        return keras.layers.AveragePooling2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
        )


class _ZeroPadding2D(BaseModel):
    layer: Literal["ZeroPadding2D"]
    padding: PositiveIntTuple = 1

    def to_keras_layer(self) -> keras.layers.Layer:
        return keras.layers.ZeroPadding2D(padding=self.padding)


class _SqueezeExcite(BaseModel):
    layer: Literal["SqueezeExcite"]
    ratio: PositiveFloat = 1.0

    def to_keras_layer(self) -> keras.layers.Layer:
        return SqueezeExcite(ratio=self.ratio)


class _BatchNormalization(BaseModel):
    layer: Literal["BatchNormalization"]
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


class _Dropout(BaseModel):
    layer: Literal["Dropout"]
    rate: PositiveFloat

    def to_keras_layer(self):
        return keras.layers.Dropout(rate=self.rate)


class _SpatialDropout2D(BaseModel):
    layer: Literal["SpatialDropout2D"]
    rate: PositiveFloat

    def to_keras_layer(self) -> keras.layers.Layer:
        return keras.layers.SpatialDropout2D(rate=self.rate)


class _GaussianNoise(BaseModel):
    layer: Literal["GaussianNoise"]
    stddev: PositiveFloat
    seed: int | None = None

    def to_keras_layer(self) -> keras.layers.Layer:
        return keras.layers.GaussianNoise(stddev=self.stddev, seed=self.seed)


class _LayerNorm(BaseModel):
    layer: Literal["LayerNorm"]
    epsilon: PositiveFloat = 1e-3

    def to_keras_layer(self) -> keras.layers.Layer:
        return keras.layers.LayerNormalization(epsilon=self.epsilon)


class _RMSNorm(BaseModel):
    layer: Literal["RMSNorm"]
    epsilon: PositiveFloat = 1e-6

    def to_keras_layer(self) -> keras.layers.Layer:
        return RMSNormalization(epsilon=self.epsilon)


class _DyT(BaseModel):
    layer: Literal["DyT"]
    alpha_init_value: PositiveFloat = 0.5

    def to_keras_layer(self) -> keras.layers.Layer:
        return DyT(alpha_init_value=self.alpha_init_value)


LayerConfig = Annotated[
    _Activation
    | _Conv2D
    | _CoordConv2D
    | _DepthwiseConv2D
    | _SeparableConv2D
    | _MLP
    | _MaxBlurPooling2D
    | _MaxPooling2D
    | _AveragePooling2D
    | _ZeroPadding2D
    | _SqueezeExcite
    | _BatchNormalization
    | _Dropout
    | _SpatialDropout2D
    | _GaussianNoise
    | _LayerNorm
    | _RMSNorm
    | _DyT,
    Field(discriminator="layer"),
]


class _CCTTokenizerConfig(BaseModel):
    blocks: list[LayerConfig]
    patch_size: PositiveInt = 1
    patch_dim: PositiveInt | None = None
    patch_mlp: _MLP | None = None
    positional_emb: bool = True


class _CCTTransformerEncoderConfig(BaseModel):
    layers: PositiveInt
    heads: PositiveInt
    projection_dim: PositiveInt
    units: list[PositiveInt]
    activation: ActivationStr = "gelu"
    stochastic_depth: UnitFloat = 0.1
    attention_dropout: UnitFloat = 0.1
    mlp_dropout: UnitFloat = 0.1
    head_mlp_dropout: UnitFloat = 0.2
    token_reducer_heads: PositiveInt = 2
    normalization: NormalizationStr = "layer_norm"

    @model_validator(mode="after")
    def _consistency_checks(self):
        if self.units[-1] != self.projection_dim:
            raise ValueError(
                "'units[-1]' must equal 'projection_dim' "
                f"(got {self.units[-1]} vs {self.projection_dim})."
            )
        return self


class CCTModelConfig(BaseModel):
    model: Literal["cct"] = "cct"
    rescaling: _Rescaling
    tokenizer: _CCTTokenizerConfig
    transformer_encoder: _CCTTransformerEncoderConfig


AnyModelConfig = Annotated[CCTModelConfig, Field(discriminator="model")]
"""Supported model-architecture. New model configs should be added here."""


def load_model_config_from_yaml(yaml_path: PathLike) -> AnyModelConfig:
    """
    Load, parse and validate a YAML file that describes any of the supported model architectures.
    """
    with open(yaml_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return AnyModelConfig(**data)
