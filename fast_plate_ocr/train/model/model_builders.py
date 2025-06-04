"""
Model builder functions for supported architectures.
"""

from collections.abc import Sequence

import keras
import numpy as np
from keras import layers

from fast_plate_ocr.train.model.config import PlateOCRConfig
from fast_plate_ocr.train.model.layers import (
    PositionEmbedding,
    TokenReducer,
    TransformerBlock,
    VocabularyProjection,
)
from fast_plate_ocr.train.model.model_schema import AnyModelConfig, CCTModelConfig, LayerConfig


def _build_stem_from_config(specs: Sequence[LayerConfig]) -> keras.Sequential:
    return keras.Sequential([spec.to_keras_layer() for spec in specs], name="conv_stem")


def _build_cct_model(
    cfg: CCTModelConfig,
    input_shape: tuple[int, int, int],
    max_plate_slots: int,
    vocabulary_size: int,
) -> keras.Model:
    # 1. Input
    inputs = layers.Input(shape=input_shape)

    # 2. Rescale & conv stem
    data_rescale = cfg.rescaling.to_keras_layer()
    x = _build_stem_from_config(cfg.tokenizer.blocks)(data_rescale(inputs))

    # 3. Flatten and add positional embedding (if applicable)
    h, w = x.shape[1:3]  # static dims after conv stem
    if h is None or w is None:
        raise ValueError("Conv-stem must produce fixed spatial dims to add position embeddings.")
    # (B, H, W, C) -> (B, H*W, C)
    x = layers.Reshape((-1, x.shape[-1]))(x)

    if cfg.tokenizer.positional_emb:
        seq_len = h * w
        x += PositionEmbedding(sequence_length=seq_len)(x)

    # 4. N x TransformerBlock's
    dpr = list(
        np.linspace(0.0, cfg.transformer_encoder.stochastic_depth, cfg.transformer_encoder.layers)
    )
    for i, rate in enumerate(dpr, 1):
        x = TransformerBlock(
            projection_dim=cfg.transformer_encoder.projection_dim,
            num_heads=cfg.transformer_encoder.heads,
            mlp_units=cfg.transformer_encoder.units,
            attention_dropout=cfg.transformer_encoder.attention_dropout,
            mlp_dropout=cfg.transformer_encoder.mlp_dropout,
            drop_path_rate=rate,
            norm_type=cfg.transformer_encoder.normalization,
            name=f"transformer_block_{i}",
        )(x)

    # 5. Reduce to a fixed number of tokens, then project to vocab
    x = TokenReducer(
        num_tokens=max_plate_slots,
        projection_dim=cfg.transformer_encoder.projection_dim,
        num_heads=cfg.transformer_encoder.token_reducer_heads,
    )(x)

    logits = VocabularyProjection(
        vocabulary_size=vocabulary_size,
        dropout_rate=cfg.transformer_encoder.head_mlp_dropout,
        name="vocab_projection",
    )(x)

    return keras.Model(inputs, logits, name="CCT_OCR")


def build_model(model_cfg: AnyModelConfig, plate_cfg: PlateOCRConfig) -> keras.Model:
    """
    Build a Keras OCR model based on the specified model and plate configuration.
    """
    if model_cfg.model == "cct":
        return _build_cct_model(
            cfg=model_cfg,
            input_shape=(plate_cfg.img_height, plate_cfg.img_width, plate_cfg.num_channels),
            max_plate_slots=plate_cfg.max_plate_slots,
            vocabulary_size=plate_cfg.vocabulary_size,
        )
    raise ValueError(f"Unsupported model type: {model_cfg.model!r}")
