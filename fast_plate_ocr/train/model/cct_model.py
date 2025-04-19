from collections.abc import Sequence

import keras
import numpy as np
from keras import layers

# positional_emb = True
# conv_layers = 2
# projection_dim = 128
#
# num_heads = 2
# transformer_units = [
#     projection_dim,
#     projection_dim,
# ]
# transformer_layers = 2
# stochastic_depth_rate = 0.1
#
# learning_rate = 0.001
# weight_decay = 0.0001
# batch_size = 128
# num_epochs = 30
# image_size = 32
# input_shape = (70, 140, 3)


class CCTTokenizer(layers.Layer):
    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        num_conv_layers: int = 2,
        num_output_channels: Sequence[int] = (64, 128),
        positional_emb: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # This is our tokenizer.
        self.conv_model = keras.Sequential()
        for i in range(num_conv_layers):
            self.conv_model.add(
                layers.Conv2D(
                    num_output_channels[i],
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            self.conv_model.add(layers.ZeroPadding2D(padding))
            self.conv_model.add(layers.MaxPooling2D(pooling_kernel_size, pooling_stride, "same"))

        self.positional_emb = positional_emb

    def call(self, images):
        outputs = self.conv_model(images)
        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences.
        reshaped = keras.ops.reshape(
            outputs,
            (
                -1,
                keras.ops.shape(outputs)[1] * keras.ops.shape(outputs)[2],
                keras.ops.shape(outputs)[-1],
            ),
        )
        return reshaped


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
        """
        Args:
            num_tokens: The number of output tokens (e.g. max plate characters).
            projection_dim: The dimension of the token embeddings.
            num_heads: Number of attention heads.
        """
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)

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
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop
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


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.ops.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def vocab_projection(x, vocabulary_size, dropout_rate=None):
    if dropout_rate is not None:
        x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)
    x = layers.TimeDistributed(layers.Dense(vocabulary_size, activation="softmax"))(x)
    return x


# Note the rescaling layer. These layers have pre-defined inference behavior.
data_augmentation = keras.Sequential(
    [
        layers.Rescaling(scale=1.0 / 255),
    ],
    name="data_augmentation",
)


def create_cct_model(
    max_plate_slots: int,
    vocabulary_size: int,
    transformer_layers: int,
    conv_layers: int,
    num_output_channels: Sequence[int],
    input_shape: tuple[int, int, int],
    num_heads: int,
    projection_dim: int,
    transformer_units: Sequence[int],
    stochastic_depth_rate: float = 0.1,
    positional_emb: bool = True,
):
    inputs = layers.Input(input_shape)

    # Augment data.
    augmented = data_augmentation(inputs)

    # Encode patches.
    cct_tokenizer = CCTTokenizer(
        num_conv_layers=conv_layers,
        num_output_channels=num_output_channels,
        positional_emb=positional_emb,
    )
    encoded_patches = cct_tokenizer(augmented)

    # Apply positional embedding.
    if positional_emb:
        sequence_length = encoded_patches.shape[1]
        encoded_patches += PositionEmbedding(sequence_length=sequence_length)(encoded_patches)

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    reduced_tokens = TokenReducer(num_tokens=max_plate_slots, projection_dim=projection_dim)(
        encoded_patches
    )
    logits = vocab_projection(reduced_tokens, vocabulary_size)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


#
#
# model = create_cct_model(
#     max_plate_slots=9,
#     vocabulary_size=37,
# )
#
# dummy_input = np.random.rand(1, 70, 140, 3)
# y = model.predict(dummy_input)
# pass
