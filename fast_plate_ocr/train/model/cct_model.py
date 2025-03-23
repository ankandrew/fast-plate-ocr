import keras
import numpy as np
from keras import layers

positional_emb = True
conv_layers = 2
projection_dim = 128

num_heads = 2
transformer_units = [
    projection_dim,
    projection_dim,
]
transformer_layers = 2
stochastic_depth_rate = 0.1

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
num_epochs = 30
image_size = 32
input_shape = (70, 140, 3)


class CCTTokenizer(layers.Layer):
    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        num_conv_layers=conv_layers,
        num_output_channels=[64, 128],
        positional_emb=positional_emb,
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config

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
        # trim to match the length of the input sequence, which might be less
        # than the sequence_length of the layer.
        position_embeddings = keras.ops.convert_to_tensor(self.position_embeddings)
        position_embeddings = keras.ops.slice(
            position_embeddings,
            (start_index, 0),
            (sequence_length, feature_length),
        )
        return keras.ops.broadcast_to(position_embeddings, shape)

    def compute_output_shape(self, input_shape):
        return input_shape


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


class SequencePooling(layers.Layer):
    def __init__(self):
        super().__init__()
        self.attention = layers.Dense(1)

    def call(self, x):
        attention_weights = keras.ops.softmax(self.attention(x), axis=1)
        attention_weights = keras.ops.transpose(attention_weights, axes=(0, 2, 1))
        weighted_representation = keras.ops.matmul(attention_weights, x)
        return keras.ops.squeeze(weighted_representation, -2)


class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (keras.ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + keras.random.uniform(shape, 0, 1, seed=self.seed_generator)
            random_tensor = keras.ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


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
    input_shape=input_shape,
    num_heads=num_heads,
    projection_dim=projection_dim,
    transformer_units=transformer_units,
    stochastic_depth_rate: float = 0.1,
):
    inputs = layers.Input(input_shape)

    # Augment data.
    augmented = data_augmentation(inputs)

    # Encode patches.
    cct_tokenizer = CCTTokenizer()
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
