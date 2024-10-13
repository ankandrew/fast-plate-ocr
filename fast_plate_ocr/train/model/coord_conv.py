import keras
from keras import Layer, ops
from keras.src.saving import get_custom_objects

FLOATX = keras.config.floatx()
IMAGE_DATA_FORMAT = keras.config.image_data_format()


class CoordinateChannel2D(Layer):
    def __init__(self, use_radius=False, **kwargs):
        super().__init__(**kwargs)
        self.use_radius = use_radius

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        height = ops.shape(inputs)[1]
        width = ops.shape(inputs)[2]
        # Create coordinate channels
        xx_channel = ops.tile(
            ops.reshape(ops.linspace(-1.0, 1.0, width), [1, 1, width, 1]),
            [batch_size, height, 1, 1],
        )
        yy_channel = ops.tile(
            ops.reshape(ops.linspace(-1.0, 1.0, height), [1, height, 1, 1]),
            [batch_size, 1, width, 1],
        )

        # Optionally add the radius channel
        if self.use_radius:
            rr = ops.sqrt(ops.square(xx_channel) + ops.square(yy_channel))
            outputs = ops.concatenate([inputs, xx_channel, yy_channel, rr], axis=-1)
        else:
            outputs = ops.concatenate([inputs, xx_channel, yy_channel], axis=-1)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.use_radius:
            channel_count = 3
        else:
            channel_count = 2
        return input_shape[:-1] + (input_shape[-1] + channel_count,)

    def get_config(self):
        config = super(CoordinateChannel2D, self).get_config()
        config.update({"use_radius": self.use_radius})
        return config


get_custom_objects().update({"CoordinateChannel2D": CoordinateChannel2D})
