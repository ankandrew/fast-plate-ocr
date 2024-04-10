"""
Utils for Keras supported backends.
"""

import os
from typing import Literal, TypeAlias

Framework: TypeAlias = Literal["jax", "tensorflow", "torch"]
"""Supported backend frameworks for Keras."""


def set_jax_backend() -> None:
    """Set Keras backend to jax."""
    set_keras_backend("jax")


def set_tensorflow_backend() -> None:
    """Set Keras backend to tensorflow."""
    set_keras_backend("tensorflow")


def set_pytorch_backend() -> None:
    """Set Keras backend to pytorch."""
    set_keras_backend("torch")


def set_keras_backend(framework: Framework) -> None:
    """Set the Keras backend to a given framework."""
    os.environ["KERAS_BACKEND"] = framework


def reload_keras_backend(framework: Framework) -> None:
    """Reload the Keras backend with a given framework."""
    # pylint: disable=import-outside-toplevel
    import keras

    keras.config.set_backend(framework)
