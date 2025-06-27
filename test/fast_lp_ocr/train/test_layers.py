"""
Test for custom layers module.
"""

import inspect
from typing import Type

import keras
import pytest

from fast_plate_ocr.train.model import layers


def _layer_classes() -> list[Type[keras.layers.Layer]]:
    """
    Return every Layer subclass defined in the `layers` module.
    """
    return [
        obj
        for _, obj in inspect.getmembers(layers, inspect.isclass)
        if issubclass(obj, keras.layers.Layer) and obj.__module__ == layers.__name__
    ]


@pytest.mark.parametrize("cls", _layer_classes())
def test_all_layers_are_registered(cls) -> None:
    registered_name = keras.saving.get_registered_name(cls)

    assert registered_name.startswith("fast_plate_ocr>"), (
        f"{cls.__name__} not decorated with @register_keras_serializable"
    )
