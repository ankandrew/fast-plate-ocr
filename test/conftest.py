"""
The `conftest.py` file serves as a means of providing fixtures for an entire directory. Fixtures
defined in a `conftest.py` can be used by any test in that package without needing to import them
(pytest will automatically discover them).
"""

import pathlib
import tempfile
import textwrap
from collections.abc import Iterator

import cv2
import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def dummy_dataset() -> Iterator[pathlib.Path]:
    """
    Dummy dataset that follows the format expect by `PlateRecognitionPyDataset`.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = pathlib.Path(tmp_dir)
        img_dir = root / "images"
        img_dir.mkdir()

        rel_paths: list[str] = []
        for idx in range(3):
            img = (np.random.rand(32, 128, 3) * 255).astype("uint8")
            img_path = img_dir / f"img_{idx}.png"
            cv2.imwrite(str(img_path), img)
            rel_paths.append(str(img_path.relative_to(root)))

        annotations = pd.DataFrame(
            {"image_path": rel_paths, "plate_text": ["ABC123", "XYZ987", "CAR007"]}
        )
        csv_path = root / "annotations.csv"
        annotations.to_csv(csv_path, index=False)

        yield csv_path


@pytest.fixture()
def dummy_plate_config() -> str:
    return textwrap.dedent(
        """
        max_plate_slots: 9
        alphabet: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
        pad_char: '_'
        img_height: 64
        img_width: 128
        """
    )


@pytest.fixture()
def dummy_cct_model_config() -> str:
    return textwrap.dedent(
        """
        model: cct
        rescaling:
          scale: 0.00392156862745098
          offset: 0.0
        tokenizer:
          positional_emb: true
          blocks:
            - { layer: Conv2D, filters: 2, kernel_size: 3, stride: 4 }
            - { layer: MaxBlurPooling2D }
            - { layer: Conv2D, filters: 4, kernel_size: 3, stride: 4 }
            - { layer: MaxBlurPooling2D }
            - { layer: Conv2D, filters: 6, kernel_size: 3, stride: 4 }
            - { layer: MaxBlurPooling2D }
        transformer_encoder:
          layers: 1
          heads: 1
          projection_dim: 6
          units: [6, 6]
          activation: relu
          stochastic_depth: 0.0
          attention_dropout: 0.0
          mlp_dropout: 0.0
          head_mlp_dropout: 0.0
          token_reducer_heads: 1
          normalization: layer_norm
        """
    )
