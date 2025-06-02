"""
The `conftest.py` file serves as a means of providing fixtures for an entire directory. Fixtures
defined in a `conftest.py` can be used by any test in that package without needing to import them
(pytest will automatically discover them).
"""

import pathlib
import tempfile
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
