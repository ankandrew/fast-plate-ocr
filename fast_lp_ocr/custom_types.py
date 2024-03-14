"""
Typing module
"""

import os
from typing import Literal, TypeAlias

Framework: TypeAlias = Literal["jax", "tensorflow", "torch"]
"""Supported backend frameworks for Keras."""
FilePath: TypeAlias = str | os.PathLike[str]
"""Types accepted for file path arguments."""
PilInterpolation: TypeAlias = Literal["nearest", "bilinear", "bicubic", "hamming", "box", "lanczos"]
"""Supported interpolation methods for PIL."""
