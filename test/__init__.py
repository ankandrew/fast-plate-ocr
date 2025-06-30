"""
Test package.
"""

from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
"""Path to project root dir."""
MODELS_CONFIG_DIR = PROJECT_ROOT_DIR / "models"
"""Path to models config dir."""
PLATE_CONFIG_DIR = PROJECT_ROOT_DIR / "config"
"""Path to plate config dir."""
MODEL_CONFIG_PATHS = [f for f in MODELS_CONFIG_DIR.iterdir() if f.suffix in (".yaml", ".yml")]
"""Default models configs present in the project."""
