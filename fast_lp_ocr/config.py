"""
Config values used throughout the code.
"""

MAX_PLATE_SLOTS: int = 7
"""Max number of plate slots supported. This represents the number of model classification heads."""
VOCABULARY_SIZE: int = 37
"""Vocabulary size, which influences the output size of each head."""
MODEL_ALPHABET: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
"""All the possible character set for the model output."""
PAD_CHAR: str = "_"
"""Padding character for plates which length is smaller than MAX_PLATE_SLOTS."""
DEFAULT_IMG_HEIGHT: int = 70
"""Default image height which is fed to the model."""
DEFAULT_IMG_WIDTH: int = 140
"""Default image width which is fed to the model."""
