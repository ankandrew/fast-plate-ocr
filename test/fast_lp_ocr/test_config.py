"""
Tests for config module
"""

from fast_plate_ocr.config import MODEL_ALPHABET, PAD_CHAR, VOCABULARY_SIZE


def test_pad_char_in_model_alphabet() -> None:
    assert PAD_CHAR in MODEL_ALPHABET


def test_vocabulary_size_is_model_vocab_length() -> None:
    assert len(MODEL_ALPHABET) == VOCABULARY_SIZE
