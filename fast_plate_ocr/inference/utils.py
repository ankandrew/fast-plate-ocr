"""
Utilities used around the inference package.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


@contextmanager
def safe_write(
    file: str | os.PathLike[str],
    mode: str = "wb",
    encoding: str | None = None,
    **kwargs: Any,
) -> Iterator[IO]:
    """
    Context manager for safe file writing.

    Opens the specified file for writing and yields a file object.
    If an exception occurs during writing, the file is removed before raising the exception.
    """
    try:
        with open(file, mode, encoding=encoding, **kwargs) as f:
            yield f
    except Exception as e:
        Path(file).unlink(missing_ok=True)
        raise e
