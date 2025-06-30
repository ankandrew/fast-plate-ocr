"""
Common utilities used across the package.
"""

import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


@contextmanager
def log_time_taken(process_name: str) -> Iterator[None]:
    """
    A concise context manager to time code snippets and log the result.

    Usage:
        ```python
        with log_time_taken("process_name"):
            # Code snippet to be timed
        ```

    Args:
        process_name: Name of the process being timed.
    """
    time_start: float = time.perf_counter()
    try:
        yield
    finally:
        time_end: float = time.perf_counter()
        time_elapsed: float = time_end - time_start
        logger = logging.getLogger(__name__)
        logger.info("Computation time of '%s' = %.3fms", process_name, 1_000 * time_elapsed)


@contextmanager
def measure_time() -> Iterator[Callable[[], float]]:
    """
    A context manager for measuring execution time (in milliseconds) within its code block.

    Usage:
        ```python
        with measure_time() as timer:
            # Code snippet to be timed
        print(f"Code took: {timer()} ms")
        ```

    Returns:
        A function that returns the elapsed time in milliseconds.
    """
    start_time = end_time = time.perf_counter()
    yield lambda: (end_time - start_time) * 1_000
    end_time = time.perf_counter()


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

    Args:
        file: Path to the file to write.
        mode: File open mode (e.g. ``"wb"``, ``"w"``, etc.). Defaults to ``"wb"``.
        encoding: Encoding to use (for text modes). Ignored in binary mode.
        **kwargs: Additional arguments passed to ``open()``.

    Returns:
        A writable file object.
    """
    try:
        with open(file, mode, encoding=encoding, **kwargs) as f:
            yield f
    except Exception as e:
        Path(file).unlink(missing_ok=True)
        raise e
