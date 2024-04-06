"""
Common utilities used across the package.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def log_time_taken(process_name: str) -> Iterator[None]:
    """
    A concise context manager to time code snippets and log the result.

    Usage:
    with log_time_taken("process_name"):
        # Code snippet to be timed

    :param process_name: Name of the process being timed.
    """
    time_start: float = time.perf_counter()
    try:
        yield
    finally:
        time_end: float = time.perf_counter()
        time_elapsed: float = time_end - time_start
        logger = logging.getLogger(__name__)
        logger.info("Computation time of '%s' = %.3fms", process_name, 1000 * time_elapsed)
