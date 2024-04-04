"""
Common utilities used across the package.
"""

import logging
import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def log_time_taken(process_name: str) -> Iterator[None]:
    """A concise context manager to time code snippets and log the result."""
    time_start: float = time.perf_counter()
    try:
        yield
    finally:
        time_end: float = time.perf_counter()
        time_elapsed: float = time_end - time_start
        logging.info("Computation time of '%s' = %.3fms", process_name, 1000 * time_elapsed)
