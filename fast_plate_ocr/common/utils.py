"""
Common utilities used across the package.
"""

import logging
import time
from collections.abc import Callable, Iterator
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
        logger.info("Computation time of '%s' = %.3fms", process_name, 1_000 * time_elapsed)


@contextmanager
def measure_time() -> Iterator[Callable[[], float]]:
    """
    A context manager for measuring execution time (in milliseconds) within its code block.

    usage:
        with code_timer() as timer:
            # Code snippet to be timed
        print(f"Code took: {timer()} seconds")
    """
    start_time = end_time = time.perf_counter()
    yield lambda: (end_time - start_time) * 1_000
    end_time = time.perf_counter()
