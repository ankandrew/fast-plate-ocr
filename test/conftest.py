"""
The `conftest.py` file serves as a means of providing fixtures for an entire directory. Fixtures
defined in a `conftest.py` can be used by any test in that package without needing to import them
(pytest will automatically discover them).
"""

import pytest


@pytest.fixture(scope="function")
def temp_directory(tmpdir):
    """
    Example fixture to create a temporary directory for testing.
    """
    temp_dir = tmpdir.mkdir("temp")
    yield temp_dir
    temp_dir.remove()
