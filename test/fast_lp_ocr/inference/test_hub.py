"""
Tests for ONNX hub module.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from http import HTTPStatus

import requests

from fast_plate_ocr.inference.hub import AVAILABLE_ONNX_MODELS


def _check_url(url: str) -> tuple[str, int | str]:
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        return url, response.status_code
    except requests.RequestException as e:
        return url, str(e)


def test_model_and_config_urls():
    urls = [url for model in AVAILABLE_ONNX_MODELS.values() for url in model]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_check_url, url): url for url in urls}

        for future in as_completed(futures):
            url, result = future.result()
            assert result == HTTPStatus.OK, f"URL {url} is not accessible, got {result}"
