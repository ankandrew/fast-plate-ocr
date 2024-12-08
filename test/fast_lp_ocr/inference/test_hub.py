"""
Tests for ONNX hub module.
"""

from http import HTTPStatus

import pytest
import requests

from fast_plate_ocr.inference.hub import AVAILABLE_ONNX_MODELS


@pytest.mark.parametrize("model_name", AVAILABLE_ONNX_MODELS.keys())
def test_model_and_config_urls(model_name):
    """
    Test to check if the model and config URLs for AVAILABLE_ONNX_MODELS are valid.
    """
    model_url, config_url = AVAILABLE_ONNX_MODELS[model_name]

    for url in [model_url, config_url]:
        response = requests.head(url, timeout=5, allow_redirects=True)
        assert (
            response.status_code == HTTPStatus.OK
        ), f"URL {url} is not accessible, got {response.status_code}"
