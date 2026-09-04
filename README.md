# Fast & Lightweight License Plate OCR

[![Actions status](https://github.com/ankandrew/fast-plate-ocr/actions/workflows/test.yaml/badge.svg)](https://github.com/ankandrew/fast-plate-ocr/actions)
[![Actions status](https://github.com/ankandrew/fast-plate-ocr/actions/workflows/release.yaml/badge.svg)](https://github.com/ankandrew/fast-plate-ocr/actions)
[![Keras 3](https://img.shields.io/badge/Keras-3-red?logo=keras&logoColor=red&labelColor=white)](https://keras.io/keras_3/)
[![image](https://img.shields.io/pypi/v/fast-plate-ocr.svg)](https://pypi.python.org/pypi/fast-plate-ocr)
[![image](https://img.shields.io/pypi/pyversions/fast-plate-ocr.svg)](https://pypi.python.org/pypi/fast-plate-ocr)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![ONNX Model](https://img.shields.io/badge/model-ONNX-blue?logo=onnx&logoColor=white)](https://onnx.ai/)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-orange)](https://huggingface.co/spaces/ankandrew/fast-alpr)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://ankandrew.github.io/fast-plate-ocr/)
[![image](https://img.shields.io/pypi/l/fast-plate-ocr.svg)](https://pypi.python.org/pypi/fast-plate-ocr)

![Intro](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/readme_demo.gif)

---

## Introduction

**Lightweight** and **fast** OCR models for license plate text recognition. You can train models from scratch or use
the trained models for inference.

The idea is to use this after a plate object detector, since the OCR expects the cropped plates.

## Features

- **Keras 3 Backend Support**: Train seamlessly using **[TensorFlow](https://www.tensorflow.org/)**, **[JAX](https://github.com/google/jax)**, or **[PyTorch](https://pytorch.org/)** backends 🧠
- **Efficient Execution**: **Lightweight** models that are cheap to run 💰
- **ONNX Runtime Inference**: **Fast** and **optimized** inference with **[ONNX runtime](https://onnxruntime.ai/)** ⚡
- **User-Friendly CLI**: Simplified **CLI** for **training** and **validating** OCR models 🛠️
- **Region Recognition (Optional)**: Predict **region/country** of the license plate 🌍
- **Model HUB**: Access to a collection of **pre-trained models** ready for inference 🌟
- **Train**/**Fine-tune**: Easily train or **fine-tune** your own models 🔧
- **Export-Friendly**: Export easily to **CoreML**, **TFLite**, or **ONNX** formats 📦

## Available Models

Optimized, ready to use models with config files for inference or fine-tuning.

| Model Name               | Size | Arch                                                                                                                         | b=1 Avg. Latency (ms) | Plates/sec (PPS) | Model Config                                                                                                                     | Plate Config                                                                                                                     | Val Results                                                                                                           |
|--------------------------|------|------------------------------------------------------------------------------------------------------------------------------|-----------------------|------------------|----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| [`cct-s-v2-global-model`](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v2_global.onnx)  | S    | [CCT](https://ankandrew.github.io/fast-plate-ocr/latest/training/config/model_config/#compact-convolutional-transformer-cct) | **0.6758**            | **1479.61**      | [model_config.yaml](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v2_global_model_config.yaml)  | [plate_config.yaml](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v2_global_plate_config.yaml)  | [results](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v2_global_val_results.json)  |
| [`cct-xs-v2-global-model`](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_xs_v2_global.onnx) | XS   | [CCT](https://ankandrew.github.io/fast-plate-ocr/latest/training/config/model_config/#compact-convolutional-transformer-cct) | **0.4664**            | **2144.14**      | [model_config.yaml](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_xs_v2_global_model_config.yaml) | [plate_config.yaml](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_xs_v2_global_plate_config.yaml) | [results](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_xs_v2_global_val_results.json) |
| [`cct-s-v1-global-model`](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v1_global.onnx)  | S    | [CCT](https://ankandrew.github.io/fast-plate-ocr/latest/training/config/model_config/#compact-convolutional-transformer-cct) | **0.5877**            | **1701.63**      | [model_config.yaml](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v1_global_model_config.yaml)  | [plate_config.yaml](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v1_global_plate_config.yaml)  | [results](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v1_global_val_results.json)  |
| [`cct-xs-v1-global-model`](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_xs_v1_global.onnx) | XS   | [CCT](https://ankandrew.github.io/fast-plate-ocr/latest/training/config/model_config/#compact-convolutional-transformer-cct) | **0.3232**            | **3094.21**      | [model_config.yaml](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_xs_v1_global_model_config.yaml) | [plate_config.yaml](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_xs_v1_global_plate_config.yaml) | [results](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_xs_v1_global_val_results.json) |

> [!TIP]
> 🚀 Try the above models in [Hugging Spaces](https://huggingface.co/spaces/ankandrew/fast-alpr).

> [!NOTE]
> **Benchmark Setup**
>
> These results were obtained with:
>
> - **Hardware**: NVIDIA RTX 3090 GPU
> - **Execution Providers**: `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
> - **Install dependencies**: `pip install fast-plate-ocr[onnx-gpu]`

Explore the [Model Zoo](https://ankandrew.github.io/fast-plate-ocr/latest/inference/model_zoo/) to see all available models.

## Inference

For doing inference, install:

```shell
pip install fast-plate-ocr[onnx-gpu]
```

By default, **no ONNX runtime is installed**. To run inference, you **must** install at least one ONNX backend using an appropriate extra.

| Platform/Use Case  | Install Command                             | Notes                |
|--------------------|---------------------------------------------|----------------------|
| CPU (default)      | `pip install fast-plate-ocr[onnx]`          | Cross-platform       |
| NVIDIA GPU (CUDA)  | `pip install fast-plate-ocr[onnx-gpu]`      | Linux/Windows        |
| Intel (OpenVINO)   | `pip install fast-plate-ocr[onnx-openvino]` | Best on Intel CPUs   |
| Windows (DirectML) | `pip install fast-plate-ocr[onnx-directml]` | For DirectML support |
| Qualcomm (QNN)     | `pip install fast-plate-ocr[onnx-qnn]`      | Qualcomm chipsets    |


### Usage

To predict from disk image:

```python
from fast_plate_ocr import LicensePlateRecognizer

m = LicensePlateRecognizer('cct-s-v2-global-model')
print(m.run('test_plate.png'))
```

If your model includes a **region head** (and `plate_regions` is defined in the plate config), predictions
also include `region`. The `region_prob` field is populated when `return_confidence=True`:

```python
from fast_plate_ocr import LicensePlateRecognizer

m = LicensePlateRecognizer('cct-s-v2-global-model')
pred = m.run('test_plate.png', return_confidence=True)[0]
print(pred.region, pred.region_prob)
```

To run a model benchmark:

```python
from fast_plate_ocr import LicensePlateRecognizer

m = LicensePlateRecognizer('cct-s-v2-global-model')
m.benchmark()
```

For more examples and the full API, see the
[Inference Guide](https://ankandrew.github.io/fast-plate-ocr/latest/inference/running_inference/) and
[Reference](https://ankandrew.github.io/fast-plate-ocr/latest/reference/).

## Training

You can train models from scratch or fine-tune a pre-trained one using your own license plate dataset.

Install the training dependencies:

```shell
pip install fast-plate-ocr[train]
```

### Fine-tuning Tutorial

A complete tutorial notebook is available for fine-tuning a license plate OCR model on your own dataset:
[`examples/fine_tune_workflow.ipynb`](examples/tutorial_fine_tune_plate_model.ipynb). It covers the full workflow, from
preparing your dataset to training and exporting the model.

For full details on data preparation, model configs, fine-tuning, and training commands, check out the
[docs](https://ankandrew.github.io/fast-plate-ocr/1.0/training/intro/).

For **region recognition** and export-friendly activations, use the v2 models.

## Contributing

Contributions to the repo are greatly appreciated. Whether it's bug fixes, feature enhancements, or new models,
your contributions are warmly welcomed.

To start contributing or to begin development, you can follow these steps:

1. Clone repo
    ```shell
    git clone https://github.com/ankandrew/fast-plate-ocr.git
    ```
2. Install all dependencies (make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed):
    ```shell
    make install
    ```
3. To ensure your changes pass linting and tests before submitting a PR:
    ```shell
    make checks
    ```

## Citations

```bibtex
@article{hassani2021escaping,
    title   = {Escaping the Big Data Paradigm with Compact Transformers},
    author  = {Ali Hassani and Steven Walton and Nikhil Shah and Abulikemu Abuduweili and Jiachen Li and Humphrey Shi},
    year    = 2021,
    url     = {https://arxiv.org/abs/2104.05704},
    eprint  = {2104.05704},
    archiveprefix = {arXiv},
    primaryclass = {cs.CV}
}
```
