# Fast & Lightweight License Plate OCR

![Intro](https://raw.githubusercontent.com/ankandrew/fast-plate-ocr/4a7dd34c9803caada0dc50a33b59487b63dd4754/extra/demo.gif)

**FastPlateOCR** is a **lightweight** and **fast** OCR framework for **license plate text recognition**. You can train
models from scratch or use the trained models for inference.

The idea is to use this after a plate object detector, since the OCR expects the cropped plates.

### Features

- **Keras 3 Backend Support**: Compatible with **TensorFlow**, **JAX**, and **PyTorch** backends 🧠
- **Augmentation Variety**: Diverse augmentations via **Albumentations** library 🖼️
- **Efficient Execution**: **Lightweight** models that are cheap to run 💰
- **ONNX Runtime Inference**: **Fast** and **optimized** inference with ONNX runtime ⚡
- **User-Friendly CLI**: Simplified **CLI** for **training** and **validating** OCR models 🛠️
- **Model HUB**: Access to a collection of pre-trained models ready for inference 🌟

### Model Zoo

We currently have the following available models:

|           Model Name           | Time b=1<br/> (ms)<sup>[1]</sup> | Throughput <br/> (plates/second)<sup>[1]</sup> |                                                      Dataset                                                      | Accuracy<sup>[2]</sup> |              Dataset              |
|:------------------------------:|:--------------------------------:|:----------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|:----------------------:|:---------------------------------:|
| `argentinian-plates-cnn-model` |              2.0964              |                      477                       | [arg_plate_dataset.zip](https://github.com/ankandrew/fast-plate-ocr/releases/download/v1.0/arg_plate_dataset.zip) |         94.05%         | Non-synthetic, plates up to 2020. |

_<sup>[1]</sup> Inference on Mac M1 chip using CPUExecutionProvider. Utilizing CoreMLExecutionProvider accelerates speed
by 5x._

_<sup>[2]</sup> Accuracy is what we refer as plate_acc. See [metrics](architecture.md#model-metrics) section._

<details>
  <summary>Reproduce results.</summary>

Calculate Inference Time:

  ```shell
  pip install fast_plate_ocr  # CPU
  # or
  pip install fast_plate_ocr[inference_gpu]  # GPU
  ```

  ```python
  from fast_plate_ocr import ONNXPlateRecognizer

  m = ONNXPlateRecognizer("argentinian-plates-cnn-model")
  m.benchmark()
  ```

Calculate Model accuracy:

  ```shell
  pip install fast-plate-ocr[train]
  curl -LO https://github.com/ankandrew/fast-plate-ocr/releases/download/v1.0/arg_cnn_ocr_config.yaml
  curl -LO https://github.com/ankandrew/fast-plate-ocr/releases/download/v1.0/arg_cnn_ocr.keras
  curl -LO https://github.com/ankandrew/fast-plate-ocr/releases/download/v1.0/arg_plate_benchmark.zip
  unzip arg_plate_benchmark.zip
  fast_plate_ocr valid \
      -m arg_cnn_ocr.keras \
      --config-file arg_cnn_ocr_config.yaml \
      --annotations benchmark/annotations.csv
  ```

</details>
