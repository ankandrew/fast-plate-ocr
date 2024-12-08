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

|                Model Name                | Time b=1<br/> (ms)<sup>[1]</sup> | Throughput <br/> (plates/second)<sup>[1]</sup> | Accuracy<sup>[2]</sup> |                                                                                           Dataset                                                                                            |
|:----------------------------------------:|:--------------------------------:|:----------------------------------------------:|:----------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      `argentinian-plates-cnn-model`      |               2.1                |                      476                       |         94.05%         |              Non-synthetic, plates up to 2020. Dataset [arg_plate_dataset.zip](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/arg_plate_dataset.zip).              |
|   `argentinian-plates-cnn-synth-model`   |               2.1                |                      476                       |         94.19%         | Plates up to 2020 + synthetic plates. Dataset [arg_plate_dataset_plus_synth.zip](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/arg_plate_dataset_plus_synth.zip). |
|  `european-plates-mobile-vit-v2-model`   |               2.9                |                      344                       |  92.5%<sup>[3]</sup>   |                                                                European plates (from +40 countries, trained on 40k+ plates).                                                                 |
| 🆕🔥 `global-plates-mobile-vit-v2-model` |               2.9                |                      344                       |  93.3%<sup>[4]</sup>   |                                                                Worldwide plates (from +65 countries, trained on 85k+ plates).                                                                |

_<sup>[1]</sup> Inference on Mac M1 chip using CPUExecutionProvider. Utilizing CoreMLExecutionProvider accelerates speed
by 5x in the CNN models._

_<sup>[2]</sup> Accuracy is what we refer as plate_acc. See [metrics section](#model-metrics)._

_<sup>[3]</sup> For detailed accuracy for each country see [results](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/european_mobile_vit_v2_ocr_results.json) and
the corresponding [val split](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/european_mobile_vit_v2_ocr_val.zip) used._

_<sup>[4]</sup> For detailed accuracy for each country see [results](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/global_mobile_vit_v2_ocr_results.json)._

<details>
  <summary>Reproduce results.</summary>

Calculate Inference Time:

  ```shell
  pip install fast_plate_ocr
  ```

  ```python
  from fast_plate_ocr import ONNXPlateRecognizer

  m = ONNXPlateRecognizer("argentinian-plates-cnn-model")
  m.benchmark()
  ```

Calculate Model accuracy:

  ```shell
  pip install fast-plate-ocr[train]
  curl -LO https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/arg_cnn_ocr_config.yaml
  curl -LO https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/arg_cnn_ocr.keras
  curl -LO https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/arg_plate_benchmark.zip
  unzip arg_plate_benchmark.zip
  fast_plate_ocr valid \
      -m arg_cnn_ocr.keras \
      --config-file arg_cnn_ocr_config.yaml \
      --annotations benchmark/annotations.csv
  ```

</details>
