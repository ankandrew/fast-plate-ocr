## Fast & Lightweight License Plate OCR

[![Actions status](https://github.com/ankandrew/fast-plate-ocr/actions/workflows/main.yaml/badge.svg)](https://github.com/ankandrew/fast-plate-ocr/actions)
[![Keras 3](https://img.shields.io/badge/Keras-3-red?logo=keras&logoColor=red&labelColor=white)](https://keras.io/keras_3/)
[![image](https://img.shields.io/pypi/v/fast-plate-ocr.svg)](https://pypi.python.org/pypi/fast-plate-ocr)
[![image](https://img.shields.io/pypi/pyversions/fast-plate-ocr.svg)](https://pypi.python.org/pypi/fast-plate-ocr)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![image](https://img.shields.io/pypi/l/fast-plate-ocr.svg)](https://pypi.python.org/pypi/fast-plate-ocr)

![Intro](https://raw.githubusercontent.com/ankandrew/fast-plate-ocr/4a7dd34c9803caada0dc50a33b59487b63dd4754/extra/demo.gif)

---

### Introduction

**Lightweight** and **fast** OCR models for license plate text recognition. You can train models from scratch or use
the trained models for inference.

The idea is to use this after a plate object detector, since the OCR expects the cropped plates.

### Features

- **Keras 3 Backend Support**: Compatible with **[TensorFlow](https://www.tensorflow.org/)**, **[JAX](https://github.com/google/jax)**, and **[PyTorch](https://pytorch.org/)** backends 🧠
- **Augmentation Variety**: Diverse **augmentations** via **[Albumentations](https://albumentations.ai/)** library 🖼️
- **Efficient Execution**: **Lightweight** models that are cheap to run 💰
- **ONNX Runtime Inference**: **Fast** and **optimized** inference with **[ONNX runtime](https://onnxruntime.ai/)** ⚡
- **User-Friendly CLI**: Simplified **CLI** for **training** and **validating** OCR models 🛠️
- **Model HUB**: Access to a collection of **pre-trained models** ready for inference 🌟

### Available Models

|                 Model Name                  | Time b=1<br/> (ms)<sup>[1]</sup> | Throughput <br/> (plates/second)<sup>[1]</sup> |                                                              Dataset                                                               | Accuracy<sup>[2]</sup> |                Dataset                |
|:-------------------------------------------:|:--------------------------------:|:----------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------:|:----------------------:|:-------------------------------------:|
|       `argentinian-plates-cnn-model`        |               2.1                |                      476                       |      [arg_plate_dataset.zip](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/arg_plate_dataset.zip)       |         94.05%         |   Non-synthetic, plates up to 2020.   |
|    `argentinian-plates-cnn-synth-model`     |               2.1                |                      476                       | [arg_plate_dataset.zip](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/arg_plate_dataset_plus_synth.zip) |         94.19%         | Plates up to 2020 + synthetic plates. |
|  🆕 `european-plates-mobile-vit-v2-model`   |               2.9                |                      344                       |                                                                 -                                                                  |  92.5%<sup>[3]</sup>   |   European plates (+40 countries).    |

_<sup>[1]</sup> Inference on Mac M1 chip using CPUExecutionProvider. Utilizing CoreMLExecutionProvider accelerates speed
by 5x in the CNN models._

_<sup>[2]</sup> Accuracy is what we refer as plate_acc. See [metrics section](#model-metrics)._

_<sup>[3]</sup> For detailed accuracy for each country see [results](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/european_mobile_vit_v2_ocr_results.json) and
the corresponding [val split](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/european_mobile_vit_v2_ocr_val.zip) used._

<details>
  <summary>Reproduce results.</summary>

* Calculate Inference Time:

  ```shell
  pip install fast_plate_ocr
  ```

  ```python
  from fast_plate_ocr import ONNXPlateRecognizer

  m = ONNXPlateRecognizer("argentinian-plates-cnn-model")
  m.benchmark()
  ```
* Calculate Model accuracy:

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

### Inference

For inference, install:

```shell
pip install fast_plate_ocr
```

#### Usage

To predict from disk image:

```python
from fast_plate_ocr import ONNXPlateRecognizer

m = ONNXPlateRecognizer('argentinian-plates-cnn-model')
print(m.run('test_plate.png'))
```

<details>
  <summary>run demo</summary>

![Run demo](https://github.com/ankandrew/fast-plate-ocr/blob/ac3d110c58f62b79072e3a7af15720bb52a45e4e/extra/inference_demo.gif?raw=true)

</details>

To run model benchmark:

```python
from fast_plate_ocr import ONNXPlateRecognizer

m = ONNXPlateRecognizer('argentinian-plates-cnn-model')
m.benchmark()
```

<details>
  <summary>benchmark demo</summary>

![Benchmark demo](https://github.com/ankandrew/fast-plate-ocr/blob/ac3d110c58f62b79072e3a7af15720bb52a45e4e/extra/benchmark_demo.gif?raw=true)

</details>

Make sure to check out the [docs](https://ankandrew.github.io/fast-plate-ocr) for more information.

### CLI

<img src="https://github.com/ankandrew/fast-plate-ocr/blob/ac3d110c58f62b79072e3a7af15720bb52a45e4e/extra/cli_screenshot.png?raw=true" alt="CLI">

To train or use the CLI tool, you'll need to install:

```shell
pip install fast_plate_ocr[train]
```

#### Train Model

To train the model you will need:

1. A configuration used for the OCR model. Depending on your use case, you might have more plate slots or different set
   of characters. Take a look at the config for Argentinian license plate as an example:
    ```yaml
    # Config example for Argentinian License Plates
    # The old license plates contain 6 slots/characters (i.e. JUH697)
    # and new 'Mercosur' contain 7 slots/characters (i.e. AB123CD)

    # Max number of plate slots supported. This represents the number of model classification heads.
    max_plate_slots: 7
    # All the possible character set for the model output.
    alphabet: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
    # Padding character for plates which length is smaller than MAX_PLATE_SLOTS. It should still be present in the alphabet.
    pad_char: '_'
    # Image height which is fed to the model.
    img_height: 70
    # Image width which is fed to the model.
    img_width: 140
    ```
2. A labeled dataset,
   see [arg_plate_dataset.zip](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/arg_plate_dataset.zip)
   for the expected data format.
3. Run train script:
    ```shell
    # You can set the backend to either TensorFlow, JAX or PyTorch
    # (just make sure it is installed)
    KERAS_BACKEND=tensorflow fast_plate_ocr train \
        --annotations path_to_the_train.csv \
        --val-annotations path_to_the_val.csv \
        --config-file config.yaml \
        --batch-size 128 \
        --epochs 750 \
        --dense \
        --early-stopping-patience 100 \
        --reduce-lr-patience 50
    ```

You will probably want to change the augmentation pipeline to apply to your dataset.

In order to do this define an Albumentations pipeline:

```python
import albumentations as A

transform_pipeline = A.Compose(
    [
        # ...
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
        A.MotionBlur(blur_limit=(3, 5), p=0.1),
        A.CoarseDropout(max_holes=10, max_height=4, max_width=4, p=0.3),
        # ... and any other augmentation ...
    ]
)

# Export to a file (this resultant YAML can be used by the train script)
A.save(transform_pipeline, "./transform_pipeline.yaml", data_format="yaml")
```

And then you can train using the custom transformation pipeline with the `--augmentation-path` option.

#### Visualize Augmentation

It's useful to visualize the augmentation pipeline before training the model. This helps us to identify
if we should apply more heavy augmentation or less, as it can hurt the model.

You might want to see the augmented image next to the original, to see how much it changed:

```shell
fast_plate_ocr visualize-augmentation \
    --img-dir benchmark/imgs \
    --columns 2 \
    --show-original \
    --augmentation-path '/transform_pipeline.yaml'
```

You will see something like:

![Augmented Images](https://github.com/ankandrew/fast-plate-ocr/blob/ac3d110c58f62b79072e3a7af15720bb52a45e4e/extra/image_augmentation.gif?raw=true)

#### Validate Model

After finishing training you can validate the model on a labeled test dataset.

Example:

```shell
fast_plate_ocr valid \
    --model arg_cnn_ocr.keras \
    --config-file arg_plate_example.yaml \
    --annotations benchmark/annotations.csv
```

#### Visualize Predictions

Once you finish training your model, you can view the model predictions on raw data with:

```shell
fast_plate_ocr visualize-predictions \
    --model arg_cnn_ocr.keras \
    --img-dir benchmark/imgs \
    --config-file arg_cnn_ocr_config.yaml
```

You will see something like:

![Visualize Predictions](https://github.com/ankandrew/fast-plate-ocr/blob/ac3d110c58f62b79072e3a7af15720bb52a45e4e/extra/visualize_predictions.gif?raw=true)

#### Export as ONNX

Exporting the Keras model to ONNX format might be beneficial to speed-up inference time.

```shell
fast_plate_ocr export-onnx \
	--model arg_cnn_ocr.keras \
	--output-path arg_cnn_ocr.onnx \
	--opset 18 \
	--config-file arg_cnn_ocr_config.yaml
```

### Keras Backend

To train the model, you can install the ML Framework you like the most. **Keras 3** has
support for **TensorFlow**, **JAX** and **PyTorch** backends.

To change the Keras backend you can either:

1. Export `KERAS_BACKEND` environment variable, i.e. to use JAX for training:
    ```shell
    KERAS_BACKEND=jax fast_plate_ocr train --config-file ...
    ```
2. Edit your local config file at `~/.keras/keras.json`.

_Note: You will probably need to install your desired framework for training._

### Model Architecture

The current model architecture is quite simple but effective.
See [cnn_ocr_model](https://github.com/ankandrew/cnn-ocr-lp/blob/e59b738bad86d269c82101dfe7a3bef49b3a77c7/fast_plate_ocr/train/model/models.py#L23-L23)
for implementation details.

The model output consists of several heads. Each head represents the prediction of a character of the
plate. If the plate consists of 7 characters at most (`max_plate_slots=7`), then the model would have 7 heads.

Example of Argentinian plates:

![Model head](https://raw.githubusercontent.com/ankandrew/fast-plate-ocr/4a7dd34c9803caada0dc50a33b59487b63dd4754/extra/FCN.png)

Each head will output a probability distribution over the `vocabulary` specified during training. So the output
prediction for a single plate will be of shape `(max_plate_slots, vocabulary_size)`.

### Model Metrics

During training, you will see the following metrics

* **plate_acc**: Compute the number of **license plates** that were **fully classified**. For a single plate, if the
  ground truth is `ABC123` and the prediction is also `ABC123`, it would score 1. However, if the prediction was
  `ABD123`, it would score 0, as **not all characters** were correctly classified.

* **cat_acc**: Calculate the accuracy of **individual characters** within the license plates that were
  **correctly classified**. For example, if the correct label is `ABC123` and the prediction is `ABC133`, it would yield
  a precision of 83.3% (5 out of 6 characters correctly classified), rather than 0% as in plate_acc, because it's not
  completely classified correctly.

* **top_3_k**: Calculate how frequently the true character is included in the **top-3 predictions**
  (the three predictions with the highest probability).

### Contributing

Contributions to the repo are greatly appreciated. Whether it's bug fixes, feature enhancements, or new models,
your contributions are warmly welcomed.

To start contributing or to begin development, you can follow these steps:

1. Clone repo
    ```shell
    git clone https://github.com/ankandrew/fast-plate-ocr.git
    ```
2. Install all dependencies using [Poetry](https://python-poetry.org/docs/#installation):
    ```shell
    poetry install --all-extras
    ```
3. To ensure your changes pass linting and tests before submitting a PR:
    ```shell
    make checks
    ```

If you want to train a model and share it, we'll add it to the HUB 🚀

### TODO

- [ ] Expand model zoo.
- [ ] Use synthetic image plates.
- [ ] Finish and push TorchServe files.
- [ ] Use Google docstring style
