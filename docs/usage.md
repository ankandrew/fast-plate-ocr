### API

To predict from disk image:

```python
from fast_plate_ocr import ONNXPlateRecognizer

m = ONNXPlateRecognizer('argentinian-plates-cnn-model')
print(m.run('test_plate.png'))
```

<details>
  <summary>Demo</summary>

<div style="margin-top: 10px;">
<img src="https://github.com/ankandrew/fast-plate-ocr/blob/ac3d110c58f62b79072e3a7af15720bb52a45e4e/extra/inference_demo.gif?raw=true" alt="Inference Demo"/>
</div>

</details>

To run model benchmark:

```python
from fast_plate_ocr import ONNXPlateRecognizer

m = ONNXPlateRecognizer('argentinian-plates-cnn-model')
m.benchmark()
```

<details>
  <summary>Demo</summary>

<div style="margin-top: 10px;">
<img src="https://github.com/ankandrew/fast-plate-ocr/blob/ac3d110c58f62b79072e3a7af15720bb52a45e4e/extra/benchmark_demo.gif?raw=true" alt="Benchmark Demo"/>
</div>

</details>

For a full list of options see [Reference](reference.md).

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

???+ tip
    **Usually training with JAX and TensorFlow is faster.**

_Note: You will probably need to install your desired framework for training._
