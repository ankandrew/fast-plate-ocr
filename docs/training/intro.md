# Train Workflow

## Requirements

To **train** and use the **CLI scripts**, you'll need to install:

```shell
pip install fast_plate_ocr[train]
```

---

## Prepare for Training

To train the model you will need:

1. A license **plate configuration** that defines how license plate images and text should be preprocessed for the **OCR** model.
   **For example**, the following config is used for plates that have at **maximum 9 chars** and that are based on the **latin-alphabet**:
    ```yaml
    # Config for Latin-alphabet plates

    # Max number of plate slots supported. This represents the number of model classification heads.
    max_plate_slots: 9
    # All the possible character set for the model output.
    alphabet: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
    # Padding character for plates which length is smaller than MAX_PLATE_SLOTS. It should still be present in the alphabet.
    pad_char: '_'
    # Image height which is fed to the model.
    img_height: 64
    # Image width which is fed to the model.
    img_width: 128
    # Keep the aspect ratio of the input image.
    keep_aspect_ratio: false
    # Interpolation method used for resizing the input image.
    interpolation: linear
    # Input image color mode. Use 'grayscale' for single-channel input or 'rgb' for 3-channel input.
    image_color_mode: rgb
    ```
   See [**Plate Config**](config/plate_config.md) section for more details.
2. A **model configuration** that defines the architecture of the OCR model. You can customize the architecture entirely
   via YAML without editing code. See the [**Model Config**](config/model_config.md) section for supported architectures and examples.
3. A **labeled dataset**, see [**Dataset**](dataset.md) section for more info.
4. Run **train script**:
    ```shell
    # You can set the backend to either TensorFlow, JAX or PyTorch
    # (just make sure it is installed)
    KERAS_BACKEND=tensorflow fast-plate-ocr train \
      --model-config-file models/cct_s_v1.yaml \
      --plate-config-file config/latin_plates.yaml \
      --annotations data/train.csv \
      --val-annotations data/val.csv \
      --epochs 150 \
      --batch-size 64 \
      --output-dir trained_models/
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
