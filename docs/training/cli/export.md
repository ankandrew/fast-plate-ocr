# Exporting a Trained OCR Model

The `export` command converts a trained `.keras` model into alternative formats like **ONNX**, **TFLite**, or **CoreML**, enabling
deployment to different platforms and devices.

Inside `fast-plate-ocr` ecosystem, only ONNX inference is supported. But you are free to export trained models and easily
use in any other of the exported formats!

---

## Export to ONNX

### Basic Usage

```shell
fast-plate-ocr export \
  --model trained_models/best.keras \
  --plate-config-file config/latin_plates.yaml \
  --format onnx
```

### Channels first AND input dtype float32

By default, the ONNX models are exported with channels last and input dtype of `uint8`. There might be cases that
you want channels first (`BxCxHxW`) and input dtype of `float32`. This is useful for
[RKNN](https://github.com/rockchip-linux/rknn-toolkit2) See this issue for context:
[fast-plate-ocr/issues/46](https://github.com/ankandrew/fast-plate-ocr/issues/46).

```shell
fast-plate-ocr export \
  --model trained_models/best.keras \
  --plate-config-file config/latin_plates.yaml \
  --format onnx \
  --onnx-data-format channels_first \
  --onnx-input-dtype float32
```

??? info "Model shape compatibility"
    Some formats (like TFLite) only support **fixed batch sizes**, whereas ONNX allows **dynamic batching**.
    The export script handles these differences automatically.

---

## Export to TFLite

TensorFlow Lite is ideal for deploying models to mobile and edge devices.

```shell
fast-plate-ocr export \
  --model trained_models/best.keras \
  --plate-config-file config/latin_plates.yaml \
  --format tflite
```

???+ note "TFLite batch dim"
    **TFLite** does **not** support **dynamic batch sizes**, so input is fixed to `batch_size=1`.


---

## Export to CoreML

```shell
fast-plate-ocr export \
  --model trained_models/best.keras \
  --plate-config-file config/latin_plates.yaml \
  --format coreml
```

This will produce a `.mlpackage` file, compatible with CoreML and Xcode deployments.
