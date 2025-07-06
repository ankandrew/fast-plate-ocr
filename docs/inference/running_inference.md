### Inference Guide

The `fast-plate-ocr` library performs **high-performance** license plate recognition using **ONNX Runtime** for **inference**.

To run inference use the [`LicensePlateRecognizer`](../reference/inference/inference_class.md) class, which supports a wide
range of input types:

- File paths (str or Path)
- NumPy arrays representing single images (grayscale or RGB)
- Lists of paths or NumPy arrays
- Pre-batched NumPy arrays (4D shape: (N, H, W, C))

The model automatically handles resizing, padding, and format conversion according to its configuration. Predictions
can optionally include character-level confidence scores.



### Predict a single image

```python
from fast_plate_ocr import LicensePlateRecognizer

plate_recognizer = LicensePlateRecognizer("cct-xs-v1-global-model")
print(plate_recognizer.run("test_plate.png"))
```

<details>
  <summary>Demo</summary>

<div style="margin-top: 10px;">
<img src="https://github.com/ankandrew/fast-plate-ocr/blob/ac3d110c58f62b79072e3a7af15720bb52a45e4e/extra/inference_demo.gif?raw=true" alt="Inference Demo"/>
</div>

</details>

### Predict a batch in memory

```python
import cv2
from fast_plate_ocr import LicensePlateRecognizer

plate_recognizer = LicensePlateRecognizer("cct-xs-v1-global-model")
imgs = [cv2.imread(p) for p in ["plate1.jpg", "plate2.jpg"]]
res = plate_recognizer.run(imgs)
```

### Return confidence scores

```python
from fast_plate_ocr import LicensePlateRecognizer

plate_recognizer = LicensePlateRecognizer("cct-xs-v1-global-model")
plates, conf = plate_recognizer.run("test_plate.png", return_confidence=True)
```

### Benchmark the model

```python
from fast_plate_ocr import LicensePlateRecognizer

m = LicensePlateRecognizer("cct-xs-v1-global-model")
m.benchmark()
```

<details>
  <summary>Demo</summary>

<div style="margin-top: 10px;">
<img src="https://github.com/ankandrew/fast-plate-ocr/blob/ac3d110c58f62b79072e3a7af15720bb52a45e4e/extra/benchmark_demo.gif?raw=true" alt="Benchmark Demo"/>
</div>

</details>

For a full list of options see [Reference](../reference/inference/inference_class.md).
