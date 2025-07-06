### Inference

For **inference**, install:

```shell
pip install fast-plate-ocr[onnx]
```

???+ warning
    By default, **no ONNX runtime is installed**.

    To run inference, you **must install** one of the ONNX extras:

    - `onnx` - for CPU inference (cross-platform)
    - `onnx-gpu` - for NVIDIA GPUs (CUDA)
    - `onnx-openvino` - for Intel CPUs / VPUs
    - `onnx-directml` - for Windows devices via DirectML
    - `onnx-qnn` - for Qualcomm chips on mobile

Dependencies for inference are kept **minimal by default**. Inference-related packages like **ONNX runtimes** are
**optional** and not installed unless **explicitly requested via extras**.

### Train

Training code uses **Keras 3**, which works with multiple backends like **TensorFlow**, **JAX**, and **PyTorch**. You
can choose the one that fits your needs, no code changes required.

To **train** or use the **CLI tool**, you'll need to install:

```shell
pip install fast-plate-ocr[train]
```

???+ tip
    You will need to **install** your desired framework for training as `fast-plate-ocr` doesn't
    enforce you to use any specific framework. See [Keras backend](training/backend.md) section.


???+ example "Using TensorFlow with GPU support"
    If you want to use **TensorFlow with GPU support** as the backend, install it with:

    ```shell
    pip install tensorflow[and-cuda]
    ```

    This ensures that the required CUDA libraries are included. For details, see [TensorFlow GPU setup](https://www.tensorflow.org/install/pip#linux).
