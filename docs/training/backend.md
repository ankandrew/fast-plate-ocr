## Installation

Make sure to install the backend framework you want to use:

- [**TensorFlow**](https://www.tensorflow.org/install).
- [**JAX**](https://docs.jax.dev/en/latest/installation.html).
- [**PyTorch**](https://pytorch.org/get-started/locally/).

Keras will automatically use the one selected via `KERAS_BACKEND`.

???+ example "Example Using TensorFlow with GPU support"
    If you want to use **TensorFlow with GPU support** as the backend, install it with:

    ```shell
    pip install tensorflow[and-cuda]
    ```

    This ensures that the required CUDA libraries are included. For details, see [TensorFlow GPU setup](https://www.tensorflow.org/install/pip#linux).

---

## Keras Backend

To train the model, you can install the ML Framework you like the most. **Keras 3** has
support for **TensorFlow**, **JAX** and **PyTorch** backends.

To change the Keras backend you can either:

1. Export `KERAS_BACKEND` environment variable, i.e. to use JAX for training:
    ```shell
    KERAS_BACKEND=tensorflow fast_plate_ocr train --config-file ...
    ```
2. Edit your local config file at `~/.keras/keras.json`.

???+ tip
    **Usually training with JAX and TensorFlow is faster.**
