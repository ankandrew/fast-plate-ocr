# Model Configuration

The `model_config.yaml` defines the **architecture** of the **OCR** model used in `fast-plate-ocr`. This configuration
allows you to **customize key components** of the model, i.e., convolutional tokenizers, patching strategies, attention
settings, etc. (all **without modifying code**).

**All model configurations are validated** using **Pydantic**, ensuring that every field, layer, and parameter is checked
when building the model from the config.

Currently, the supported architectures are:

- [**CCT**](#compact-convolutional-transformer-cct) (Compact Convolutional Transformer [^1])

## Supported Architectures

---

### Compact Convolutional Transformer (CCT)

Inspired by the [**CCT** architecture](https://arxiv.org/abs/2104.05704), this model structure:

1. Uses a **convolutional tokenizer** to extract patch representations from the input image.
2. Processes the resulting sequence with a **Transformer Encoder**.

#### Config Example

```yaml title="cct_model_config.yaml" linenums="1"
model: cct

rescaling:  # (1)!
  scale: 0.00392156862745098
  offset: 0.0

tokenizer:
  blocks:  # (2)!
    - { layer: Conv2D, filters: 32, kernel_size: 3, strides: 1 }  # (3)!
    - { layer: MaxBlurPooling2D, pool_size: 2, filter_size: 3 }
    - { layer: Conv2D, filters: 48, kernel_size: 3, strides: 1 }
    - { layer: Conv2D, filters: 64, kernel_size: 3, strides: 1 }
    - { layer: MaxBlurPooling2D, pool_size: 2, filter_size: 3 }
    - { layer: Conv2D, filters: 80, kernel_size: 3, strides: 1 }
    - { layer: Conv2D, filters: 96, kernel_size: 3, strides: 1 }

  positional_emb: true
  patch_size: 2
  patch_mlp:
    layer: MLP
    hidden_units: [64]
    activation: gelu
    dropout_rate: 0.05

transformer_encoder:
  layers: 4
  heads: 1
  projection_dim: 64
  units: [64, 64]
  activation: gelu
  stochastic_depth: 0.05
  attention_dropout: 0.05
  mlp_dropout: 0.1
  head_mlp_dropout: 0.05
  token_reducer_heads: 4
  normalization: dyt  # (4)!
```

1. This scales values between range `[0, 1]`
2. **Supports** a **wide** variety of **layer** types, such as `Conv2D`, `MaxPooling2D`, `DepthwiseConv2D`, `SqueezeExcite`, etc. See [**Model Config**](../../reference/train/model_config.md) for all available option.
3. Each layer **supports** the **full set** of corresponding **Keras parameters**. For example, `Conv2D` accepts `filters`, `kernel_size`, `strides`, etc.
4. See `Transformers without Normalization` [^2]

???+ note "Note on plate/model configs"
    The [plate config](plate_config.md) is used throughout both **inference** and **training** scripts.
    In contrast, the **model config** (shown above) is **only used for training**, as it defines the architecture to be built.

#### Building Custom Tokenizers with Any Keras Layer

You can define your own tokenizer stacks by composing **any supported layer** like `Conv2D`, `DepthwiseConv2D`, `SqueezeExcite`, and many more directly in YAML, without writing any code.

Each layer accepts all its typical Keras parameters, and the model schema is **validated with Pydantic**, so typos or misconfigured fields are caught immediately.

Here's an example with a more diverse set of layers:

```yaml title="custom_model_config.yaml"
tokenizer:
  blocks:
    - { layer: Conv2D, filters: 64, kernel_size: 3, activation: relu }
    - { layer: SqueezeExcite, ratio: 0.5 }
    - { layer: DepthwiseConv2D, kernel_size: 3, strides: 1 }
    - { layer: BatchNormalization }
    - { layer: MaxBlurPooling2D, pool_size: 2, filter_size: 3 }
    - { layer: Conv2D, filters: 128, kernel_size: 3 }
    - { layer: CoordConv2D, filters: 96, kernel_size: 3, with_r: true }
```

???+ tip
    Each `layer:` value corresponds to a class in the [**Model Schema**](../../reference/train/model.md), check it out
    to see all the supported layers and options!

[^1]: [Escaping the Big Data Paradigm with Compact Transformers](https://arxiv.org/abs/2104.05704).
[^2]: [Transformers without Normalization](https://arxiv.org/abs/2503.10622v1).
