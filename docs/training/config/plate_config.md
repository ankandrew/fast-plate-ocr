# Plate Configuration File

The plate config file defines how license plate images and text should be preprocessed for OCR model training and inference.

This file is parsed using the [`PlateOCRConfig`](../../reference/train/model.md) class and validated with [Pydantic](https://docs.pydantic.dev/latest/).

---

## Config Fields

Below is a summary of the supported fields in the YAML config:

| Field               | Type                        | Description                                                          |
|---------------------|-----------------------------|----------------------------------------------------------------------|
| `max_plate_slots`   | `int`                       | Maximum number of characters the model can recognize on a plate      |
| `alphabet`          | `str`                       | The full set of characters that the model can output (no duplicates) |
| `pad_char`          | `str`                       | A single character used for padding shorter plate texts              |
| `img_height`        | `int`                       | Height of input images                                               |
| `img_width`         | `int`                       | Width of input images                                                |
| `keep_aspect_ratio` | `bool`                      | Whether to keep the original aspect ratio of the image               |
| `interpolation`     | `"cubic"`, `"linear"`, etc. | Resizing interpolation method                                        |
| `image_color_mode`  | `"grayscale"` or `"rgb"`    | Color mode for input images                                          |
| `padding_color`     | `int` or `(int, int, int)`  | Color used to pad the image if aspect ratio is preserved             |

---

## Config Example

```yaml title="plate_config.yaml"
max_plate_slots: 9
alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"  # (1)!
pad_char: "_"  # (2)!
img_height: 64
img_width: 160
keep_aspect_ratio: true
interpolation: linear  # (3)!
image_color_mode: grayscale
padding_color: 114  # (4)!
```

1. **All** the **possible character set** for the model output. **Must** include the **pad character**.
2. Padding character for plates which **length** is **smaller** than MAX_PLATE_SLOTS.
3. Matches OpenCV's interpolation names.
4. Only used when `keep_aspect_ratio` is **True**.

???+ tip
    For examples used in the default models, checkout
    [config](https://github.com/ankandrew/fast-plate-ocr/tree/master/config) directory (located in root dir of project).
