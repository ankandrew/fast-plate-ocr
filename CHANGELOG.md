# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-06-07

### Added

- Inference now works smoothly with different onnxruntime variants like `onnxruntime-gpu`, `onnxruntime-openvino`, etc.
- Support for building and customizing CCT (Compact Convolutional Transformer) models from YAML configs.
- New model building logic, allows users to build custom-based architectures while validating it with Pydantic.
- New metric `val_plate_len_acc`.
- Added support for categorical focal loss.
- Added more test coverage (configs, train scripts, etc.).
- New `validate_dataset.py` script to help check datasets before training.
- Export script now officially supports more formats like TFLite and CoreML.
- New plate config support: `keep_aspect_ratio`, `interpolation`, `image_color_mode` and `padding_color`.
- New default augmentation for RGB image mode.
- New default models, trained with much more data.
- Added examples and more docs.

### Changed

- Visualize augmentation script now respects config-based preprocessing.
- Improved plate config validation.
- `ONNXPlateRecognizer` is now called `LicensePlateRecognizer`.

## [0.3.0] - 2024-12-08

### Added

- New Global model using MobileViTV2 trained with data from +65 countries, with 85k+ plates 🚀 .

[0.2.0]: https://github.com/ankandrew/fast-plate-ocr/compare/v0.2.0...v0.3.0

## [0.2.0] - 2024-10-14

### Added

- New European model using MobileViTV2 - trained on +40 countries 🚀 .
- Added more logging to train script.

[0.2.0]: https://github.com/ankandrew/fast-plate-ocr/compare/v0.1.6...v0.2.0

## [0.1.6] - 2024-05-09

### Added

- Add new Argentinian model trained with more (synthetic) data.
- Add option to visualize only predictions which have low char prob.
- Add onnxsim for simplifying ONNX model when exporting.

[0.1.6]: https://github.com/ankandrew/fast-plate-ocr/compare/v0.1.5...v0.1.6
