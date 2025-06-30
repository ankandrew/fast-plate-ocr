## Model Zoo

Optimized, ready to use models with config files for inference or fine-tuning.

| Model Name               | Size | Arch                                                                            | b=1 Avg. Latency (ms) | Plates/sec (PPS) | Model Config                                                                                                        | Plate Config                                                                                                        |
|--------------------------|------|---------------------------------------------------------------------------------|-----------------------|------------------|---------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `cct-s-v1-global-model`  | S    | [CCT](../training/config/model_config.md#compact-convolutional-transformer-cct) | **0.5877**            | **1701.63**      | [link](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v1_global_model_config.yaml)  | [link](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v1_global_plate_config.yaml)  |
| `cct-xs-v1-global-model` | XS   | [CCT](../training/config/model_config.md#compact-convolutional-transformer-cct) | **0.3232**            | **3094.21**      | [link](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_xs_v1_global_model_config.yaml) | [link](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_xs_v1_global_plate_config.yaml) |

??? info "Benchmarking Setup"
    These results were obtained with:

    - **Hardware**: NVIDIA RTX 3090 GPU
    - **Execution Providers**: `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
    - **Install dependencies**:
      ```bash
      pip install fast-plate-ocr[onnx-gpu]
      ```

---

## Legacy Models

These are pre-trained models from earlier iterations of `fast-plate-ocr`, primarily kept for inference purposes.

|              Model Name               | Time b=1<br/> (ms)<sup>[1]</sup> | Throughput <br/> (plates/second)<sup>[1]</sup> | Accuracy<sup>[2]</sup> |                                                                                           Dataset                                                                                            |
|:-------------------------------------:|:--------------------------------:|:----------------------------------------------:|:----------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|    `argentinian-plates-cnn-model`     |               2.1                |                      476                       |         94.05%         |              Non-synthetic, plates up to 2020. Dataset [arg_plate_dataset.zip](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/arg_plate_dataset.zip).              |
| `argentinian-plates-cnn-synth-model`  |               2.1                |                      476                       |         94.19%         | Plates up to 2020 + synthetic plates. Dataset [arg_plate_dataset_plus_synth.zip](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/arg_plate_dataset_plus_synth.zip). |
| `european-plates-mobile-vit-v2-model` |               2.9                |                      344                       |  92.5%<sup>[3]</sup>   |                                                                European plates (from +40 countries, trained on 40k+ plates).                                                                 |
|  `global-plates-mobile-vit-v2-model`  |               2.9                |                      344                       |  93.3%<sup>[4]</sup>   |                                                                Worldwide plates (from +65 countries, trained on 85k+ plates).                                                                |

??? warning "Legacy Notice"
    These are older models maintained **for compatibility and inference only**.
    They are **not recommended for fine-tuning or continued development**.
    For best results, use the newer models from the [Model Zoo](#model-zoo).

??? note "Inference & Evaluation Info"
    _<sup>[1]</sup> Inference on Mac M1 chip using CPUExecutionProvider. Utilizing CoreMLExecutionProvider accelerates speed
    by 5x in the CNN models._

    _<sup>[2]</sup> Accuracy is what we refer as plate_acc. See [metrics section](../training/metrics.md)._

    _<sup>[3]</sup> For detailed accuracy for each country see [results](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/european_mobile_vit_v2_ocr_results.json) and
    the corresponding [val split](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/european_mobile_vit_v2_ocr_val.zip) used._

    _<sup>[4]</sup> For detailed accuracy for each country see [results](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/global_mobile_vit_v2_ocr_results.json)._

??? example "Reproduce results"
    Calculate Inference Time:

      ```shell
      pip install fast_plate_ocr[onnx-gpu]
      ```

      ```python
      from fast_plate_ocr import LicensePlateRecognizer

      m = LicensePlateRecognizer("argentinian-plates-cnn-model")
      m.benchmark()
      ```

    Calculate Model accuracy:

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
