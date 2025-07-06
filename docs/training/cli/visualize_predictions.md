# Visualize Predictions

Once your model is trained, you can visually inspect how it performs on **unseen images** using:

```shell
fast-plate-ocr visualize-predictions \
  --model arg_cnn_ocr.keras \
  --img-dir benchmark/imgs \
  --plate-config-file arg_cnn_ocr_config.yaml
```

---

### What You'll See

For each image, the model will:

* Predict the license plate text
* Overlay each predicted character with its **confidence score**
* Optionally color low-confidence characters (default: red if below `0.35`)

---

![Visualize Predictions](https://github.com/ankandrew/fast-plate-ocr/blob/ac3d110c58f62b79072e3a7af15720bb52a45e4e/extra/visualize_predictions.gif?raw=true)

---

## Example: Filter Uncertain Plates

```shell
fast-plate-ocr visualize-predictions \
  --model model.keras \
  --img-dir raw_eval_imgs \
  --plate-config-file config.yaml \
  --filter-conf 0.5
```

This filters out predictions unless **at least one character** has confidence < 0.5, helpful when manually reviewing
**low-quality** predictions.
