### ConvNet (CNN) model

The current model architecture is quite simple but effective. It just consists of a few CNN layers with several output
heads.
See [cnn_ocr_model](https://github.com/ankandrew/cnn-ocr-lp/blob/e59b738bad86d269c82101dfe7a3bef49b3a77c7/fast_plate_ocr/train/model/models.py#L23-L23)
for implementation details.

The model output consists of several heads. Each head represents the prediction of a character of the
plate. If the plate consists of 7 characters at most (`max_plate_slots=7`), then the model would have 7 heads.

Example of Argentinian plates:

![Model head](https://raw.githubusercontent.com/ankandrew/fast-plate-ocr/4a7dd34c9803caada0dc50a33b59487b63dd4754/extra/FCN.png)

Each head will output a probability distribution over the `vocabulary` specified during training. So the output
prediction for a single plate will be of shape `(max_plate_slots, vocabulary_size)`.

### Model Metrics

During training, you will see the following metrics

* **plate_acc**: Compute the number of **license plates** that were **fully classified**. For a single plate, if the
  ground truth is `ABC123` and the prediction is also `ABC123`, it would score 1. However, if the prediction was
  `ABD123`, it would score 0, as **not all characters** were correctly classified.

* **cat_acc**: Calculate the accuracy of **individual characters** within the license plates that were
  **correctly classified**. For example, if the correct label is `ABC123` and the prediction is `ABC133`, it would yield
  a precision of 83.3% (5 out of 6 characters correctly classified), rather than 0% as in plate_acc, because it's not
  completely classified correctly.

* **top_3_k**: Calculate how frequently the true character is included in the **top-3 predictions**
  (the three predictions with the highest probability).
