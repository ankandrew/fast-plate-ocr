# Metrics

The following metrics are tracked during model training and validation to help evaluate OCR model performance
at both character level and plate level granularity.

## Available Metrics

During training, you will see the following metrics:

* **plate_acc**: Compute the number of **license plates** that were **fully classified**. For a single plate, if the
  ground truth is `ABC123` and the prediction is also `ABC123`, it would score 1. However, if the prediction was
  `ABD123`, it would score 0, as **not all characters** were correctly classified.

* **cat_acc**: Calculate the accuracy of **individual characters** within the license plates that were
  **correctly classified**. For example, if the correct label is `ABC123` and the prediction is `ABC133`, it would yield
  a precision of 83.3% (5 out of 6 characters correctly classified), rather than 0% as in plate_acc, because it's not
  completely classified correctly.

* **top_3_k**: Calculate how frequently the true character is included in the **top-3 predictions**
  (the three predictions with the highest probability).

* **plate_len_acc**: Measures how often the predicted **length** of the license plate matches the ground truth.
  For example, if the target plate has 6 characters and the prediction also has 6, it scores 1 (regardless of content).

## Example Cases

| Ground Truth | Prediction | plate_acc | char_acc | plate_len_acc | Notes                       |
|--------------|------------|-----------|----------|---------------|-----------------------------|
| `ABC123`     | `ABC123`   | 100%      | 100%     | 100%          | Perfect match               |
| `ABC123`     | `ABD123`   | 0%        | 83.3%    | 100%          | 5 / 6 chars correct         |
| `XYZ9`       | `XYZ9`     | 100%      | 100%     | 100%          | Short plate, all correct    |
| `XYZ9`       | `XYZ99`    | 0%        | 75.0%    | 0%            | Length mismatch + one wrong |
| `ABC123`     | `ABX1Y3`   | 0%        | 66.7%    | 100%          | Two chars wrong             |
