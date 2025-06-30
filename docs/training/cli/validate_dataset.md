# Dataset Validation

Before training your OCR model, it's strongly recommended to **validate your dataset** using the `validate-dataset`
CLI command. This ensures image integrity, label consistency, and format compatibility with your plate config.

---

## What It Checks

The validator performs the following:

- **Image existence**: Verifies that all image paths exist.
- **Image readability**: Confirms that images are decodable and not corrupted.
- **Minimum resolution**: Flags images smaller than a safe size (i.e., 2x2).
- **Resizing feasibility**: Ensures images won't be resized below 1 pixel.
- **Text length**: Verifies plate text length are less or equal than `max_plate_slots`.
- **Alphabet coverage**: Ensures all characters are inside the allowed `alphabet`.
- **Duplicate entries**: Warns about repeated image paths.
- **Unused characters**: Identifies characters in your `alphabet` that are not used at all.

---

## Basic Usage

```shell
fast-plate-ocr validate-dataset \
  --annotations-file my-dataset/train.csv \
  --plate-config-file config/latin_plates.yaml
```

---

## Fix and Export Cleaned File

To automatically export a cleaned version of your dataset:

```shell
fast-plate-ocr validate-dataset \
  --annotations-file my-dataset/train.csv \
  --plate-config-file config/latin_plates.yaml \
  --export-fixed train_clean.csv
```

This creates `train_clean.csv` with only valid entries, skipping corrupted rows and malformed labels.

---

## Allow Warnings but Exit on Errors

By default, the validator exits with code `1` if any **error** occurs. Use `--warn-only` to suppress the exit:

```shell
fast-plate-ocr validate-dataset \
  --annotations-file my-dataset/train.csv \
  --plate-config-file config/latin_plates.yaml \
  --warn-only
```

---

## Control Minimum Resolution

Adjust what you consider "too small" for images:

```shell
fast-plate-ocr validate-dataset \
  --annotations-file my-dataset/train.csv \
  --plate-config-file config/latin_plates.yaml \
  --min-height 16 \
  --min-width 32
```

---

## Output Example

After validation, a **summary table** is printed to the console using rich formatting:

```
 Validation Summary
┌──────────┬───────┐
│ Category │ Count │
├──────────┼───────┤
│ Errors   │   1   │
│ Warnings │   1   │
└──────────┴───────┘

                      Errors
┌──────┬──────────────────────────────────────────────┐
│ Line │ Message                                      │
├──────┼──────────────────────────────────────────────┤
│ 4554 │ Resize would give 0x0 (0x128)                │
│      │ from ./img/img_00001.jpg                     │
└──────┴──────────────────────────────────────────────┘

                     Warnings
┌──────┬──────────────────────────────────────────────┐
│ Line │ Message                                      │
├──────┼──────────────────────────────────────────────┤
│ 4554 │ Tiny image (1x1437 < 2x2):                   │
│      │ ./img/img_00001.jpg                          │
└──────┴──────────────────────────────────────────────┘
```

If no errors are found, you're safe to proceed with training.
