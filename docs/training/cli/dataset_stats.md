#  Dataset Statistics

The `dataset_stats` CLI command allows you to show statistics about a dataset prepared for training with `fast-plate-ocr`.

It provides insights into:

- Plate text lengths
- Image dimensions (height, width, aspect ratio)
- File extensions
- Most frequent characters

---

##  Usage

```bash
fast_plate_ocr dataset-stats \
  --annotations path/to/annotations.csv \
  --plate-config-file path/to/plate_config.yaml
```


??? note "📊 Example Output"
    ```
    ┌────────────────────────── Dataset Statistics ──────────────────────────┐
    │                       Plate Lengths                                    │
    │          ╷      ╷      ╷      ╷      ╷      ╷      ╷                   │
    │    count │ mean │  std │  min │  max │   5% │  50% │  95%              │
    │  ════════╪══════╪══════╪══════╪══════╪══════╪══════╪═════              │
    │  4397.00 │ 6.57 │ 0.97 │ 4.00 │ 9.00 │ 5.00 │ 7.00 │ 8.00              │
    │          ╵      ╵      ╵      ╵      ╵      ╵      ╵                   │
    │                            Image Height                                │
    │          ╷       ╷       ╷       ╷        ╷       ╷       ╷            │
    │    count │  mean │   std │   min │    max │    5% │   50% │    95%     │
    │  ════════╪═══════╪═══════╪═══════╪════════╪═══════╪═══════╪═══════     │
    │  4397.00 │ 76.87 │ 36.74 │ 16.00 │ 673.00 │ 32.00 │ 73.00 │ 133.00     │
    │          ╵       ╵       ╵       ╵        ╵       ╵       ╵            │
    │                              Image Width                               │
    │          ╷        ╷       ╷       ╷         ╷       ╷        ╷         │
    │    count │   mean │   std │   min │     max │    5% │    50% │    95%  │
    │  ════════╪════════╪═══════╪═══════╪═════════╪═══════╪════════╪═══════  │
    │  4397.00 │ 190.02 │ 95.52 │ 42.00 │ 1437.00 │ 83.00 │ 174.00 │ 324.00  │
    │          ╵        ╵       ╵       ╵         ╵       ╵        ╵         │
    │                        Aspect Ratio                                    │
    │          ╷      ╷      ╷      ╷      ╷      ╷      ╷                   │
    │    count │ mean │  std │  min │  max │   5% │  50% │  95%              │
    │  ════════╪══════╪══════╪══════╪══════╪══════╪══════╪═════              │
    │  4397.00 │ 2.62 │ 0.86 │ 0.57 │ 6.48 │ 1.24 │ 2.65 │ 3.90              │
    │          ╵      ╵      ╵      ╵      ╵      ╵      ╵                   │
    │   Extensions                                                           │
    │       ╷                                                                │
    │  Ext  │ Count                                                          │
    │  ═════╪══════                                                          │
    │  .jpg │  4397                                                          │
    │       ╵                                                                │
    │  Top 10 Chars                                                          │
    │       ╷                                                                │
    │  Char │ Count                                                          │
    │  ═════╪══════                                                          │
    │  1    │  2002                                                          │
    │  0    │  1825                                                          │
    │  2    │  1819                                                          │
    │  3    │  1669                                                          │
    │  7    │  1655                                                          │
    │  9    │  1631                                                          │
    │  5    │  1623                                                          │
    │  4    │  1621                                                          │
    │  6    │  1600                                                          │
    │  8    │  1553                                                          │
    │       ╵                                                                │
    └────────────────────────────────────────────────────────────────────────┘
    ```
