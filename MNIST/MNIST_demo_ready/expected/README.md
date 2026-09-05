# Result guide

A completed run should create:

```text
REPORT.md
outputs/
├── metrics.json
├── predictions.csv
└── figures/
    ├── confusion_matrix.png
    └── sample_predictions.png
```

`metrics.json` records the model, seed, training and test row counts, accuracy, and macro F1. `predictions.csv` contains the true and predicted labels for all 10,000 test rows.

The confusion matrix uses digit order `0–9`. The sample figure labels each image with its true and predicted digit.

Expect test accuracy near 90% from this baseline. If it is substantially lower, check CSV parsing, label separation, and pixel scaling before changing the model. This is a diagnostic reference, not a value to copy into the report.

`REPORT.md` must match the saved metrics and include the exact commands used to generate the outputs and run the tests.
