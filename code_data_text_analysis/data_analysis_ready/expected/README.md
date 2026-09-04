# Expected Result

This document defines the completion criteria without prescribing a specific implementation.

## Required project structure

```text
src/                    reusable analysis code
tests/                  automated tests
outputs/metrics.json    baseline and model metrics
outputs/predictions.csv held-out predictions
outputs/*.png           predicted-versus-actual plot
REPORT.md               method, results, commands, and limitations
```

A dependency manifest must be present. One documented command must run the analysis from the repository root.

## Acceptance criteria

- The source CSV is unchanged. The split contains each of the 32 rows exactly once, with no overlap between training and test sets.
- The fixed split produces 24 training rows and 8 test rows.
- `community_id` is absent from the feature matrix but present in saved predictions.
- Missing values are handled using training data only.
- The model and baseline use the same test rows and report MAE, RMSE, and R².
- Metrics in `REPORT.md` match `outputs/metrics.json`; the prediction table and plot use the same test rows.
- Tests and the full analysis command complete successfully.

No performance threshold relative to the baseline is imposed. Evaluation is based on correctness, reproducibility, and accurate reporting.
