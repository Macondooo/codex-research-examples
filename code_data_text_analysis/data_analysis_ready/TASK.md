# Project Task

## Outcome

Build a reproducible Python project that predicts `life_expectancy_years` from `data/life_expectancy.csv`. Compare a single regression model with a training-set mean baseline.

## Experiment

- Use a fixed 75/25 train-test split with random seed `42`.
- Exclude `community_id` from model features.
- Handle missing feature values without learning from the test set.
- Evaluate the model and baseline on the same test rows using MAE, RMSE, and R².
- Use exactly one regression model; model selection and hyperparameter tuning are out of scope.

## Deliverables

- A dependency manifest and a documented command for running the analysis;
- Reusable code under `src/` and automated tests under `tests/`;
- `outputs/metrics.json`, `outputs/predictions.csv`, and one predicted-versus-actual plot;
- `REPORT.md` with the question, method, results, limitations, and exact run and test commands.

The prediction table must retain `community_id`, actual values, model predictions, and baseline predictions. Tests must cover data loading, feature exclusion, missing-value handling, and the expected output schema.

## Boundaries

Use only the supplied dataset and leave it unchanged. Do not make medical, individual-level, or causal claims. Model performance relative to the baseline is not an acceptance criterion; all results must be reported accurately.
