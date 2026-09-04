# Life Expectancy Regression Report

## Question

Can a Ridge regression model predict community-level `life_expectancy_years` in the supplied synthetic dataset, and how does its held-out performance compare with a training-set mean baseline?

## Method

The analysis uses the supplied 32-row dataset only. It excludes `community_id` from the feature matrix and uses a fixed 75/25 train-test split with random seed 42 (24 training rows and 8 test rows). A scikit-learn pipeline median-imputes missing feature values, standardizes the numeric predictors, and fits one Ridge regression model with its default `alpha=1.0`. Because the complete pipeline is fitted only on the training split, imputation and scaling do not learn from test data. The baseline predicts the training target mean for every held-out row. Both methods are evaluated on the same test rows with MAE, RMSE, and R².

Environment setup: `python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`

Test command: `.venv/bin/python -m pytest`

Analysis command: `.venv/bin/python -m src.analysis`

## Results

The saved `outputs/metrics.json` reports:

| Predictor | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Ridge model | 0.18240343351585686 | 0.2461432280340689 | 0.9986914365290058 |
| Training-mean baseline | 6.274999999999997 | 6.967783004657934 | -0.04859611231101435 |

On these eight held-out communities, the Ridge model has lower MAE and RMSE and higher R² than the baseline. The prediction table and plot use these same held-out rows.

## Limitations

The dataset contains only 32 fictional communities, so the eight-row test result is sensitive to the fixed split and does not establish generalization to real communities. This exercise uses a single untuned Ridge model, and it makes neither causal nor individual-level or medical claims.
