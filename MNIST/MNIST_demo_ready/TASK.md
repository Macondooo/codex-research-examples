# MNIST classification

## Aim

Build a command-line experiment that trains and evaluates a handwritten-digit classifier on the supplied MNIST CSV files. Preserve the supplied train/test split.

## Method

- Read each row as one label followed by 784 pixel values.
- Scale pixel values from `0–255` to `0–1`.
- Train one scikit-learn `SGDClassifier` with `loss="log_loss"` and `random_state=42`.
- Evaluate the fitted model on `mnist_test.csv` with accuracy and macro F1.

## Outputs

Provide a single project-root command that regenerates the metrics, predictions, figures, and `REPORT.md`. The completed project should include:

- reusable code in `src/`, tests covering the required behavior, and a dependency file with tested versions;
- `outputs/metrics.json` and `outputs/predictions.csv`;
- `outputs/figures/confusion_matrix.png` and `outputs/figures/sample_predictions.png`;
- `REPORT.md` with the method, measured metrics, run command, and test command.

Tests should cover headerless CSV loading, label separation, image reshaping, and a small end-to-end run. Keep the raw CSV files unchanged. Do not add model tuning, another classifier, a notebook, or an interface.
