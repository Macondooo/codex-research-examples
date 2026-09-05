"""Reproducible MNIST CSV baseline: run from the project root."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mnist-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

RUN_COMMAND = ".venv/bin/python -m src.mnist_experiment"
TEST_COMMAND = ".venv/bin/python -m unittest discover -s tests -v"


def load_csv(path):
    """Read all headerless rows, separate labels, and scale row-major pixels."""
    rows = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if rows.shape[0] == 0 or rows.shape[1] != 785:
        raise ValueError("Expected nonempty CSV with 785 columns")
    if not np.isfinite(rows).all() or not np.equal(rows, np.floor(rows)).all():
        raise ValueError("CSV values must be finite integers")
    labels, pixels = rows[:, 0], rows[:, 1:]
    if np.any((labels < 0) | (labels > 9)) or np.any((pixels < 0) | (pixels > 255)):
        raise ValueError("Labels must be 0–9 and pixels 0–255")
    return pixels / 255.0, labels.astype(np.int64)


def image_from_pixels(pixels):
    return np.asarray(pixels).reshape(28, 28, order="C")


def save_figures(directory, matrix, pixels, truth, predicted):
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 15, "axes.titlesize": 20, "axes.labelsize": 17}):
        fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
        im = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(im, ax=ax, label="Test images")
        for row in range(10):
            for col in range(10):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                        fontsize=13, color="white" if matrix[row, col] > matrix.max() / 2 else "black")
        ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit",
               ylabel="True digit", title="MNIST test confusion matrix")
        fig.savefig(directory / "confusion_matrix.png", dpi=160)
        plt.close(fig)

        # Fixed first 20 test rows; no selection based on prediction quality.
        fig, axes = plt.subplots(4, 5, figsize=(14, 12), layout="constrained")
        fig.suptitle("MNIST — first 20 test images\nTrue and predicted digits (errors in red)", fontsize=22)
        for index, ax in enumerate(axes.flat):
            ax.axis("off")
            if index < len(truth):
                ax.imshow(image_from_pixels(pixels[index]), cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
                ax.set_title(f"True: {truth[index]}   Pred: {predicted[index]}", fontsize=17,
                             color="#b22222" if truth[index] != predicted[index] else "black")
        fig.savefig(directory / "sample_predictions.png", dpi=160)
        plt.close(fig)


def run_experiment(train_path, test_path, output_dir, report_path):
    train_pixels, train_labels = load_csv(train_path)
    test_pixels, test_labels = load_csv(test_path)
    model = SGDClassifier(loss="log_loss", random_state=42)
    model.fit(train_pixels, train_labels)
    predicted = model.predict(test_pixels)
    matrix = confusion_matrix(test_labels, predicted, labels=np.arange(10))
    metrics = {
        "model": "SGDClassifier", "loss": "log_loss", "seed": 42,
        "training_rows": len(train_labels), "test_rows": len(test_labels),
        "accuracy": float(accuracy_score(test_labels, predicted)),
        "macro_f1": float(f1_score(test_labels, predicted, average="macro", zero_division=0)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", "predicted_label"])
        writer.writerows(zip(test_labels, predicted))
    save_figures(output_dir / "figures", matrix, test_pixels, test_labels, predicted)
    # Generate the report by reading the saved metrics and predictions.
    write_report(output_dir, Path(report_path))
    return metrics


def write_report(output_dir, report_path):
    metrics = json.loads((output_dir / "metrics.json").read_text())
    saved = np.loadtxt(output_dir / "predictions.csv", delimiter=",", skiprows=1, dtype=int, ndmin=2)
    matrix = confusion_matrix(saved[:, 0], saved[:, 1], labels=np.arange(10))
    confusions = sorted(((int(matrix[t, p]), t, p) for t in range(10) for p in range(10)
                         if t != p and matrix[t, p]), key=lambda item: (-item[0], item[1], item[2]))[:5]
    details = "\n".join(f"- True {t} → predicted {p}: {count} images."
                        for count, t, p in confusions) or "No incorrect predictions."
    report_path.write_text(f"""# MNIST experiment results

Each headerless row contains one label and 784 row-major pixels. Pixels are divided
by 255. One scikit-learn `SGDClassifier(loss=\"log_loss\", random_state=42)` is fitted
on the supplied training file; all other model parameters retain library defaults.
The supplied test file is used only for evaluation. No tuning is performed.

Training rows: {metrics['training_rows']:,}. Test rows: {metrics['test_rows']:,}.

- Accuracy: {metrics['accuracy']:.10f} ({metrics['accuracy']:.2%}).
- Macro F1: {metrics['macro_f1']:.10f}.

Macro F1 is the unweighted mean of per-class F1 scores. The most common directional
confusions in the saved final predictions are:

{details}

The confusion matrix uses true digits on rows and predicted digits on columns,
both ordered 0–9. The sample figure shows the first 20 test rows in source order;
incorrect predictions have red titles. These examples are not a separate evaluation.

## Reproduction

From the project root, create a Python {metrics['versions']['python']} environment and install dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the experiment (regenerates all artifacts and this report):

```sh
{RUN_COMMAND}
```

Run the tests:

```sh
{TEST_COMMAND}
```

Tested versions: {json.dumps(metrics['versions'], sort_keys=True)}.
Metrics and this report derive from the same predictions saved in
`outputs/predictions.csv`; figures are in `outputs/figures/`.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/mnist_train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/raw/mnist_test.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--report", type=Path, default=Path("REPORT.md"))
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.train, args.test, args.output, args.report), indent=2))


if __name__ == "__main__":
    main()
