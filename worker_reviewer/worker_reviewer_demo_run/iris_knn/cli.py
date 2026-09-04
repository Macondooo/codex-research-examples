"""Command-line entry point that trains the model and saves compact results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from .model import data_quality_notes, evaluate, load_iris_data, split_data, train_model


def run(output_dir: Path) -> dict[str, object]:
    data = load_iris_data()
    split = split_data(data)
    model = train_model(split)
    result = evaluate(model, split)

    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "accuracy": result.accuracy,
        "confusion_matrix": result.confusion_matrix.tolist(),
        "class_labels": list(data.target_names),
        "random_state": 42,
        "test_size": 0.2,
        "n_neighbors": 5,
        "train_samples": int(split.x_train.shape[0]),
        "test_samples": int(split.x_test.shape[0]),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    notes = data_quality_notes(data)
    (output_dir / "data_quality.txt").write_text(
        "\n".join(f"- {note}" for note in notes) + "\n", encoding="utf-8"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=result.confusion_matrix,
        display_labels=data.target_names,
    )
    display.plot(cmap="Blues", colorbar=False, values_format="d")
    display.ax_.set_title("Iris KNN confusion matrix (test set)")
    display.figure_.tight_layout()
    display.figure_.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(display.figure_)

    return payload


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    payload = run(project_dir / "results")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

