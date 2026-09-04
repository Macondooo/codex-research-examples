"""Run the reproducible life-expectancy regression experiment."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

# Use writable, stable cache locations in restricted and CI environments.
_CACHE_DIR = str(Path(tempfile.gettempdir()) / "life_expectancy_analysis_cache")
os.environ.setdefault("MPLCONFIGDIR", _CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ID_COLUMN = "community_id"
TARGET_COLUMN = "life_expectancy_years"
FEATURE_COLUMNS = [
    "income_per_capita_usd",
    "healthcare_access_pct",
    "schooling_years",
    "clean_water_pct",
    "smoking_rate_pct",
    "infant_mortality_per_1000",
]
REQUIRED_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
RANDOM_SEED = 42
TEST_SIZE = 0.25


def load_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the source data without modifying it."""
    path = Path(path)
    frame = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "Input data is missing required columns: " + ", ".join(missing_columns)
        )
    if frame[ID_COLUMN].isna().any():
        raise ValueError(f"Input data contains missing values in {ID_COLUMN}")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"Input data contains duplicate values in {ID_COLUMN}")

    numeric_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    try:
        frame[numeric_columns] = frame[numeric_columns].apply(
            pd.to_numeric, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Feature and target columns must be numeric") from exc
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"Input data contains missing values in {TARGET_COLUMN}")
    return frame


def split_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the fixed train/test split while excluding the identifier."""
    features = frame.loc[:, FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN]
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )


def build_pipeline() -> Pipeline:
    """Build the preprocessing and single Ridge regression model."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", Ridge()),
        ]
    )


def calculate_metrics(actual: pd.Series, predicted: Sequence[float]) -> dict[str, float]:
    """Calculate the three required regression metrics."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def save_plot(predictions: pd.DataFrame, path: Path) -> None:
    """Save a predicted-versus-actual comparison for both predictors."""
    actual = predictions["actual_life_expectancy_years"]
    low = min(
        actual.min(),
        predictions["model_prediction"].min(),
        predictions["baseline_prediction"].min(),
    )
    high = max(
        actual.max(),
        predictions["model_prediction"].max(),
        predictions["baseline_prediction"].max(),
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predictions["model_prediction"], label="Ridge model", s=55)
    ax.scatter(
        actual,
        predictions["baseline_prediction"],
        label="Training-mean baseline",
        marker="x",
        s=65,
    )
    ax.plot([low, high], [low, high], linestyle="--", color="gray", label="Ideal")
    ax.set_xlabel("Actual life expectancy (years)")
    ax.set_ylabel("Predicted life expectancy (years)")
    ax.set_title("Predicted versus actual life expectancy")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_analysis(input_path: str | Path, output_dir: str | Path) -> dict:
    """Fit, evaluate, and save every required artifact."""
    frame = load_data(input_path)
    x_train, x_test, y_train, y_test = split_data(frame)

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    model_predictions = pipeline.predict(x_test)
    baseline_value = float(y_train.mean())
    baseline_predictions = [baseline_value] * len(y_test)

    metrics = {
        "model": calculate_metrics(y_test, model_predictions),
        "baseline": calculate_metrics(y_test, baseline_predictions),
        "split": {
            "random_seed": RANDOM_SEED,
            "test_size": TEST_SIZE,
            "training_rows": len(x_train),
            "test_rows": len(x_test),
        },
    }

    predictions = pd.DataFrame(
        {
            ID_COLUMN: frame.loc[y_test.index, ID_COLUMN].to_numpy(),
            "actual_life_expectancy_years": y_test.to_numpy(),
            "model_prediction": model_predictions,
            "baseline_prediction": baseline_predictions,
        }
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    save_plot(predictions, output_dir / "predicted_vs_actual.png")
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the life-expectancy Ridge regression experiment."
    )
    parser.add_argument("--input", default="data/life_expectancy.csv")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = run_analysis(args.input, args.output_dir)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
