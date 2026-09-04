import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from src.analysis import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    REQUIRED_COLUMNS,
    build_pipeline,
    load_data,
    run_analysis,
    split_data,
)


DATA_PATH = Path("data/life_expectancy.csv")


def test_load_data_and_missing_column_validation(tmp_path):
    frame = load_data(DATA_PATH)
    assert len(frame) == 32
    assert list(frame.columns) == REQUIRED_COLUMNS

    invalid_path = tmp_path / "invalid.csv"
    frame.drop(columns=["schooling_years"]).to_csv(invalid_path, index=False)
    with pytest.raises(ValueError, match="schooling_years"):
        load_data(invalid_path)


def test_identifier_is_excluded_and_split_is_complete():
    frame = load_data(DATA_PATH)
    x_train, x_test, y_train, y_test = split_data(frame)

    assert list(x_train.columns) == FEATURE_COLUMNS
    assert ID_COLUMN not in x_train.columns
    assert len(x_train) == len(y_train) == 24
    assert len(x_test) == len(y_test) == 8
    assert set(x_train.index).isdisjoint(x_test.index)
    assert set(x_train.index) | set(x_test.index) == set(frame.index)


def test_pipeline_handles_missing_values_with_training_fitted_imputer():
    frame = load_data(DATA_PATH)
    x_train, x_test, y_train, _ = split_data(frame)
    pipeline = build_pipeline().fit(x_train, y_train)

    imputer = pipeline.named_steps["preprocessing"].named_transformers_[
        "numeric"
    ].named_steps["imputer"]
    assert isinstance(imputer, SimpleImputer)
    expected_medians = x_train.median().to_numpy()
    assert imputer.statistics_ == pytest.approx(expected_medians)
    assert len(pipeline.predict(x_test)) == 8


def test_output_schema_and_shared_test_rows(tmp_path):
    run_analysis(DATA_PATH, tmp_path)

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    assert set(metrics) == {"model", "baseline", "split"}
    assert set(metrics["model"]) == {"mae", "rmse", "r2"}
    assert set(metrics["baseline"]) == {"mae", "rmse", "r2"}
    assert metrics["split"]["training_rows"] == 24
    assert metrics["split"]["test_rows"] == 8
    assert list(predictions.columns) == [
        ID_COLUMN,
        "actual_life_expectancy_years",
        "model_prediction",
        "baseline_prediction",
    ]
    assert len(predictions) == 8
    assert predictions[ID_COLUMN].is_unique
    assert (tmp_path / "predicted_vs_actual.png").stat().st_size > 0
