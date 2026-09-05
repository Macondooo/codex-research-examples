"""Data preparation, training, and evaluation for the Iris KNN example."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20
N_NEIGHBORS = 5


@dataclass(frozen=True)
class IrisData:
    features: NDArray[np.float64]
    targets: NDArray[np.int64]
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]


@dataclass(frozen=True)
class DataSplit:
    x_train: NDArray[np.float64]
    x_test: NDArray[np.float64]
    y_train: NDArray[np.int64]
    y_test: NDArray[np.int64]


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    confusion_matrix: NDArray[np.int64]
    predictions: NDArray[np.int64]


def load_iris_data() -> IrisData:
    """Load the bundled Iris dataset without network access."""
    bunch = load_iris()
    return IrisData(
        features=np.asarray(bunch.data, dtype=np.float64),
        targets=np.asarray(bunch.target, dtype=np.int64),
        feature_names=tuple(bunch.feature_names),
        target_names=tuple(bunch.target_names),
    )


def split_data(data: IrisData) -> DataSplit:
    """Create a stable, stratified 80/20 train/test split."""
    x_train, x_test, y_train, y_test = train_test_split(
        data.features,
        data.targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data.targets,
    )
    return DataSplit(x_train, x_test, y_train, y_test)


def train_model(split: DataSplit) -> Pipeline:
    """Fit scaling and KNN as one pipeline, preventing test-data leakage."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=N_NEIGHBORS)),
        ]
    )
    model.fit(split.x_train, split.y_train)
    return model


def evaluate(model: Pipeline, split: DataSplit) -> Evaluation:
    """Evaluate a fitted model on the held-out test partition."""
    predictions = np.asarray(model.predict(split.x_test), dtype=np.int64)
    matrix = confusion_matrix(split.y_test, predictions, labels=[0, 1, 2])
    return Evaluation(
        accuracy=float(accuracy_score(split.y_test, predictions)),
        confusion_matrix=np.asarray(matrix, dtype=np.int64),
        predictions=predictions,
    )


def data_quality_notes(data: IrisData) -> list[str]:
    """Return explicit observations about missing, invalid, and unusual data."""
    features = data.features
    labels, counts = np.unique(data.targets, return_counts=True)
    duplicate_rows = features.shape[0] - np.unique(features, axis=0).shape[0]
    return [
        f"Rows: {features.shape[0]}; numeric features: {features.shape[1]}.",
        f"Missing values: {int(np.isnan(features).sum())}.",
        f"Non-finite values: {int((~np.isfinite(features)).sum())}.",
        f"Non-positive measurements: {int((features <= 0).sum())}.",
        "Class counts: "
        + ", ".join(
            f"{data.target_names[int(label)]}={int(count)}"
            for label, count in zip(labels, counts)
        )
        + ".",
        f"Exact duplicate feature rows: {duplicate_rows}; retained because they are valid observations.",
        "No missing, invalid, or class-imbalance issue requires correction.",
    ]

