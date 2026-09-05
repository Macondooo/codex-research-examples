import tempfile
import unittest
from pathlib import Path

import numpy as np

from iris_knn.cli import run
from iris_knn.model import (
    N_NEIGHBORS,
    RANDOM_STATE,
    data_quality_notes,
    evaluate,
    load_iris_data,
    split_data,
    train_model,
)


class IrisKNNTests(unittest.TestCase):
    def test_data_and_split_are_valid_and_reproducible(self) -> None:
        data = load_iris_data()
        first = split_data(data)
        second = split_data(data)

        self.assertEqual(data.features.shape, (150, 4))
        self.assertEqual(first.x_train.shape, (120, 4))
        self.assertEqual(first.x_test.shape, (30, 4))
        np.testing.assert_array_equal(first.x_train, second.x_train)
        np.testing.assert_array_equal(first.y_test, second.y_test)
        self.assertEqual(RANDOM_STATE, 42)
        self.assertTrue(np.isfinite(data.features).all())

    def test_model_uses_training_only_scaling_and_k_five(self) -> None:
        data = load_iris_data()
        split = split_data(data)
        model = train_model(split)

        scaler = model.named_steps["scaler"]
        knn = model.named_steps["knn"]
        np.testing.assert_allclose(scaler.mean_, split.x_train.mean(axis=0))
        self.assertFalse(np.allclose(scaler.mean_, data.features.mean(axis=0)))
        self.assertEqual(knn.n_neighbors, N_NEIGHBORS)
        self.assertEqual(N_NEIGHBORS, 5)

    def test_prediction_and_evaluation_flow(self) -> None:
        split = split_data(load_iris_data())
        result = evaluate(train_model(split), split)

        self.assertEqual(result.predictions.shape, (30,))
        self.assertEqual(result.confusion_matrix.shape, (3, 3))
        self.assertEqual(int(result.confusion_matrix.sum()), 30)
        self.assertGreaterEqual(result.accuracy, 0.8)

    def test_cli_writes_all_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            payload = run(output_dir)

            self.assertEqual(payload["n_neighbors"], 5)
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertTrue((output_dir / "data_quality.txt").is_file())
            self.assertGreater((output_dir / "confusion_matrix.png").stat().st_size, 0)
            self.assertIn("Missing values: 0", (output_dir / "data_quality.txt").read_text())

    def test_quality_notes_disclose_duplicate_rows(self) -> None:
        notes = data_quality_notes(load_iris_data())
        self.assertTrue(any("Exact duplicate feature rows" in note for note in notes))


if __name__ == "__main__":
    unittest.main()

