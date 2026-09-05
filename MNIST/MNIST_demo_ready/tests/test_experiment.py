import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.mnist_experiment import image_from_pixels, load_csv, run_experiment


class ExperimentTests(unittest.TestCase):
    def test_headerless_label_separation_and_scaling(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.csv"
            rows = np.zeros((2, 785), dtype=int)
            rows[:, 0] = [7, 2]
            rows[0, 1:4] = [255, 128, 0]
            rows[1, -1] = 255
            np.savetxt(path, rows, delimiter=",", fmt="%d")
            pixels, labels = load_csv(path)
            np.testing.assert_array_equal(labels, [7, 2])
            self.assertEqual(pixels.shape, (2, 784))
            np.testing.assert_allclose(pixels[0, :3], [1, 128 / 255, 0])
            self.assertEqual(pixels[1, -1], 1)

    def test_row_major_reshape(self):
        pixels = np.arange(784)
        image = image_from_pixels(pixels)
        self.assertEqual(image.shape, (28, 28))
        self.assertEqual(image[0, 27], 27)
        self.assertEqual(image[1, 0], 28)
        self.assertEqual(image[27, 27], 783)

    def test_invalid_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.csv"
            for column, value in [(0, 10), (1, 256), (1, -1), (1, 0.5), (1, float("nan"))]:
                rows = np.zeros((1, 785))
                rows[0, column] = value
                np.savetxt(path, rows, delimiter=",")
                with self.assertRaises(ValueError):
                    load_csv(path)
            path.write_text("1,2,3\n")
            with self.assertRaises(ValueError):
                load_csv(path)

    def test_end_to_end_and_training_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = np.zeros((40, 785), dtype=int)
            train[:, 0] = np.tile([0, 1], 20)
            train[:, 1] = train[:, 0] * 255
            train[:, 2] = (1 - train[:, 0]) * 255
            test = train[[1, 0, 3, 2]].copy()
            test[:, 3] = 128  # Distinguishes test rows from every training row.
            for name, rows in [("train", train), ("test", test)]:
                np.savetxt(root / f"{name}.csv", rows, delimiter=",", fmt="%d")
            real_fit = SGDClassifier.fit
            fits = []

            def tracked_fit(model, x, y):
                fits.append((x.copy(), y.copy()))
                return real_fit(model, x, y)

            with patch.object(SGDClassifier, "fit", tracked_fit):
                metrics = run_experiment(root / "train.csv", root / "test.csv",
                                         root / "outputs", root / "REPORT.md")
            self.assertEqual(len(fits), 1)
            np.testing.assert_array_equal(fits[0][0], train[:, 1:] / 255)
            np.testing.assert_array_equal(fits[0][1], train[:, 0])
            saved = np.loadtxt(root / "outputs/predictions.csv", delimiter=",", skiprows=1, dtype=int)
            np.testing.assert_array_equal(saved[:, 0], test[:, 0])
            self.assertEqual(metrics["accuracy"], accuracy_score(saved[:, 0], saved[:, 1]))
            self.assertEqual(metrics["macro_f1"], f1_score(saved[:, 0], saved[:, 1], average="macro"))
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertEqual(metrics["training_rows"], 40)
            self.assertEqual(metrics["test_rows"], 4)
            self.assertEqual(metrics, json.loads((root / "outputs/metrics.json").read_text()))
            report = (root / "REPORT.md").read_text()
            self.assertIn(f"{metrics['macro_f1']:.10f}", report)
            for name in ["confusion_matrix", "sample_predictions"]:
                self.assertGreater((root / f"outputs/figures/{name}.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
