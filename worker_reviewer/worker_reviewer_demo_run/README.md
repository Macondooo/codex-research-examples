# Iris KNN demo

This small project trains a 5-nearest-neighbors classifier on scikit-learn's
built-in Iris dataset. The split is reproducible, and the scaler is fitted only
on the training partition.

## Run

```bash
python -m iris_knn.cli
python -m unittest discover -s tests -v
```

The CLI writes `results/metrics.json`, `results/data_quality.txt`, and
`results/confusion_matrix.png`.

