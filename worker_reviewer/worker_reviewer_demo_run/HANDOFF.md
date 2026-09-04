# Worker handoff

## Files added

- `iris_knn/model.py`: dataset loading, reproducible split, pipeline training,
  evaluation, and data-quality checks.
- `iris_knn/cli.py`: command-line run plus JSON, text, and labeled heatmap output.
- `tests/test_model.py`: five tests covering data, reproducibility, leakage,
  KNN configuration, prediction/evaluation, quality notes, and output artifacts.
- `results/metrics.json`: compact accuracy and confusion-matrix result.
- `results/data_quality.txt`: missing, invalid, balance, and duplicate-row notes.
- `results/confusion_matrix.png`: labeled test-set heatmap.
- `README.md`, `requirements.txt`, `.gitignore`, and this handoff.
- `AGENTS.md`, `workflow.md`, `worker.md`, and `reviewer.md` were copied
  byte-for-byte from the source folder and left unchanged.

## Commands run

From `worker_reviewer_demo_run`:

```bash
# Failed because the host Anaconda SciPy cannot load libgfortran.5.dylib:
python -m unittest discover -s tests -v

# Created a healthy isolated environment after the host failure:
/opt/homebrew/bin/python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Passed: 5 tests.
.venv/bin/python -m unittest discover -s tests -v

# Passed and generated the checked-in results.
MPLCONFIGDIR=.matplotlib .venv/bin/python -m iris_knn.cli
```

The four `cmp` checks against the source workflow documents also passed.
The generated PNG was inspected at its native 1024×768 size; its title, axes,
class labels, and cell values are legible.

## Main result

- Accuracy: `0.9333333333333333` (28 correct predictions out of 30).
- Confusion matrix, with rows=true and columns=predicted in the order setosa,
  versicolor, virginica: `[[10, 0, 0], [0, 10, 0], [0, 2, 8]]`.
- Configuration: `random_state=42`, stratified 80/20 split, training-fitted
  `StandardScaler`, and `KNeighborsClassifier(n_neighbors=5)`.

## Data notes and uncertainties

The bundled dataset has 150 rows, four numeric features, balanced classes,
zero missing values, zero non-finite values, and zero non-positive measurements.
It contains one exact duplicate feature row; this valid observation was retained.

The result is deterministic for the installed dependency versions. Exact behavior
could vary with substantially different future scikit-learn versions. This is a
Worker handoff, not an approval; an independent Reviewer still needs to inspect it.
