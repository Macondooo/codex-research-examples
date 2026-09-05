# MNIST experiment results

Each headerless row contains one label and 784 row-major pixels. Pixels are divided
by 255. One scikit-learn `SGDClassifier(loss="log_loss", random_state=42)` is fitted
on the supplied training file; all other model parameters retain library defaults.
The supplied test file is used only for evaluation. No tuning is performed.

Training rows: 60,000. Test rows: 10,000.

- Accuracy: 0.9155000000 (91.55%).
- Macro F1: 0.9141348109.

Macro F1 is the unweighted mean of per-class F1 scores. The most common directional
confusions in the saved final predictions are:

- True 5 → predicted 3: 51 images.
- True 4 → predicted 9: 40 images.
- True 7 → predicted 9: 40 images.
- True 2 → predicted 8: 33 images.
- True 8 → predicted 3: 32 images.

The confusion matrix uses true digits on rows and predicted digits on columns,
both ordered 0–9. The sample figure shows the first 20 test rows in source order;
incorrect predictions have red titles. These examples are not a separate evaluation.

## Reproduction

From the project root, create a Python 3.13.5 environment and install dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the experiment (regenerates all artifacts and this report):

```sh
.venv/bin/python -m src.mnist_experiment
```

Run the tests:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tested versions: {"matplotlib": "3.11.1", "numpy": "2.5.2", "python": "3.13.5", "scikit-learn": "1.9.0"}.
Metrics and this report derive from the same predictions saved in
`outputs/predictions.csv`; figures are in `outputs/figures/`.
