# Workflow

The task is to reproduce a simple Iris classifier with scikit-learn. Use the built-in Iris dataset, keep the train/test split reproducible with `random_state=42`, scale the features, and train a KNN model with `k=5`.

The final project should show the model's accuracy and confusion matrix, include a labeled confusion-matrix heatmap, basic tests, and note any missing, invalid, or unusual data found along the way.

The Worker implements and runs the example, then leaves a short handoff. A separate Reviewer checks the code, tests, outputs, and data issues without changing the project. The Main Agent reads that review and decides whether the Worker needs a focused fix.

Do not download another dataset, tune several models, or add notebooks, interfaces, and unrelated features. The point is the handoff and independent review, not building a large machine-learning project.

