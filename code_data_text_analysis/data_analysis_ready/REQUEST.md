# Request

## Outcome

Build a reproducible command-line project that uses scikit-learn Ridge regression to complete the experiment in `TASK.md`. A single documented command from the repository root must rerun the analysis and regenerate every reported artifact.

## Additional Constraints

- Limit direct dependencies to pandas, scikit-learn, Matplotlib, and pytest. Use a scikit-learn pipeline for preprocessing and the model.
- Provide a command-line interface; do not require notebook or interactive execution.
- Use `pytest` for the required tests. Limit `REPORT.md` to the sections required by `TASK.md`.
- Do not add model tuning, extra models, dashboards, or packaging infrastructure.

## Acceptance Criteria

- The documented environment setup, test command, and analysis command execute successfully from a fresh copy of this directory.
- The complete test suite passes and its output is reported.
- The analysis command creates `metrics.json`, `predictions.csv`, and the plot under `outputs/`.
- The saved artifacts satisfy `expected/README.md`, and `REPORT.md` reports values read from those artifacts.
- The final response lists the commands executed, summarizes the measured result, and states any remaining limitations.
