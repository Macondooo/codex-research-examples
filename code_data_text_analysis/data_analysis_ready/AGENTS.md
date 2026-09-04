# Repository Instructions

## Required context

- Read `TASK.md`, `REQUEST.md`, and `data/README.md` before changing files.
- Treat `TASK.md` as the project contract. `REQUEST.md` may add constraints but must not weaken the contract.
- Keep `TASK.md`, `REQUEST.md`, `data/`, and `expected/` unchanged during implementation.

## Implementation requirements

- Implement only the functionality required by `TASK.md` and `REQUEST.md`; avoid unrelated features or refactoring.
- Put reusable Python code in `src/`, tests in `tests/`, and generated artifacts in `outputs/`.
- Provide one documented command that runs the analysis from the repository root.
- Use a fixed random seed. Fit preprocessing only on training data.
- Keep the source CSV read-only and preserve `community_id` in result tables for traceability.
- Do not add a dashboard, service, or notebook-only workflow.

## Verification

- Validate the input schema and raise an explicit error when required columns are missing.
- Run the tests and the complete analysis before reporting completion.
- Derive reported values from the saved run artifacts rather than manually transcribed calculations.
- In the final response, list the commands executed, the principal result, and any remaining limitations.
