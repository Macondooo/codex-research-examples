# Repository Instructions

## Scope

- Read `TASK.md`, `REQUEST.md`, and `data/README.md` before making changes.
- Treat `TASK.md`, `data/`, and `expected/` as supplied course material. Do not edit them.
- Treat `REQUEST.md` as the task-specific execution brief. Follow it without modifying it.
- Implement only the stated requirements. Do not add unrelated features or restructure supplied material.

## Project layout

- Put reusable Python code in `src/` and automated checks in `tests/`.
- Write generated tables, summaries, and figures to `outputs/`.
- Record dependencies in `requirements.txt` or `pyproject.toml`.
- Use `REPORT.md` for the final analysis; keep the course `README.md` intact.

## Verification

- Test the tokenizer against every row in `data/tokenizer_cases.csv`.
- Run the documented test and analysis commands before reporting completion.
- Check saved outputs against `expected/README.md` and record any unresolved limitations.
