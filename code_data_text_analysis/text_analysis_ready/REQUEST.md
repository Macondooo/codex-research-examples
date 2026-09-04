# Request

## Outcome

Build a Python command-line application that reads `data/corpus.csv`, follows the supplied tokenizer contract, and regenerates every required analysis output.

## Additional Constraints

- Use the Python standard library for CSV, JSON, tokenization, and counting.
- Use Matplotlib for chart generation and pytest for automated tests. Do not introduce other third-party packages.
- Use `data/corpus.csv` and `outputs/` as the default input and output paths. Provide optional command-line arguments for both paths.
- Document the exact commands for environment setup, analysis execution, and testing.

## Acceptance Criteria

- The documented setup, analysis, and test commands execute successfully in a clean virtual environment.
- Tests cover all supplied tokenizer cases, frequency ordering, and count consistency, and they pass.
- A single documented analysis command regenerates the required CSV, JSON, chart, and report from the unchanged input data.
- The totals in `outputs/summary.json` match the aggregates in `outputs/word_frequencies.csv`, and `outputs/top_words.png` is derived from the same frequency table.
- The final response records the commands executed, test evidence, output consistency checks, and any unresolved limitation.
