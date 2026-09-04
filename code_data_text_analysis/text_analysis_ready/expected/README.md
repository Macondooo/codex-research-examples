# Reference Result

A complete submission contains executable code, automated tests, a dependency file, `REPORT.md`, and all three required files in `outputs/`.

Apply the following acceptance checks:

- The summary reports 20 documents.
- All 6 supplied tokenizer cases pass.
- The sum of `count` in `word_frequencies.csv` equals the token count in `summary.json`.
- The number of frequency rows equals the unique-token count in `summary.json`.
- Equal-frequency tokens are ordered alphabetically.
- The chart is derived from `word_frequencies.csv`.
- `REPORT.md` records the values stored in the generated outputs and the executed test evidence.

Package structure and library selection are not prescribed. Evaluation is based on correctness and reproducibility; additional features do not contribute to acceptance.
