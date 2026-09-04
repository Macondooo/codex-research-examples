# Assignment

Implement a Python command-line project that tokenizes every document in `data/corpus.csv` and reports word frequencies for the complete corpus.

## Required behavior

- Follow the tokenizer rules in `data/README.md` exactly.
- Process each corpus row exactly once.
- Provide one documented command that regenerates all analysis outputs.
- Sort frequencies by count from high to low, then alphabetically for ties.
- Include an automated test for every example in `data/tokenizer_cases.csv`.

## Required outputs

- `outputs/word_frequencies.csv` with `token` and `count` columns
- `outputs/summary.json` with document, token, and unique-token counts
- `outputs/top_words.png` showing the 15 most frequent tokens
- `REPORT.md` with the method, main observations, and limitations
- automated tests and a dependency file

Do not remove stopwords, perform stemming or lemmatization, perform topic modeling, or replace the supplied rules with library defaults.
