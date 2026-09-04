# Corpus Word-Frequency Report

## Method

The analysis read each row of `data/corpus.csv` once. Text was lowercased, curly apostrophes were converted to straight apostrophes, and tokens were extracted as English-letter sequences with apostrophes permitted only inside words. Hyphens therefore act as separators, while numbers and other punctuation are excluded. No stopwords were removed, and no stemming or lemmatization was performed.

Token counts were aggregated across the full corpus and sorted by descending count, with alphabetical ordering for ties. `outputs/top_words.png` was produced from the first 15 rows reloaded from `outputs/word_frequencies.csv`.

## Results and observations

- Documents: 20
- Tokens: 178
- Unique tokens: 126
- Five leading frequency rows: `the` (18), `a` (6), `and` (5), `data` (4), `team` (4)

The most frequent words reflect both the corpus subject matter and retained function words. Because inflectional variants are intentionally not normalized, forms such as singulars and plurals remain distinct entries.

## Reproduction commands

From the project root, create and activate an isolated environment, then install the two allowed dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Regenerate the CSV, JSON, chart, and this report with one analysis command:

```bash
python -m src.text_analysis
```

Run the automated tests with:

```bash
python -m pytest -q
```

## Verification evidence

- Supplied tokenizer cases: 6/6 passed during report generation.
- Frequency/summary consistency: passed (`sum(count) = 178` and `126` frequency rows).
- Required frequency ordering: passed.
- Final automated test run: 8 passed (`python -m pytest -q`).

## Limitations

The corpus contains only 20 synthetic documents, so these frequencies should not be generalized beyond this dataset. The contract deliberately recognizes only English letters, discards numeric content, retains stopwords, and does not merge morphological variants. Frequency counts describe occurrence, not meaning or importance.
