# Data Card

The dataset is an original synthetic corpus prepared for this exercise. It contains no personal data or reproduced external text.

## Files

- `corpus.csv`: 20 documents with `doc_id`, `category`, and `text` columns.
- `tokenizer_cases.csv`: 6 tokenizer test cases. In `expected_tokens`, `|` is a delimiter and is not part of any token.

## Tokenizer contract

- Convert letters to lowercase before counting.
- A token contains English letters and may contain an apostrophe inside the word.
- Convert curly apostrophes to straight apostrophes. Preserve apostrophes that occur inside words.
- Treat hyphens as token separators.
- Exclude numbers and punctuation.
- Do not remove stopwords, stem, or lemmatize.

Both data files are immutable during the exercise.
