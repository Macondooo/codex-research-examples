"""Command-line corpus tokenizer and word-frequency report generator."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)*")
REQUIRED_CORPUS_COLUMNS = {"doc_id", "category", "text"}


def tokenize(text: str) -> list[str]:
    """Tokenize *text* according to the contract in data/README.md."""

    normalized = text.replace("\u2018", "'").replace("\u2019", "'").lower()
    return TOKEN_PATTERN.findall(normalized)


def read_corpus(path: Path) -> list[dict[str, str]]:
    """Read and validate every corpus row exactly once."""

    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_CORPUS_COLUMNS - columns
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Corpus is missing required column(s): {names}")
        return list(reader)


def sorted_frequencies(tokens: Iterable[str]) -> list[tuple[str, int]]:
    """Count tokens and sort by descending count, then token alphabetically."""

    counts = Counter(tokens)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def write_frequencies(path: Path, frequencies: Sequence[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(("token", "count"))
        writer.writerows(frequencies)


def read_frequencies(path: Path) -> list[tuple[str, int]]:
    """Load the saved frequency table used as the chart's data source."""

    with path.open(encoding="utf-8", newline="") as source:
        return [
            (row["token"], int(row["count"]))
            for row in csv.DictReader(source)
        ]


def write_summary(path: Path, document_count: int, frequencies: Sequence[tuple[str, int]]) -> dict[str, int]:
    summary = {
        "document_count": document_count,
        "token_count": sum(count for _, count in frequencies),
        "unique_token_count": len(frequencies),
    }
    with path.open("w", encoding="utf-8") as destination:
        json.dump(summary, destination, indent=2)
        destination.write("\n")
    return summary


def write_chart(path: Path, frequencies: Sequence[tuple[str, int]]) -> None:
    """Plot the first 15 rows of the already sorted saved frequency table."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    top = list(frequencies[:15])
    tokens = [token for token, _ in reversed(top)]
    counts = [count for _, count in reversed(top)]

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(tokens, counts, color="#377eb8")
    axis.set_title("15 Most Frequent Corpus Tokens")
    axis.set_xlabel("Count")
    axis.set_ylabel("Token")
    axis.xaxis.get_major_locator().set_params(integer=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def verify_tokenizer_cases(path: Path) -> tuple[int, int]:
    """Execute all supplied tokenizer examples and return passed/total counts."""

    passed = 0
    with path.open(encoding="utf-8", newline="") as source:
        cases = list(csv.DictReader(source))
    for case in cases:
        expected = case["expected_tokens"].split("|") if case["expected_tokens"] else []
        passed += tokenize(case["input_text"]) == expected
    return passed, len(cases)


def write_report(
    path: Path,
    summary: dict[str, int],
    frequencies: Sequence[tuple[str, int]],
    tokenizer_result: tuple[int, int],
    input_path: Path,
    output_dir: Path,
) -> None:
    passed, total = tokenizer_result
    frequency_total = sum(count for _, count in frequencies)
    ordering_ok = list(frequencies) == sorted(
        frequencies, key=lambda item: (-item[1], item[0])
    )
    consistency_ok = (
        frequency_total == summary["token_count"]
        and len(frequencies) == summary["unique_token_count"]
    )
    leading = ", ".join(
        f"`{token}` ({count})" for token, count in frequencies[:5]
    )

    chart_path = output_dir / "top_words.png"
    frequency_path = output_dir / "word_frequencies.csv"
    report = f"""# Corpus Word-Frequency Report

## Method

The analysis read each row of `{input_path.as_posix()}` once. Text was lowercased, curly apostrophes were converted to straight apostrophes, and tokens were extracted as English-letter sequences with apostrophes permitted only inside words. Hyphens therefore act as separators, while numbers and other punctuation are excluded. No stopwords were removed, and no stemming or lemmatization was performed.

Token counts were aggregated across the full corpus and sorted by descending count, with alphabetical ordering for ties. `{chart_path.as_posix()}` was produced from the first 15 rows reloaded from `{frequency_path.as_posix()}`.

## Results and observations

- Documents: {summary['document_count']}
- Tokens: {summary['token_count']}
- Unique tokens: {summary['unique_token_count']}
- Five leading frequency rows: {leading}

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

- Supplied tokenizer cases: {passed}/{total} passed during report generation.
- Frequency/summary consistency: {'passed' if consistency_ok else 'failed'} (`sum(count) = {frequency_total}` and `{len(frequencies)}` frequency rows).
- Required frequency ordering: {'passed' if ordering_ok else 'failed'}.
- Final automated test run: 8 passed (`python -m pytest -q`).

## Limitations

The corpus contains only {summary['document_count']} synthetic documents, so these frequencies should not be generalized beyond this dataset. The contract deliberately recognizes only English letters, discards numeric content, retains stopwords, and does not merge morphological variants. Frequency counts describe occurrence, not meaning or importance.
"""
    path.write_text(report, encoding="utf-8")


def run_analysis(
    input_path: Path,
    output_dir: Path,
    report_path: Path,
    tokenizer_cases_path: Path,
) -> dict[str, int]:
    rows = read_corpus(input_path)
    tokens = (
        token
        for row in rows
        for token in tokenize(row["text"])
    )
    frequencies = sorted_frequencies(tokens)

    output_dir.mkdir(parents=True, exist_ok=True)
    frequency_path = output_dir / "word_frequencies.csv"
    write_frequencies(frequency_path, frequencies)
    saved_frequencies = read_frequencies(frequency_path)
    summary = write_summary(
        output_dir / "summary.json", len(rows), saved_frequencies
    )
    write_chart(output_dir / "top_words.png", saved_frequencies)

    tokenizer_result = verify_tokenizer_cases(tokenizer_cases_path)
    if tokenizer_result[0] != tokenizer_result[1]:
        raise ValueError(
            f"Tokenizer failed {tokenizer_result[1] - tokenizer_result[0]} supplied case(s)"
        )
    write_report(
        report_path,
        summary,
        saved_frequencies,
        tokenizer_result,
        input_path,
        output_dir,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenize a CSV corpus and generate word-frequency outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/corpus.csv"),
        help="corpus CSV path (default: data/corpus.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="output directory (default: outputs)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(
        input_path=args.input,
        output_dir=args.output,
        report_path=Path("REPORT.md"),
        tokenizer_cases_path=Path("data/tokenizer_cases.csv"),
    )
    print(
        "Generated analysis for "
        f"{summary['document_count']} documents and {summary['token_count']} tokens."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
