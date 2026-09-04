import csv
import json
from pathlib import Path

import pytest

from src.text_analysis import run_analysis, sorted_frequencies, tokenize


ROOT = Path(__file__).parents[1]


def supplied_tokenizer_cases():
    with (ROOT / "data/tokenizer_cases.csv").open(encoding="utf-8", newline="") as source:
        return [
            pytest.param(
                row["input_text"],
                row["expected_tokens"].split("|"),
                id=f"supplied-case-{index}",
            )
            for index, row in enumerate(csv.DictReader(source), start=1)
        ]


@pytest.mark.parametrize("input_text,expected", supplied_tokenizer_cases())
def test_every_supplied_tokenizer_case(input_text, expected):
    assert tokenize(input_text) == expected


def test_frequencies_sort_by_count_then_alphabetically():
    assert sorted_frequencies(
        ["beta", "alpha", "gamma", "beta", "alpha", "top", "top"]
    ) == [
        ("alpha", 2),
        ("beta", 2),
        ("top", 2),
        ("gamma", 1),
    ]


def test_generated_counts_are_consistent(tmp_path):
    output_dir = tmp_path / "outputs"
    run_analysis(
        input_path=ROOT / "data/corpus.csv",
        output_dir=output_dir,
        report_path=tmp_path / "REPORT.md",
        tokenizer_cases_path=ROOT / "data/tokenizer_cases.csv",
    )

    with (output_dir / "summary.json").open(encoding="utf-8") as source:
        summary = json.load(source)
    with (output_dir / "word_frequencies.csv").open(encoding="utf-8", newline="") as source:
        frequencies = [
            (row["token"], int(row["count"]))
            for row in csv.DictReader(source)
        ]

    assert summary["document_count"] == 20
    assert sum(count for _, count in frequencies) == summary["token_count"]
    assert len(frequencies) == summary["unique_token_count"]
    assert frequencies == sorted(frequencies, key=lambda item: (-item[1], item[0]))
    assert (output_dir / "top_words.png").stat().st_size > 0
    assert (tmp_path / "REPORT.md").stat().st_size > 0
