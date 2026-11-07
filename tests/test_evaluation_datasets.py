"""Tests for multiple-choice benchmark loading."""

from __future__ import annotations

import datasets
import pytest

from looped_internals.evaluation.datasets import load_mcq_dataset


@pytest.mark.parametrize(
    ("dataset_name", "expected_args", "document"),
    [
        (
            "arc_easy",
            ("allenai/ai2_arc", "ARC-Easy"),
            {
                "question": "Which answer is correct?",
                "choices": {"label": ["A", "B", "C", "D"], "text": ["a", "b", "c", "d"]},
                "answerKey": "B",
            },
        ),
        (
            "arc_challenge",
            ("allenai/ai2_arc", "ARC-Challenge"),
            {
                "question": "Which answer is correct?",
                "choices": {"label": ["A", "B", "C", "D"], "text": ["a", "b", "c", "d"]},
                "answerKey": "B",
            },
        ),
        (
            "hellaswag",
            ("Rowan/hellaswag",),
            {"ctx": "A person starts", "endings": ["a", "b", "c", "d"], "label": "1"},
        ),
        (
            "winogrande",
            ("allenai/winogrande", "winogrande_xl"),
            {
                "sentence": "The _ is here.",
                "option1": "first",
                "option2": "second",
                "answer": "2",
            },
        ),
    ],
)
def test_load_mcq_dataset_uses_parquet_hub_repositories(
    monkeypatch, dataset_name, expected_args, document
):
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [document]

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)

    examples = load_mcq_dataset(dataset_name, "validation")

    assert len(examples) == 1
    assert calls == [(expected_args, {"split": "validation"})]
