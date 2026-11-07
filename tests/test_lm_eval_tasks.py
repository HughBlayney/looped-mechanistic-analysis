"""Exercise packaged YAMLs through lm-eval without dataset/model downloads."""

from __future__ import annotations

import logging
import math

import datasets
import lm_eval
import pytest
from lm_eval.models.dummy import DummyLM

from looped_internals.evaluation.lm_eval import get_task_manager


@pytest.fixture
def task_manager(monkeypatch, tmp_path):
    """Keep real task loading/preprocessing, replacing only dataset downloads."""
    examples = {
        "openai/gsm8k": {
            "question": "What is 1 + 1?",
            "answer": "One plus one is two.\n#### 2",
        },
        "Rowan/hellaswag": {
            "activity_label": "Cooking",
            "ctx_a": "A person puts a pan on the stove.",
            "ctx_b": "they",
            "endings": [
                "put on their shoes.",
                "jump into a pool.",
                "turn on the burner. [artifact]",
                "park the car.",
            ],
            "label": "2",
        },
        "allenai/c4": {"text": "café @-@ noir ."},
    }

    def load_dataset(path, *args, **kwargs):
        example = datasets.Dataset.from_list([examples[path]])
        splits = datasets.DatasetDict(
            {split: example for split in ["train", "validation", "test"]}
        )
        if path == "Rowan/hellaswag":
            # The real test split is unlabelled and cannot supply the reference.
            splits["test"] = datasets.Dataset.from_list(
                [{**examples[path], "label": ""}]
            )
        return splits

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    monkeypatch.chdir(tmp_path)
    return get_task_manager()


def _request(task, doc):
    return task.construct_requests(
        doc, ctx=task.doc_to_text(doc), metadata=(task.config.task, 0, 1)
    )


def test_packaged_tasks_and_builtins_load_outside_repo(task_manager):
    names = ["gsm8k_likelihood", "hellaswag_likelihood", "c4_likelihood"]
    tasks = task_manager.load([*names, "hellaswag"])["tasks"]
    for name in names:
        task = tasks[name]
        assert task.config.num_fewshot == 0
        request = _request(task, task.eval_docs[0])
        assert request.request_type == "loglikelihood"
        assert all(isinstance(arg, str) for arg in request.args)
    assert tasks["hellaswag"].OUTPUT_TYPE == "multiple_choice"


def test_hellaswag_scores_same_text_as_correct_multiple_choice(task_manager):
    tasks = task_manager.load(["hellaswag_likelihood", "hellaswag"])["tasks"]
    likelihood = tasks["hellaswag_likelihood"]
    standard = tasks["hellaswag"]
    assert likelihood.config.validation_split == "validation"
    assert likelihood.config.test_split is None
    request = _request(likelihood, likelihood.eval_docs[0])
    choices = _request(standard, standard.eval_docs[0])

    assert request.args == (
        "Cooking: A person puts a pan on the stove. They",
        " turn on the burner. ",
    )
    assert len(choices) == 4
    assert request.args == choices[2].args


def test_simple_evaluate_uses_packaged_correct_ending(
    task_manager, monkeypatch, caplog
):
    model = DummyLM()
    scored = []

    def score(requests):
        scored.extend(request.args for request in requests)
        return [(-3.0, False) for _ in requests]

    monkeypatch.setattr(model, "loglikelihood", score)
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=["hellaswag_likelihood"],
        task_manager=task_manager,
        num_fewshot=0,
        bootstrap_iters=0,
        log_samples=False,
    )
    assert scored == [
        ("Cooking: A person puts a pan on the stove. They", " turn on the burner. ")
    ]
    assert results["results"]["hellaswag_likelihood"]["acc,none"] == 0
    assert not any(
        "Overwriting default num_fewshot" in record.getMessage()
        for record in caplog.records
    )


def test_gsm8k_zero_shot_evaluation_has_no_override_warning(task_manager, caplog):
    with caplog.at_level(logging.WARNING):
        results = lm_eval.simple_evaluate(
            model=DummyLM(),
            tasks=["gsm8k_likelihood"],
            task_manager=task_manager,
            num_fewshot=0,
            bootstrap_iters=0,
            log_samples=False,
        )
    assert results["configs"]["gsm8k_likelihood"]["num_fewshot"] == 0
    assert not any(
        "Overwriting default num_fewshot" in record.getMessage()
        for record in caplog.records
    )


def test_gsm8k_preserves_full_answer_and_original_spacing(task_manager):
    task = task_manager.load(["gsm8k_likelihood"])["tasks"]["gsm8k_likelihood"]
    assert _request(task, task.eval_docs[0]).args == (
        "Question: What is 1 + 1?\nAnswer:",
        "One plus one is two.\n#### 2",
    )


def test_c4_unpacks_loglikelihood_and_aggregates_perplexity(task_manager):
    task = task_manager.load(["c4_likelihood"])["tasks"]["c4_likelihood"]
    doc = task.eval_docs[0]
    assert _request(task, doc).args == ("", "café-noir .")
    scores = task.process_results(doc, [(-6.0, False)])
    byte_count = len(doc["text"].encode("utf-8"))
    assert scores == {
        "word_perplexity": (-6.0, 4),
        "byte_perplexity": (-6.0, byte_count),
        "bits_per_byte": (-6.0, byte_count),
    }
    aggregated = {
        name: aggregate([scores[name]])
        for name, aggregate in task.aggregation().items()
    }
    assert aggregated["word_perplexity"] == pytest.approx(math.exp(6.0 / 4))
    assert aggregated["byte_perplexity"] == pytest.approx(math.exp(6.0 / byte_count))
    assert aggregated["bits_per_byte"] == pytest.approx(
        6.0 / (byte_count * math.log(2))
    )
