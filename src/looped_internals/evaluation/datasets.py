"""Multiple-choice examples and benchmark loading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCQExample:
    """A multiple-choice question example.

    Attributes:
        context: Question / sentence template text.
        choices: List of answer texts.  For WinoGrande these are fully-filled
            sentences; for ARC/HellaSwag they are answer-only strings.
        label: Index of the correct choice (0-based).
        full_sentence_choices: If True, each choice is a *complete* sentence to
            be scored from BOS (WinoGrande).  If False, the choice is appended
            after the context (ARC/HellaSwag).
    """

    context: str
    choices: list[str]
    label: int
    full_sentence_choices: bool = False


def load_mcq_dataset(
    dataset_name: str,
    split: str,
    max_examples: int = 0,
) -> list[MCQExample]:
    """Load an MCQ benchmark as ``MCQExample`` objects.

    Args:
        dataset_name: One of ``"arc_easy"``, ``"arc_challenge"``,
            ``"hellaswag"``, ``"winogrande"``.
        split: ``"train"``, ``"validation"``, or ``"test"``.
        max_examples: If > 0, randomly subsample to this many examples.

    Returns:
        List of :class:`MCQExample`.
    """
    from datasets import load_dataset

    examples: list[MCQExample] = []

    if dataset_name in ("arc_easy", "arc_challenge"):
        config = "ARC-Easy" if dataset_name == "arc_easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", config, split=split)
        for doc in ds:
            labels = doc["choices"]["label"]
            texts = doc["choices"]["text"]
            answer = doc["answerKey"]
            if answer not in labels:
                continue
            # Skip uncommon examples with ≠ 4 choices to keep n_classes consistent
            if len(texts) != 4:
                continue
            examples.append(
                MCQExample(
                    context=doc["question"],
                    choices=texts,
                    label=labels.index(answer),
                    full_sentence_choices=False,
                )
            )

    elif dataset_name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split=split)
        for doc in ds:
            examples.append(
                MCQExample(
                    context=doc["ctx"],
                    choices=list(doc["endings"]),
                    label=int(doc["label"]),
                    full_sentence_choices=False,
                )
            )

    elif dataset_name == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split=split)
        for doc in ds:
            sentence = doc["sentence"]
            filled = [
                sentence.replace("_", doc["option1"]),
                sentence.replace("_", doc["option2"]),
            ]
            examples.append(
                MCQExample(
                    context=sentence,
                    choices=filled,
                    label=int(doc["answer"]) - 1,
                    full_sentence_choices=True,
                )
            )

    else:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            "Choose from: arc_easy, arc_challenge, hellaswag, winogrande."
        )

    if max_examples > 0 and len(examples) > max_examples:
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(examples), max_examples, replace=False)
        examples = [examples[i] for i in sorted(idxs)]

    return examples
