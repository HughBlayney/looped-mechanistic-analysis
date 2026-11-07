"""Downstream evaluation at intermediate virtual layer depths.

Import BridgeLM from evaluation.lm_eval when using the lm-eval harness.
"""

from .coda_lens import coda_hidden_states, coda_lens
from .datasets import MCQExample, load_mcq_dataset
from .multiple_choice import evaluate_multiple_choice_by_depth

__all__ = [
    "MCQExample",
    "coda_hidden_states",
    "coda_lens",
    "evaluate_multiple_choice_by_depth",
    "load_mcq_dataset",
]
