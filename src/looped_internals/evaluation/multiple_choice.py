"""Compare logit-lens and trained-probe accuracy across virtual layer depth."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch

from looped_internals._random import restore_rng, rng_state
from looped_internals.results import DepthAccuracy, DepthEvaluationResult
from looped_internals.structure import ModelLayerStructure

from .datasets import MCQExample
from .logit_lens import score_logitlens
from .probes import MultipleChoiceProbe, score_probes, train_probes

logger = logging.getLogger(__name__)


def evaluate_multiple_choice_by_depth(
    bridge,
    tokenizer,
    structure: ModelLayerStructure,
    train_examples: Sequence[MCQExample],
    test_examples: Sequence[MCQExample],
    n_recurrences: int,
    n_classes: int,
    device: torch.device,
    model_name: str,
    dataset_name: str,
    batch_size: int = 8,
    forward_kwargs: dict | None = None,
    probes: list[MultipleChoiceProbe] | None = None,
    compute_coda: bool = False,
    project_pre_coda: bool = False,
    resume: dict | None = None,
    checkpoint=None,
) -> tuple[DepthEvaluationResult, list[MultipleChoiceProbe]]:
    """Evaluate multiple-choice accuracy at each virtual layer.

    1. Enumerates virtual layers.
    2. Trains probes (or uses supplied ``probes`` to skip training).
    3. Evaluates LogitLens + probes on the test split.

    Args:
        bridge: TransformerBridge.
        tokenizer: Model tokenizer.
        structure: Block layout of the model.
        train_examples: Training MCQ examples.
        test_examples: Test MCQ examples.
        n_recurrences: Recurrences for looped models.
        n_classes: Number of MCQ answer classes.
        device: Device for inference.
        model_name: Model identifier, stored in result.
        dataset_name: Dataset name, stored in result.
        batch_size: Forward-pass batch size for probe training.
        forward_kwargs: Extra kwargs for bridge.forward (e.g. ``{"num_steps": 8}``).
        probes: Pre-trained probes.  If provided, training is skipped.
        compute_coda: Also score the coda lens on prelude/core states, sharing
            the ordinary logit lens's forward pass. Requires a model with a coda.
        project_pre_coda: Also score Parcae's logit lens after projecting
            prelude and core states through C, sharing the same forward pass.

    Returns:
        ``(DepthEvaluationResult, probes)`` — accuracies and trained probes,
        ready for plotting or saving.
    """
    from tqdm.auto import tqdm

    forward_kwargs = forward_kwargs or {}
    virtual_layers = list(structure.virtual_layers(n_recurrences))
    n_virtual_layers = len(virtual_layers)
    logger.info("Model has %d virtual layers.", n_virtual_layers)

    if probes is None:
        probes = train_probes(
            bridge=bridge,
            tokenizer=tokenizer,
            train_examples=train_examples,
            structure=structure,
            n_recurrences=n_recurrences,
            n_classes=n_classes,
            device=device,
            batch_size=batch_size,
            forward_kwargs=forward_kwargs,
        )

    # ── Evaluate on test split ─────────────────────────────────────────────
    ll_correct = np.zeros(n_virtual_layers, dtype=np.int64)
    ll_norm_correct = np.zeros(n_virtual_layers, dtype=np.int64)
    probe_correct = np.zeros(n_virtual_layers, dtype=np.int64)
    coda_correct = np.zeros(n_virtual_layers) if compute_coda else None
    projected_correct = (
        np.zeros(n_virtual_layers, dtype=np.int64) if project_pre_coda else None
    )
    n_test = len(test_examples)

    start = 0
    if resume is not None:
        start = resume["next_example"]
        (
            ll_correct,
            ll_norm_correct,
            probe_correct,
            coda_correct,
            projected_correct,
        ) = resume["counts"]
        restore_rng(resume["rng"], device)

    for index in tqdm(range(start, n_test), desc="Evaluating test examples"):
        ex = test_examples[index]
        ll_pred, ll_norm_pred, coda_pred, projected_pred = score_logitlens(
            bridge,
            tokenizer,
            ex,
            structure,
            n_recurrences,
            device,
            forward_kwargs,
            compute_coda=compute_coda,
            project_pre_coda=project_pre_coda,
        )
        ll_correct += ll_pred == ex.label
        ll_norm_correct += ll_norm_pred == ex.label
        if compute_coda:
            coda_correct += np.where(np.isnan(coda_pred), np.nan, coda_pred == ex.label)
        if project_pre_coda:
            projected_correct += projected_pred == ex.label

        pr_pred = score_probes(
            probes,
            bridge,
            tokenizer,
            ex,
            structure,
            n_recurrences,
            device,
            forward_kwargs,
        )
        probe_correct += pr_pred == ex.label
        if checkpoint is not None:
            checkpoint(
                {
                    "next_example": index + 1,
                    "counts": (
                        ll_correct,
                        ll_norm_correct,
                        probe_correct,
                        coda_correct,
                        projected_correct,
                    ),
                    "rng": rng_state(device),
                }
            )

    logitlens_acc = ll_correct / n_test
    logitlens_norm_acc = ll_norm_correct / n_test
    probe_acc = probe_correct / n_test

    result = DepthEvaluationResult(
        per_virtual_layer_metrics=[
            DepthAccuracy(
                layer_index=layer.layer_index,
                virtual_layer_index=layer.virtual_layer_index,
                logitlens_accuracy=torch.tensor(
                    [logitlens_acc[i]], dtype=torch.float32
                ),
                logitlens_norm_accuracy=torch.tensor(
                    [logitlens_norm_acc[i]], dtype=torch.float32
                ),
                probe_accuracy=torch.tensor([probe_acc[i]], dtype=torch.float32),
                coda_accuracy=(
                    torch.tensor([coda_correct[i] / n_test], dtype=torch.float32)
                    if compute_coda
                    else None
                ),
                projected_logitlens_accuracy=(
                    torch.tensor([projected_correct[i] / n_test], dtype=torch.float32)
                    if project_pre_coda
                    else None
                ),
            )
            for i, layer in enumerate(virtual_layers)
        ],
        model_name=model_name,
        dataset_name=dataset_name,
        n_test=n_test,
        n_train=len(train_examples),
    )
    return result, probes
