"""Multiple-choice probe inputs, training, scoring, and persistence."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from looped_internals._random import restore_rng, rng_state
from looped_internals.structure import ModelLayerStructure

from .datasets import MCQExample

logger = logging.getLogger(__name__)

PROBE_CACHE_VERSION = 3
PROBE_KIND = "choice_mean_pool_ln_scalar"


class MultipleChoiceProbe(nn.Module):
    """LayerNorm plus scalar head applied to one pooled option representation."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.score = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.score(self.norm(h)).squeeze(-1)


def _option_prompt(example: MCQExample, choice: str) -> str:
    """Return the full prompt used for one MCQ probe option."""
    if example.full_sentence_choices:
        return choice
    if not example.context:
        return choice
    return f"{example.context} {choice}"


def _build_probe_batch(
    tokenizer,
    examples: Sequence[MCQExample],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build a flattened ``(batch * choices, seq)`` probe input batch."""
    sequences: list[list[int]] = []
    n_choices = len(examples[0].choices)
    for ex in examples:
        if len(ex.choices) != n_choices:
            raise ValueError("All examples in a probe batch must have the same choices")
        for choice in ex.choices:
            ids = tokenizer.encode(_option_prompt(ex, choice), add_special_tokens=True)
            sequences.append(ids)

    max_len = max(len(ids) for ids in sequences)
    input_ids = torch.full(
        (len(sequences), max_len), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (len(sequences), max_len), dtype=torch.bool, device=device
    )
    for i, ids in enumerate(sequences):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, : len(ids)] = True

    return input_ids, attention_mask, n_choices


def _mean_pool_tokens(h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool hidden states over non-padding tokens."""
    mask = attention_mask.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (h * mask).sum(dim=1) / denom


def train_probes(
    bridge,
    tokenizer,
    train_examples: Sequence[MCQExample],
    structure: ModelLayerStructure,
    n_recurrences: int,
    n_classes: int,
    device: torch.device,
    batch_size: int = 8,
    lr: float = 5e-4,
    forward_kwargs: dict | None = None,
    resume: dict | None = None,
    checkpoint=None,
) -> list[MultipleChoiceProbe]:
    """Train one scalar MCQ probe per virtual layer on the training split.

    Each answer option is tokenized as its own ``question + " " + option``
    sequence. For each virtual layer, token representations are mean-pooled
    over non-padding tokens, scored with a non-affine LayerNorm plus scalar
    linear head, and trained with cross-entropy over the option scores.

    Args:
        bridge: TransformerBridge.
        tokenizer: Model tokenizer.
        train_examples: Training MCQ examples.
        structure: Model layer groups and hook names.
        n_recurrences: Number of applications of each recurrent layer.
        n_classes: Number of MCQ answer classes.
        device: Inference device.
        batch_size: Forward-pass batch size.
        lr: Adam learning rate for probes.
        forward_kwargs: Extra kwargs for bridge.forward.

    Returns:
        List of trained probes (one per virtual layer).
    """
    forward_kwargs = forward_kwargs or {}
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0
    virtual_layers = list(structure.virtual_layers(n_recurrences))

    d_model = bridge.cfg.d_model
    n_virtual_layers = len(virtual_layers)

    probes = [MultipleChoiceProbe(d_model).to(device) for _ in range(n_virtual_layers)]
    optimizer = Adam([p for probe in probes for p in probe.parameters()], lr=lr)

    n_examples = len(train_examples)
    logger.info("Training %d probes on %d examples...", n_virtual_layers, n_examples)

    start = 0
    if resume is not None:
        for probe, state in zip(probes, resume["probes"], strict=True):
            probe.load_state_dict(state)
        optimizer.load_state_dict(resume["optimizer"])
        start = resume["next_batch"]
        restore_rng(resume["rng"], device)

    for batch_start in range(start, n_examples, batch_size):
        batch = train_examples[batch_start : batch_start + batch_size]
        labels_t = torch.tensor(
            [ex.label for ex in batch], dtype=torch.long, device=device
        )

        input_ids, attention_mask, batch_n_choices = _build_probe_batch(
            tokenizer, batch, pad_id, device
        )
        if batch_n_choices != n_classes:
            raise ValueError(
                f"Expected {n_classes} choices, got {batch_n_choices} in batch"
            )

        with torch.no_grad():
            _, cache = bridge.run_with_cache(
                input_ids,
                accumulate=True,
                names_filter=structure.hook_names(),
                device="cpu",
                **forward_kwargs,
            )

        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        for k, layer in enumerate(virtual_layers):
            h = (
                cache[layer.residual_key].to(device).float()
            )  # (B * choices, T, d_model)
            pooled = _mean_pool_tokens(h, attention_mask)
            logits = probes[k](pooled.detach()).view(len(batch), batch_n_choices)
            loss = F.cross_entropy(logits, labels_t)
            total_loss = total_loss + loss
        total_loss.backward()
        optimizer.step()
        if checkpoint is not None:
            checkpoint(
                {
                    "probes": [p.state_dict() for p in probes],
                    "optimizer": optimizer.state_dict(),
                    "next_batch": batch_start + len(batch),
                    "rng": rng_state(device),
                }
            )

    # Move probes to CPU to free GPU memory after training
    for probe in probes:
        probe.cpu().eval()

    return probes


@torch.no_grad()
def score_probes(
    probes: list[MultipleChoiceProbe],
    bridge,
    tokenizer,
    example: MCQExample,
    structure: ModelLayerStructure,
    n_recurrences: int,
    device: torch.device,
    forward_kwargs: dict | None = None,
) -> np.ndarray:
    """Evaluate probes on a single example.

    Returns:
        ``predicted`` int array of shape ``(n_virtual_layers,)``.
    """
    forward_kwargs = forward_kwargs or {}
    virtual_layers = list(structure.virtual_layers(n_recurrences))
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0
    input_ids, attention_mask, n_choices = _build_probe_batch(
        tokenizer, [example], pad_id, device
    )

    _, cache = bridge.run_with_cache(
        input_ids,
        accumulate=True,
        names_filter=structure.hook_names(),
        device="cpu",
        **forward_kwargs,
    )

    n_virtual_layers = len(virtual_layers)
    predicted = np.empty(n_virtual_layers, dtype=np.int32)

    for k, layer in enumerate(virtual_layers):
        h = cache[layer.residual_key].to(device).float()  # (choices, T, d_model)
        pooled = _mean_pool_tokens(h, attention_mask)
        logits = probes[k].to(device)(pooled).view(1, n_choices)
        predicted[k] = int(logits.argmax(dim=-1).item())

    return predicted


def load_probes(
    path: str | Path,
    d_model: int,
    n_classes: int,
    n_virtual_layers: int,
    device: torch.device,
) -> list[MultipleChoiceProbe]:
    """Load trained probe state dicts and validate the cache metadata."""
    raw = torch.load(path, weights_only=True)
    if isinstance(raw, list):
        raise ValueError("legacy probe cache without metadata")

    metadata = raw.get("metadata", {})
    expected = {
        "version": PROBE_CACHE_VERSION,
        "kind": PROBE_KIND,
        "d_model": d_model,
        "n_classes": n_classes,
        "n_virtual_layers": n_virtual_layers,
    }
    mismatches = {
        k: (metadata.get(k), v) for k, v in expected.items() if metadata.get(k) != v
    }
    if mismatches:
        raise ValueError(f"incompatible probe cache metadata: {mismatches}")

    state_dicts = raw["state_dicts"]
    probes = []
    for sd in state_dicts:
        probe = MultipleChoiceProbe(d_model)
        probe.load_state_dict(sd)
        probe.eval()
        probes.append(probe)
    return probes


def save_probes(
    path: str | Path,
    probes: list[MultipleChoiceProbe],
    d_model: int,
    n_classes: int,
    n_virtual_layers: int,
) -> None:
    """Save trained probes with enough metadata to avoid stale reuse."""
    torch.save(
        {
            "metadata": {
                "version": PROBE_CACHE_VERSION,
                "kind": PROBE_KIND,
                "d_model": d_model,
                "n_classes": n_classes,
                "n_virtual_layers": n_virtual_layers,
            },
            "state_dicts": [p.state_dict() for p in probes],
        },
        path,
    )
