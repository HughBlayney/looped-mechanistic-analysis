# Copyright (c) 2025 Hugh Blayney. MIT license: ../licenses/transformer-tools.txt
"""Metrics of attention patterns: realised post-softmax attention probabilities."""

import logging
import math

import torch

logger = logging.getLogger(__name__)


def shannon_entropy(p_values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Shannon entropy of a (batch of) probability distribution(s).

    Uses ``xlogy`` for ``p·log(p)``, including its zero-probability limit,
    without a separate logarithm over the many zeros in causal attention.
    ``nansum`` ignores NaN contributions, including undefined probabilities
    from an all-zero residual spectrum.

    Args:
        p_values: Probability values.  Should sum to 1 along ``dim``, though
            this is not enforced.
        dim: Dimension along which to sum.

    Returns:
        Entropy with ``dim`` reduced away.
    """
    p_values = p_values.float()
    return -torch.nansum(torch.special.xlogy(p_values, p_values), dim=dim)


def get_sink_scores(
    attention_patterns: torch.Tensor,
    key_index: int = 0,
) -> torch.Tensor:
    """Average attention given to a specific key position ("sink score").

    An *attention sink* is a position that receives disproportionately high
    attention from many query positions — most commonly the BOS token at
    ``key_index=0``.

    Args:
        attention_patterns: ``(B, H, T_q, T_k)`` — attention probabilities from
            ``cache["pattern", layer]``.
        key_index: Key position to score (default ``0`` = first/BOS token).

    Returns:
        ``(B, H)`` — mean attention probability given to ``key_index`` across all
        query positions.
    """
    attention_patterns = attention_patterns.float()
    T_q = attention_patterns.shape[-2]
    # attention_patterns[..., key_index] → (B, H, T_q)
    # .sum(dim=-1) → (B, H) — total attention sent to key_index
    return attention_patterns[..., key_index].sum(dim=-1) / float(T_q)


def get_mean_sink_rate(
    attention_patterns: torch.Tensor,
    key_index: int = 0,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Fraction of attention heads whose sink score exceeds ``threshold``.

    Args:
        attention_patterns: ``(B, H, T_q, T_k)``.
        key_index: Key position to score (default ``0``).
        threshold: Minimum sink score to classify a head as a "sink head".

    Returns:
        ``(B,)`` — fraction of heads in ``[0, 1]``.
    """
    attention_patterns = attention_patterns.float()
    sink_scores = get_sink_scores(attention_patterns, key_index=key_index)  # (B, H)
    return (sink_scores >= threshold).float().mean(dim=-1)


def get_legacy_mixing_scores(
    attention_patterns: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Average row entropy of attention patterns, normalised by ``log(T_k)``.

    A high mixing score indicates that heads spread attention broadly across
    positions; a low mixing score indicates focused or sink-like attention.

    Args:
        attention_patterns: ``(B, H, T_q, T_k)`` — attention probabilities.
        normalize: If ``True``, divide each row entropy by ``log(T_k)`` before
            averaging, so the result lies in ``[0, 1]``.

    Returns:
        ``(B, H)`` — mean row entropy across all query positions.
        Range is ``[0, log(T_k)]`` when ``normalize=False``, ``[0, 1]`` when
        ``normalize=True``.

    Note:
        For causal attention the global ``log(T_k)`` normaliser exceeds the
        per-row maximum for early positions, so the achievable ceiling is
        approximately ``1 - 1/log(T)``.  See ``get_mixing_scores`` for a
        per-row normalisation that avoids this.
    """
    attention_patterns = attention_patterns.float()
    T_k = attention_patterns.shape[-1]
    row_entropies = shannon_entropy(attention_patterns, dim=-1)  # (B, H, T_q)
    if normalize:
        row_entropies = row_entropies / math.log(T_k)
    return row_entropies.mean(dim=-1)


def get_mixing_scores(attention_patterns: torch.Tensor) -> torch.Tensor:
    """Per-row-normalised average entropy of the attention pattern.

    Each row's entropy is normalised by ``log(i+1)``, the maximum entropy
    achievable at query position ``i`` under a causal mask.  Row 0 is excluded:
    it can only attend to itself, making its normalised entropy 0/0 (undefined).

    Args:
        attention_patterns: ``(B, H, T_q, T_k)`` — attention probabilities.

    Returns:
        ``(B, H)`` — mean normalised row entropy over rows ``1..T_q-1``,
        in ``[0, 1]``.
    """
    attention_patterns = attention_patterns.float()
    T_q = attention_patterns.shape[-2]
    row_entropies = shannon_entropy(
        attention_patterns[..., 1:, :], dim=-1
    )  # (B, H, T_q-1)
    log_norms = torch.tensor(
        [math.log(i + 1) for i in range(1, T_q)],
        dtype=torch.float32,
        device=attention_patterns.device,
    )
    return (row_entropies / log_norms).mean(dim=-1)


def get_colsum_concentrations(attention_patterns: torch.Tensor) -> torch.Tensor:
    """How unevenly attention is *received* across key positions.

    Computes ``1 − H_norm(col_sums)`` where ``H_norm`` is the entropy of the
    normalised column sums, normalised to ``[0, 1]`` by dividing by
    ``log(T_k)``.

    * Value ``1`` → all attention flows to a single key position (perfect
      concentration / hard sink).
    * Value ``0`` → every key position receives equal total attention.

    Args:
        attention_patterns: ``(B, H, T_q, T_k)`` — attention probabilities.

    Returns:
        ``(B, H)`` — concentration in ``[0, 1]``.
    """
    attention_patterns = attention_patterns.float()
    T_q = attention_patterns.shape[-2]
    T_k = attention_patterns.shape[-1]

    # Normalised column sums → (B, H, T_k)
    normalised_col_sums = attention_patterns.sum(dim=-2) / float(T_q)
    # Entropy of column distribution → (B, H)
    colsum_entropy = shannon_entropy(normalised_col_sums, dim=-1)
    # Normalise to [0, 1]
    max_entropy = torch.log(torch.tensor(T_k, dtype=torch.float32))
    return 1.0 - colsum_entropy / max_entropy
