"""Paired vector measurements and comparisons across activation histories.

Every vector comparison returns cosine similarity and the Euclidean norm of
the difference together. Attention patterns are matrices: they expose only a
per-head Frobenius difference norm. Histories are read one pair at a time;
these functions never stack or broadcast the full raw activation history.

``S`` counts virtual layers, ``L = max(layer_indices) + 1`` counts layer slots,
``T`` counts tokens, and ``H`` counts attention heads. Final references are the
last observed applications of each layer, without assuming convergence.
Results are float32, on the input device unless ``output_device`` is supplied.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

VectorComparisonMetric = Literal["cosine_sim", "diff_norm"]
ActivationHistory = Sequence[Tensor] | Tensor
OutputDevice = torch.device | str | None
COSINE_EPS = 1e-8


@dataclass(frozen=True, kw_only=True)
class VectorComparisonMetrics:
    """Both measurements of the same vector pairs, with identical shapes.

    ``cosine_sim`` uses a minimum norm of ``1e-8`` for each vector separately.
    Comparing a zero vector with anything, including itself, gives zero cosine
    similarity. ``diff_norm`` is ``||a-b||_2`` on the original, unnormalized
    vectors. Both tensors contain float32 values.
    """

    cosine_sim: Tensor
    diff_norm: Tensor

    @property
    def angular_diff_fraction(self) -> Tensor:
        """Angle divided by pi, derived from cosine similarity, in ``[0, 1]``.

        The zero-vector convention gives ``0.5`` when either vector is zero;
        this is a numerical convention, since a zero vector has no direction.
        """
        return self.cosine_sim.clamp(-1.0, 1.0).acos() / torch.pi


def compare_vectors(vector_a: Tensor, vector_b: Tensor) -> VectorComparisonMetrics:
    """Compare corresponding vectors along the last axis, with broadcasting.

    Inputs have shape ``(..., D)``. Outputs have their broadcast leading shape.
    Arithmetic uses float32 on the input device without modifying inputs.
    """
    vector_a, vector_b = vector_a.float(), vector_b.float()
    return VectorComparisonMetrics(
        cosine_sim=F.cosine_similarity(
            vector_a, vector_b, dim=-1, eps=COSINE_EPS
        ).clamp(-1.0, 1.0),
        diff_norm=torch.linalg.vector_norm(vector_a - vector_b, dim=-1),
    )


def compare_vector_sets(
    vectors_a: Tensor, vectors_b: Tensor
) -> VectorComparisonMetrics:
    """Compare every vector in ``(..., N, D)`` with every one in ``(..., M, D)``.

    Both outputs have shape ``(..., N, M)``. Matrix multiplication and direct
    Euclidean distance kernels avoid a temporary ``(..., N, M, D)`` tensor.
    Direct distance accumulation avoids cancellation for nearly equal vectors.
    """
    vectors_a, vectors_b = vectors_a.float(), vectors_b.float()
    cosine_sim = F.normalize(vectors_a, dim=-1, eps=COSINE_EPS) @ F.normalize(
        vectors_b, dim=-1, eps=COSINE_EPS
    ).transpose(-2, -1)
    diff_norm = torch.cdist(
        vectors_a, vectors_b, p=2, compute_mode="donot_use_mm_for_euclid_dist"
    )
    return VectorComparisonMetrics(
        cosine_sim=cosine_sim.clamp(-1.0, 1.0), diff_norm=diff_norm
    )


def compare_residuals(
    residual_a: Tensor, residual_b: Tensor
) -> VectorComparisonMetrics:
    """Per-token paired measurements for residuals shaped ``(..., T, D)``."""
    return compare_vectors(residual_a, residual_b)


def compute_attn_pattern_diff_norm(pattern_a: Tensor, pattern_b: Tensor) -> Tensor:
    """Per-head Frobenius norm over query and key axes of ``(..., H, Q, K)``.

    Patterns are compared without normalization by sequence length. Output
    shape is ``(..., H)`` and arithmetic is float32 on the input device.
    """
    return torch.linalg.vector_norm(pattern_a.float() - pattern_b.float(), dim=(-2, -1))


def _history(history: ActivationHistory, entry_ndim: int) -> Sequence[Tensor]:
    """Accept an existing sequence or a stack with an optional singleton batch."""
    if isinstance(history, Tensor):
        if history.ndim == entry_ndim + 2:
            if history.shape[1] != 1:
                raise ValueError("Comparison histories require batch size 1")
            history = history.squeeze(1)
        history = history.unbind(0)
    if not history:
        raise ValueError("At least one activation is required for comparisons")
    reference = history[0]
    if reference.ndim != entry_ndim or any(
        value.shape != reference.shape or value.device != reference.device
        for value in history
    ):
        raise ValueError(
            f"History entries must have {entry_ndim} dimensions, matching shapes "
            "and matching devices"
        )
    return history


def _indices_by_layer(layer_indices: Sequence[int], count: int) -> list[list[int]]:
    """Keep execution order within each layer, including unobserved layer slots."""
    if len(layer_indices) != count or any(layer < 0 for layer in layer_indices):
        raise ValueError("Provide one non-negative layer index per activation")
    indices_by_layer: list[list[int]] = [[] for _ in range(max(layer_indices) + 1)]
    for virtual_layer_index, layer_index in enumerate(layer_indices):
        indices_by_layer[layer_index].append(virtual_layer_index)
    return indices_by_layer


def compare_pairwise_residuals(
    residuals: ActivationHistory, *, output_device: OutputDevice = None
) -> VectorComparisonMetrics:
    """All virtual-layer pairs, including self-pairs; both outputs ``(S, S, T)``.

    Accepts a sequence of ``(T, D)`` tensors or a ``(S, T, D)`` /
    ``(S, 1, T, D)`` stack. Only one triangle is computed, then mirrored.
    """
    residuals = _history(residuals, 2)
    count, tokens = len(residuals), residuals[0].shape[0]
    device = output_device if output_device is not None else residuals[0].device
    pairwise_residual_cosine_sim = torch.empty(
        count, count, tokens, dtype=torch.float32, device=device
    )
    pairwise_residual_diff_norm = torch.empty_like(pairwise_residual_cosine_sim)
    for i in range(count):
        for j in range(i + 1):
            comparison = compare_residuals(residuals[i], residuals[j])
            pairwise_residual_cosine_sim[i, j] = comparison.cosine_sim
            pairwise_residual_diff_norm[i, j] = comparison.diff_norm
            pairwise_residual_cosine_sim[j, i] = pairwise_residual_cosine_sim[i, j]
            pairwise_residual_diff_norm[j, i] = pairwise_residual_diff_norm[i, j]
    return VectorComparisonMetrics(
        cosine_sim=pairwise_residual_cosine_sim, diff_norm=pairwise_residual_diff_norm
    )


def compute_pairwise_attn_pattern_diff_norm(
    attention_patterns: ActivationHistory, *, output_device: OutputDevice = None
) -> Tensor:
    """Symmetric ``(S, S, H)`` Frobenius differences for every virtual-layer pair.

    Accepts a sequence of ``(H, Q, K)`` tensors or a stack shaped
    ``(S, H, Q, K)`` / ``(S, 1, H, Q, K)``.
    """
    attention_patterns = _history(attention_patterns, 3)
    count, heads = len(attention_patterns), attention_patterns[0].shape[0]
    device = (
        output_device if output_device is not None else attention_patterns[0].device
    )
    pairwise_attn_pattern_diff_norm = torch.empty(
        count, count, heads, dtype=torch.float32, device=device
    )
    for i in range(count):
        for j in range(i + 1):
            pairwise_attn_pattern_diff_norm[i, j] = compute_attn_pattern_diff_norm(
                attention_patterns[i], attention_patterns[j]
            )
            pairwise_attn_pattern_diff_norm[j, i] = pairwise_attn_pattern_diff_norm[
                i, j
            ]
    return pairwise_attn_pattern_diff_norm


def compare_residuals_to_final_state(
    layer_indices: Sequence[int],
    residuals: ActivationHistory,
    *,
    output_device: OutputDevice = None,
) -> VectorComparisonMetrics:
    """Each virtual layer against each layer's final observation: ``(S, L, T)``.

    Unobserved layer slots contain ``NaN`` for both metrics. Input residuals
    follow execution order and use the same shapes as ``compare_pairwise_residuals``.
    """
    residuals = _history(residuals, 2)
    indices_by_layer = _indices_by_layer(layer_indices, len(residuals))
    device = output_device if output_device is not None else residuals[0].device
    to_final_state_residual_cosine_sim = torch.full(
        (len(residuals), len(indices_by_layer), residuals[0].shape[0]),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    to_final_state_residual_diff_norm = torch.full_like(
        to_final_state_residual_cosine_sim, float("nan")
    )
    for layer, indices in enumerate(indices_by_layer):
        if not indices:
            continue
        final_residual = residuals[indices[-1]]
        for i, residual in enumerate(residuals):
            comparison = compare_residuals(residual, final_residual)
            to_final_state_residual_cosine_sim[i, layer] = comparison.cosine_sim
            to_final_state_residual_diff_norm[i, layer] = comparison.diff_norm
    return VectorComparisonMetrics(
        cosine_sim=to_final_state_residual_cosine_sim,
        diff_norm=to_final_state_residual_diff_norm,
    )


def compute_to_final_state_attn_pattern_diff_norm(
    layer_indices: Sequence[int],
    attention_patterns: ActivationHistory,
    *,
    output_device: OutputDevice = None,
) -> Tensor:
    """Per-head differences to each layer's last pattern: ``(S, L, H)``.

    Unobserved layer slots contain ``NaN``. Patterns follow execution order.
    """
    attention_patterns = _history(attention_patterns, 3)
    indices_by_layer = _indices_by_layer(layer_indices, len(attention_patterns))
    device = (
        output_device if output_device is not None else attention_patterns[0].device
    )
    to_final_state_attn_pattern_diff_norm = torch.full(
        (
            len(attention_patterns),
            len(indices_by_layer),
            attention_patterns[0].shape[0],
        ),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    for layer, indices in enumerate(indices_by_layer):
        if not indices:
            continue
        final_attention_pattern = attention_patterns[indices[-1]]
        for i, attention_pattern in enumerate(attention_patterns):
            to_final_state_attn_pattern_diff_norm[i, layer] = (
                compute_attn_pattern_diff_norm(
                    attention_pattern, final_attention_pattern
                )
            )
    return to_final_state_attn_pattern_diff_norm


def compare_successive_recurrence_residuals(
    layer_indices: Sequence[int],
    residuals: ActivationHistory,
    *,
    output_device: OutputDevice = None,
) -> list[VectorComparisonMetrics]:
    """Paired measurements between consecutive applications of each layer.

    Returns one record per layer slot; each field is ``(max(R_l-1, 0), T)``.
    Layers with fewer than two observations have empty tensors. Pairs never
    cross between layers with different weights.
    """
    residuals = _history(residuals, 2)
    indices_by_layer = _indices_by_layer(layer_indices, len(residuals))
    device = output_device if output_device is not None else residuals[0].device
    comparisons = []
    for indices in indices_by_layer:
        successive_recurrence_residual_cosine_sim = torch.empty(
            max(len(indices) - 1, 0),
            residuals[0].shape[0],
            dtype=torch.float32,
            device=device,
        )
        successive_recurrence_residual_diff_norm = torch.empty_like(
            successive_recurrence_residual_cosine_sim
        )
        for recurrence, (i, j) in enumerate(pairwise(indices)):
            comparison = compare_residuals(residuals[i], residuals[j])
            successive_recurrence_residual_cosine_sim[recurrence] = (
                comparison.cosine_sim
            )
            successive_recurrence_residual_diff_norm[recurrence] = comparison.diff_norm
        comparisons.append(
            VectorComparisonMetrics(
                cosine_sim=successive_recurrence_residual_cosine_sim,
                diff_norm=successive_recurrence_residual_diff_norm,
            )
        )
    return comparisons


def compute_successive_recurrence_attn_pattern_diff_norm(
    layer_indices: Sequence[int],
    attention_patterns: ActivationHistory,
    *,
    output_device: OutputDevice = None,
) -> list[Tensor]:
    """One ``(max(R_l-1, 0), H)`` tensor per layer for successive recurrences.

    Layers with fewer than two observations have empty tensors.
    """
    attention_patterns = _history(attention_patterns, 3)
    indices_by_layer = _indices_by_layer(layer_indices, len(attention_patterns))
    device = (
        output_device if output_device is not None else attention_patterns[0].device
    )
    comparisons = []
    for indices in indices_by_layer:
        successive_recurrence_attn_pattern_diff_norm = torch.empty(
            max(len(indices) - 1, 0),
            attention_patterns[0].shape[0],
            dtype=torch.float32,
            device=device,
        )
        for recurrence, (i, j) in enumerate(pairwise(indices)):
            successive_recurrence_attn_pattern_diff_norm[recurrence] = (
                compute_attn_pattern_diff_norm(
                    attention_patterns[i], attention_patterns[j]
                )
            )
        comparisons.append(successive_recurrence_attn_pattern_diff_norm)
    return comparisons
