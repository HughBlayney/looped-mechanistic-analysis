"""Extract finite-rollout final reference activations by layer.

Final observed states need not be mathematical fixed points.
"""

# Copyright (c) 2025 Hugh Blayney. MIT license: ../licenses/transformer-tools.txt
import logging

import torch

logger = logging.getLogger(__name__)


def _final_state_by_layer(
    layer_indices: list[int],
    tensors: list[torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Return the *last* tensor seen for each layer index.

    For recurrent models the same layer index can appear multiple times.
    The reference is the activation from the final recurrence of that layer.

    Args:
        layer_indices: Layer index per entry in ``tensors``.
        tensors: Per-virtual-layer tensors (arbitrary shape).

    Returns:
        Mapping ``{layer_idx: tensor}`` — last occurrence wins.
    """
    fixed_point: dict[int, torch.Tensor] = {}
    for layer_idx, tensor in zip(layer_indices, tensors, strict=True):
        fixed_point[layer_idx] = tensor  # later occurrences overwrite
    return fixed_point


def get_final_residuals(
    layer_indices: list[int],
    residual_streams: list[torch.Tensor],
) -> torch.Tensor:
    """Final observed residual stream for each unique layer index.

    Args:
        layer_indices: Layer index for each entry in ``residual_streams``.
        residual_streams: Per-virtual-layer residual streams, each ``(T, D)`` (batch
            dimension already removed).

    Returns:
        ``(max_layer + 1, T, D)`` — final-reference residual for each layer index.
        Layers not present in ``layer_indices`` are left as zeros.
    """
    logger.debug(
        "get_final_residuals: %d virtual layers, %d unique layers",
        len(layer_indices),
        len(set(layer_indices)),
    )
    fp_map = _final_state_by_layer(layer_indices, residual_streams)
    max_layer = max(layer_indices)
    ref = residual_streams[0]
    result = torch.zeros(
        max_layer + 1, *ref.shape, dtype=torch.float32, device=ref.device
    )
    for idx, tensor in fp_map.items():
        result[idx] = tensor.float()
    return result


def get_final_attention_patterns(
    layer_indices: list[int],
    attention_patterns: list[torch.Tensor],
) -> torch.Tensor:
    """Final observed attention patterns for each unique layer index.

    Args:
        layer_indices: Layer index for each entry in ``attention_patterns``.
        attention_patterns: Per-virtual-layer attention patterns, each ``(H, T, T)``
            (batch dimension already removed).

    Returns:
        ``(max_layer + 1, H, T, T)`` — final-reference attention patterns.
    """
    fp_map = _final_state_by_layer(layer_indices, attention_patterns)
    max_layer = max(layer_indices)
    ref = attention_patterns[0]
    result = torch.zeros(
        max_layer + 1, *ref.shape, dtype=torch.float32, device=ref.device
    )
    for idx, tensor in fp_map.items():
        result[idx] = tensor.float()
    return result
