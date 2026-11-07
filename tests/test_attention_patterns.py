"""Tests ported from the legacy BlockData class to the functional API.

These tests verify the same mathematical properties as the original
class-based tests, now using the pure functions from
``looped_internals.analysis``.
"""

import torch

from looped_internals.analysis import (
    get_colsum_concentrations,
    get_mixing_scores,
    get_sink_scores,
)

SEQUENCE_LENGTH = 10


def test_bos_sink():
    """When all attention flows to token 0, sink score should be 1, mixing 0, concentration 1."""
    # (batch=1, heads=1, T, T)
    attention_patterns = torch.zeros(1, 1, SEQUENCE_LENGTH, SEQUENCE_LENGTH)
    attention_patterns[:, :, :, 0] = 1.0

    sink_scores = get_sink_scores(attention_patterns)  # (1, 1)
    mixing_scores = get_mixing_scores(attention_patterns)  # (1, 1)
    colsum_conc = get_colsum_concentrations(attention_patterns)  # (1, 1)

    assert torch.isclose(sink_scores[0, 0], torch.tensor(1.0)), (
        f"BOS sink score should be 1.0, got {sink_scores[0, 0]}"
    )
    assert torch.isclose(mixing_scores[0, 0], torch.tensor(0.0)), (
        f"BOS mixing score should be 0.0, got {mixing_scores[0, 0]}"
    )
    assert torch.isclose(colsum_conc[0, 0], torch.tensor(1.0)), (
        f"BOS colsum concentration should be 1.0, got {colsum_conc[0, 0]}"
    )


def test_split_sink():
    """When attention is split between two sink tokens, scores should reflect this."""
    attention_patterns = torch.zeros(1, 1, SEQUENCE_LENGTH, SEQUENCE_LENGTH)
    # First half attends to token 0; second half attends to middle token
    attention_patterns[:, :, : SEQUENCE_LENGTH // 2, 0] = 1.0
    attention_patterns[:, :, SEQUENCE_LENGTH // 2 :, SEQUENCE_LENGTH // 2] = 1.0

    sink_scores = get_sink_scores(attention_patterns)
    mixing_scores = get_mixing_scores(attention_patterns)
    colsum_conc = get_colsum_concentrations(attention_patterns)

    assert not torch.isclose(sink_scores[0, 0], torch.tensor(1.0)), (
        "Split-sink score should not be 1.0"
    )
    assert torch.isclose(mixing_scores[0, 0], torch.tensor(0.0)), (
        "Split-sink mixing score should still be 0.0 (no entropy per row)"
    )
    assert not torch.isclose(colsum_conc[0, 0], torch.tensor(1.0)), (
        "Split-sink colsum concentration should not be 1.0"
    )
