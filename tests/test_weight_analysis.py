"""Tests for looped_internals.analysis.neurons.

All tests run on CPU with small synthetic tensors — no model loading required.

Shape conventions:
    L       = number of layers
    d_mlp   = MLP hidden dimension (neurons per layer)
    d_model = model / residual-stream dimension
    d_vocab = vocabulary size
"""

import pytest
import torch

from looped_internals.analysis.neurons import (
    center_columns,
    center_rows,
    classify_neurons,
    compare_neuron_vocab_vectors,
    neuron_vocab_cosine_moments,
    prediction_suppression_fractions,
)

# ---------------------------------------------------------------------------
# Small fixed dimensions for most tests
# ---------------------------------------------------------------------------

L, d_mlp, d_model, d_vocab = 3, 8, 16, 64

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# center_columns / center_rows
# ---------------------------------------------------------------------------


def test_center_columns_zero_mean():
    W = torch.randn(L, d_mlp, d_vocab)
    W_c = center_columns(W)
    assert W_c.shape == W.shape
    assert W_c.mean(dim=-1).abs().max().item() < 1e-5


def test_center_rows_zero_mean():
    W = torch.randn(L, d_mlp, d_model)
    W_c = center_rows(W)
    assert W_c.shape == W.shape
    assert W_c.mean(dim=-2).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# compare_neuron_vocab_vectors
# ---------------------------------------------------------------------------


def test_compare_neuron_vocab_vectors_shape():
    W_out = torch.randn(L, d_mlp, d_model)
    W_U = torch.randn(d_model, d_vocab)
    cosines = compare_neuron_vocab_vectors(W_out, W_U).cosine_sim
    assert cosines.shape == (L, d_mlp, d_vocab)


def test_compare_neuron_vocab_vectors_range():
    """All cosine values must lie in [-1, 1]."""
    W_out = torch.randn(L, d_mlp, d_model)
    W_U = torch.randn(d_model, d_vocab)
    cosines = compare_neuron_vocab_vectors(W_out, W_U).cosine_sim
    assert cosines.min().item() >= -1.0 - 1e-5
    assert cosines.max().item() <= 1.0 + 1e-5


def test_compare_neuron_vocab_vectors_perfect_alignment():
    """If a neuron's output weight equals a token's unembedding direction, cosine = 1."""
    d_m, d_v = 8, 10
    W_U = torch.eye(d_m, d_v)  # first d_m tokens aligned to basis vectors
    W_out = torch.zeros(1, d_v, d_m)  # 1 layer, d_v neurons

    # Make neuron k point in same direction as token k (for k < d_m)
    for k in range(d_m):
        W_out[0, k, k] = 1.0

    cosines = compare_neuron_vocab_vectors(
        W_out, W_U
    ).cosine_sim  # (1, d_v, d_v) when d_m == d_v
    # Each neuron k should have cosine ≈ 1 with its matching token
    for k in range(d_m):
        assert cosines[0, k, k].item() == pytest.approx(1.0, abs=1e-5)


def test_compare_neuron_vocab_vectors_anti_alignment():
    """Neuron pointing opposite to a token direction → cosine = -1."""
    d_m, d_v = 4, 6
    W_U = torch.zeros(d_m, d_v)
    W_U[0, 0] = 1.0  # token 0 points along dim 0
    W_out = torch.zeros(1, 1, d_m)
    W_out[0, 0, 0] = -2.0  # neuron 0 points opposite

    cosines = compare_neuron_vocab_vectors(W_out, W_U).cosine_sim
    assert abs(cosines[0, 0, 0].item() - (-1.0)) < 1e-5


def test_compare_neuron_vocab_vectors_accepts_bfloat16():
    """Function should handle non-float32 input without error."""
    W_out = torch.randn(L, d_mlp, d_model, dtype=torch.bfloat16)
    W_U = torch.randn(d_model, d_vocab, dtype=torch.bfloat16)
    cosines = compare_neuron_vocab_vectors(W_out, W_U).cosine_sim
    assert cosines.dtype == torch.float32


# ---------------------------------------------------------------------------
# neuron_vocab_cosine_moments
# ---------------------------------------------------------------------------


def test_neuron_vocab_cosine_moments_shape():
    cosines = torch.randn(L, d_mlp, d_vocab)
    moments = neuron_vocab_cosine_moments(cosines)
    for key in ("mean", "variance", "skew", "kurtosis"):
        assert key in moments
        assert moments[key].shape == (L, d_mlp), f"wrong shape for {key}"


def test_neuron_vocab_cosine_moments_uniform_distribution():
    """Uniform distribution: skew = 0, excess kurtosis = -1.2 (uniform on [a,b])."""
    # Uniform on [-1, 1]: excess kurtosis = -6/5 = -1.2, skew = 0
    cosines = (
        torch.linspace(-1, 1, d_vocab).unsqueeze(0).unsqueeze(0).expand(L, d_mlp, -1)
    )
    moments = neuron_vocab_cosine_moments(cosines)
    assert moments["skew"].abs().max().item() < 1e-4
    # excess kurtosis of discrete uniform on n points ≈ -6/5 * (n²-1)/(n²+1)... just check negative
    assert moments["kurtosis"].max().item() < 0


def test_neuron_vocab_cosine_moments_gaussian():
    """Large sample from Gaussian: skew ≈ 0, excess kurtosis ≈ 0."""
    torch.manual_seed(0)
    cosines = torch.randn(1, 1, 100_000)
    moments = neuron_vocab_cosine_moments(cosines)
    assert abs(moments["skew"][0, 0].item()) < 0.05
    assert abs(moments["kurtosis"][0, 0].item()) < 0.1


def test_neuron_vocab_cosine_moments_known_skew():
    """Hand-crafted distribution with known positive skew."""
    # Distribution heavily skewed right: mostly -0.1 with a few large positives
    base = torch.full((1, 1, 90), -0.1)
    spike = torch.full((1, 1, 10), 5.0)
    cosines = torch.cat([base, spike], dim=-1)
    moments = neuron_vocab_cosine_moments(cosines)
    assert moments["skew"][0, 0].item() > 0.0


def test_neuron_vocab_cosine_moments_variance_positive():
    cosines = torch.randn(L, d_mlp, d_vocab)
    moments = neuron_vocab_cosine_moments(cosines)
    assert (moments["variance"] >= 0).all()


# ---------------------------------------------------------------------------
# classify_neurons
# ---------------------------------------------------------------------------


def _make_labels(L: int = 2, d_mlp: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (kurtosis, skew) tensors with known structure for testing."""
    kurtosis = torch.zeros(L, d_mlp)
    skew = torch.zeros(L, d_mlp)
    # neurons 0,1: high kurtosis + positive skew → prediction
    kurtosis[:, 0:2] = 15.0
    skew[:, 0:2] = 1.0
    # neurons 2,3: high kurtosis + negative skew → suppression
    kurtosis[:, 2:4] = 15.0
    skew[:, 2:4] = -1.0
    # neurons 4,5: low kurtosis → other
    kurtosis[:, 4:6] = 2.0
    return kurtosis, skew


def test_classify_neurons_shape():
    kurtosis, skew = _make_labels()
    labels = classify_neurons(kurtosis, skew)
    assert labels.shape == kurtosis.shape


def test_classify_neurons_prediction():
    kurtosis, skew = _make_labels()
    labels = classify_neurons(kurtosis, skew, kurtosis_threshold=10.0)
    assert (labels[:, 0:2] == 1).all()


def test_classify_neurons_suppression():
    kurtosis, skew = _make_labels()
    labels = classify_neurons(kurtosis, skew, kurtosis_threshold=10.0)
    assert (labels[:, 2:4] == -1).all()


def test_classify_neurons_other():
    kurtosis, skew = _make_labels()
    labels = classify_neurons(kurtosis, skew, kurtosis_threshold=10.0)
    assert (labels[:, 4:6] == 0).all()


def test_classify_neurons_custom_threshold():
    kurtosis = torch.full((1, 4), 5.0)
    skew = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    # With threshold=10 nothing is classified
    labels_high = classify_neurons(kurtosis, skew, kurtosis_threshold=10.0)
    assert (labels_high == 0).all()
    # With threshold=4 everything is classified
    labels_low = classify_neurons(kurtosis, skew, kurtosis_threshold=4.0)
    assert (labels_low != 0).all()


# ---------------------------------------------------------------------------
# prediction_suppression_fractions
# ---------------------------------------------------------------------------


def test_prediction_suppression_fractions_shape():
    kurtosis, skew = _make_labels(L=3, d_mlp=6)
    labels = classify_neurons(kurtosis, skew)
    pred_frac, supp_frac = prediction_suppression_fractions(labels)
    assert pred_frac.shape == (3,)
    assert supp_frac.shape == (3,)


def test_prediction_suppression_fractions_values():
    kurtosis, skew = _make_labels(L=1, d_mlp=6)
    labels = classify_neurons(kurtosis, skew, kurtosis_threshold=10.0)
    pred_frac, supp_frac = prediction_suppression_fractions(labels)
    # 2 of 6 neurons are prediction, 2 are suppression
    assert abs(pred_frac[0].item() - 2 / 6) < 1e-5
    assert abs(supp_frac[0].item() - 2 / 6) < 1e-5


def test_prediction_suppression_fractions_sum_leq_one():
    torch.manual_seed(1)
    kurtosis = torch.randn(L, d_mlp).abs() * 8
    skew = torch.randn(L, d_mlp)
    labels = classify_neurons(kurtosis, skew)
    pred_frac, supp_frac = prediction_suppression_fractions(labels)
    assert ((pred_frac + supp_frac) <= 1.0 + 1e-5).all()


def test_fractions_all_prediction():
    labels = torch.ones(2, 10, dtype=torch.int8)
    pred_frac, supp_frac = prediction_suppression_fractions(labels)
    assert (pred_frac == 1.0).all()
    assert (supp_frac == 0.0).all()


def test_fractions_all_suppression():
    labels = -torch.ones(2, 10, dtype=torch.int8)
    pred_frac, supp_frac = prediction_suppression_fractions(labels)
    assert (pred_frac == 0.0).all()
    assert (supp_frac == 1.0).all()
