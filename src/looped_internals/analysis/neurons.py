# Copyright (c) 2025 Hugh Blayney. MIT license: licenses/transformer-tools.txt
"""Weight-based analysis of transformer MLP neurons.

Functions in this module operate on raw weight tensors (no activation cache
required).  The primary use-case is classifying MLP neurons as *prediction*
or *suppression* neurons following Gurnee et al. (2024) "Universal Neurons in
GPT2 Language Models" §5.1.

The core idea: for each MLP neuron *j* in layer *l*, compute the distribution
of cosine similarities between its output weight vector ``w_out[j]`` and every
token's unembedding vector (i.e. every column of ``W_U``).  The shape of this
distribution reveals the neuron's functional role:

* **Prediction neuron** — high excess kurtosis, positive skew: the neuron
  strongly promotes a small coherent set of vocabulary tokens.
* **Suppression neuron** — high excess kurtosis, negative skew: the neuron
  strongly suppresses a small coherent set of vocabulary tokens.
* **Partition neuron** — high variance, moderate kurtosis: the neuron boosts
  roughly half the vocabulary while suppressing the other half.

Typical workflow::

    from looped_internals.analysis.neurons import (
        compare_neuron_vocab_vectors,
        neuron_vocab_cosine_moments,
        classify_neurons,
        prediction_suppression_fractions,
    )

    # W_out: (n_layers, d_mlp, d_model)  from bridge.W_out or tl_parameters()
    # W_U:   (d_model, d_vocab)           from bridge.tl_parameters()["unembed.W_U"]
    neuron_vocab_comparison = compare_neuron_vocab_vectors(W_out, W_U)
    moments = neuron_vocab_cosine_moments(neuron_vocab_comparison.cosine_sim)
    labels  = classify_neurons(moments["kurtosis"], moments["skew"])
    pred_frac, supp_frac = prediction_suppression_fractions(labels)

Shape conventions
-----------------
* ``L``       – number of layers
* ``d_mlp``   – MLP hidden dimension (neurons per layer)
* ``d_model`` – residual stream / model dimension
* ``d_vocab`` – vocabulary size
"""

from __future__ import annotations

import torch
from torch import Tensor

from .comparisons import VectorComparisonMetrics, compare_vector_sets

__all__ = [
    "center_columns",
    "center_rows",
    "classify_neurons",
    "compare_neuron_vocab_vectors",
    "neuron_vocab_cosine_moments",
    "prediction_suppression_fractions",
]

# ---------------------------------------------------------------------------
# Weight preprocessing utilities (optional, matches paper §A.1)
# ---------------------------------------------------------------------------


def center_columns(W: Tensor) -> Tensor:
    """Subtract the mean along the last dimension (column centering).

    Used to preprocess ``W_U`` per Gurnee et al. (2024) §A.1: since softmax
    is translation-invariant, components of ``W_U`` along the all-ones
    direction have no effect and can be removed.

    Args:
        W: Any tensor; centering is applied along ``dim=-1``.

    Returns:
        Tensor of the same shape with zero column means.
    """
    return W - W.mean(dim=-1, keepdim=True)


def center_rows(W: Tensor) -> Tensor:
    """Subtract the mean along the second-to-last dimension (row centering).

    Used to preprocess ``W_out`` per Gurnee et al. (2024) §A.1: writing
    weight components along the all-ones residual-stream direction have no
    downstream effect after a subsequent LayerNorm.

    Args:
        W: Tensor of shape ``(..., d_mlp, d_model)``; centering is applied
           along ``dim=-2`` (across neurons for each model dimension).

    Returns:
        Tensor of the same shape with zero row means along ``dim=-2``.
    """
    return W - W.mean(dim=-2, keepdim=True)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compare_neuron_vocab_vectors(
    W_out: Tensor, W_U: Tensor, *, neuron_chunk_size: int = 256
) -> VectorComparisonMetrics:
    """Compare every neuron output vector with every vocabulary unembedding.

    Args:
        W_out: MLP output weights, shape ``(L, d_mlp, d_model)``.
        W_U: Unembedding matrix, shape ``(d_model, d_vocab)``.
        neuron_chunk_size: Number of neurons compared at once per layer,
            bounding temporary storage while retaining both full outputs.

    Returns:
        Paired ``cosine_sim`` and ``diff_norm`` tensors, both float32 with
        shape ``(L, d_mlp, d_vocab)`` on the input device. Difference norms
        compare the original weight vectors. Cosine moments can be used for
        prediction/suppression classification via ``neuron_vocab_cosine_moments``.
    """
    if neuron_chunk_size < 1:
        raise ValueError("neuron_chunk_size must be positive")
    if W_out.ndim != 3 or W_U.ndim != 2 or W_out.shape[-1] != W_U.shape[0]:
        raise ValueError("Expected W_out (L, d_mlp, D) and W_U (D, d_vocab)")
    if W_out.device != W_U.device:
        raise ValueError("Neuron and vocabulary weights must be on the same device")
    layers, neurons, _ = W_out.shape
    neuron_vocab_cosine_sim = torch.empty(
        layers, neurons, W_U.shape[-1], dtype=torch.float32, device=W_out.device
    )
    neuron_vocab_diff_norm = torch.empty_like(neuron_vocab_cosine_sim)
    vocabulary_vectors = W_U.T.float()
    for layer in range(layers):
        for start in range(0, neurons, neuron_chunk_size):
            stop = start + neuron_chunk_size
            comparison = compare_vector_sets(
                W_out[layer, start:stop], vocabulary_vectors
            )
            neuron_vocab_cosine_sim[layer, start:stop] = comparison.cosine_sim
            neuron_vocab_diff_norm[layer, start:stop] = comparison.diff_norm
    return VectorComparisonMetrics(
        cosine_sim=neuron_vocab_cosine_sim, diff_norm=neuron_vocab_diff_norm
    )


# ---------------------------------------------------------------------------
# Statistical moments
# ---------------------------------------------------------------------------


def neuron_vocab_cosine_moments(neuron_vocab_cosine_sim: Tensor) -> dict[str, Tensor]:
    """Statistical moments of the vocab cosine distribution for each neuron.

    Args:
        neuron_vocab_cosine_sim: Cosine similarities, shape ``(L, d_mlp, d_vocab)``.

    Returns:
        Dictionary with keys:

        * ``"mean"``     — mean cosine similarity, shape ``(L, d_mlp)``
        * ``"variance"`` — variance, shape ``(L, d_mlp)``
        * ``"skew"``     — third standardised moment, shape ``(L, d_mlp)``
        * ``"kurtosis"`` — *excess* kurtosis (Gaussian = 0), shape ``(L, d_mlp)``

        Kurtosis is the excess form (subtract 3) matching scipy and the paper.
    """
    neuron_vocab_cosine_sim = neuron_vocab_cosine_sim.float()
    mean = neuron_vocab_cosine_sim.mean(dim=-1)  # (L, d_mlp)
    var = neuron_vocab_cosine_sim.var(dim=-1, unbiased=False)  # (L, d_mlp)
    std = var.sqrt().clamp(min=1e-8)

    z = (neuron_vocab_cosine_sim - mean.unsqueeze(-1)) / std.unsqueeze(
        -1
    )  # standardised
    skew = (z**3).mean(dim=-1)
    kurtosis = (z**4).mean(dim=-1) - 3.0  # excess kurtosis

    return {
        "mean": mean,
        "variance": var,
        "skew": skew,
        "kurtosis": kurtosis,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_neurons(
    kurtosis: Tensor,
    skew: Tensor,
    kurtosis_threshold: float = 10.0,
    skew_threshold: float = 0.0,
) -> Tensor:
    """Classify each MLP neuron as prediction, suppression, or other.

    Classification rules (Gurnee et al. 2024 §5.1):

    * **Prediction neuron** (label ``1``): excess kurtosis ≥ *kurtosis_threshold*
      **and** skew > *skew_threshold*.
    * **Suppression neuron** (label ``-1``): excess kurtosis ≥
      *kurtosis_threshold* **and** skew ≤ *skew_threshold*.
    * **Other** (label ``0``): kurtosis below threshold.

    Args:
        kurtosis:           Excess kurtosis per neuron, shape ``(L, d_mlp)``.
        skew:               Skewness per neuron, shape ``(L, d_mlp)``.
        kurtosis_threshold: Minimum excess kurtosis for a neuron to be
                            classified as prediction or suppression.
                            Paper uses 10.0 (Figure 6).
        skew_threshold:     Boundary between prediction (above) and suppression
                            (at or below).  Defaults to 0.0.

    Returns:
        Integer label tensor, shape ``(L, d_mlp)``:
        ``1`` = prediction, ``-1`` = suppression, ``0`` = other.
    """
    kurtosis = kurtosis.float()
    skew = skew.float()

    high_kurt = kurtosis >= kurtosis_threshold
    labels = torch.zeros_like(kurtosis, dtype=torch.int8)
    labels[high_kurt & (skew > skew_threshold)] = 1
    labels[high_kurt & (skew <= skew_threshold)] = -1
    return labels


def prediction_suppression_fractions(
    labels: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-layer fraction of prediction and suppression neurons.

    Args:
        labels: Integer label tensor, shape ``(L, d_mlp)``.
                Values: ``1`` = prediction, ``-1`` = suppression, ``0`` = other.

    Returns:
        ``(pred_frac, supp_frac)`` — two tensors of shape ``(L,)`` giving the
        fraction of neurons in each layer that are classified as prediction
        or suppression neurons respectively.
    """
    n_neurons = labels.shape[-1]
    pred_frac = (labels == 1).float().sum(dim=-1) / n_neurons
    supp_frac = (labels == -1).float().sum(dim=-1) / n_neurons
    return pred_frac, supp_frac
