"""Tensor-level scientific measurements, independent of model integrations."""

from .attention import (
    get_colsum_concentrations,
    get_legacy_mixing_scores,
    get_mean_sink_rate,
    get_mixing_scores,
    get_sink_scores,
    shannon_entropy,
)
from .comparisons import (
    VectorComparisonMetric,
    VectorComparisonMetrics,
    compare_pairwise_residuals,
    compare_residuals,
    compare_residuals_to_final_state,
    compare_successive_recurrence_residuals,
    compare_vector_sets,
    compare_vectors,
    compute_attn_pattern_diff_norm,
    compute_pairwise_attn_pattern_diff_norm,
    compute_successive_recurrence_attn_pattern_diff_norm,
    compute_to_final_state_attn_pattern_diff_norm,
)
from .convergence import (
    get_final_attention_patterns,
    get_final_residuals,
)
from .residuals import (
    ResidualSequenceMeanComparisonMetrics,
    compare_residuals_to_sequence_mean,
    get_residual_entropy,
    get_residual_norm,
)

__all__ = [
    "ResidualSequenceMeanComparisonMetrics",
    "VectorComparisonMetric",
    "VectorComparisonMetrics",
    "compare_pairwise_residuals",
    "compare_residuals",
    "compare_residuals_to_final_state",
    "compare_residuals_to_sequence_mean",
    "compare_successive_recurrence_residuals",
    "compare_vector_sets",
    "compare_vectors",
    "compute_attn_pattern_diff_norm",
    "compute_pairwise_attn_pattern_diff_norm",
    "compute_successive_recurrence_attn_pattern_diff_norm",
    "compute_to_final_state_attn_pattern_diff_norm",
    "get_colsum_concentrations",
    "get_final_attention_patterns",
    "get_final_residuals",
    "get_legacy_mixing_scores",
    "get_mean_sink_rate",
    "get_mixing_scores",
    "get_residual_entropy",
    "get_residual_norm",
    "get_sink_scores",
    "shannon_entropy",
]
