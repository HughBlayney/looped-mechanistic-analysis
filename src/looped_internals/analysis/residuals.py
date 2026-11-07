# Copyright (c) 2025 Hugh Blayney. MIT license: ../licenses/transformer-tools.txt
import logging
from dataclasses import dataclass

import torch

from .attention import shannon_entropy
from .comparisons import VectorComparisonMetrics, compare_residuals

logger = logging.getLogger(__name__)


def get_residual_entropy(
    residual_stream: torch.Tensor,
    num_components: int | None = None,
) -> torch.Tensor:
    """Effective-rank entropy of the residual stream matrix.

    Uses the squared singular spectrum to measure how evenly information is
    spread across the directions of the residual stream.  High entropy ≈ the
    activations span many dimensions evenly; low entropy ≈ concentration in a
    low-dimensional subspace.

    The "probability" assigned to each singular direction is
    ``σ_i² / Σ_j σ_j²``, which is guaranteed to sum to 1 (because
    ``Σ σ_i² = ‖A‖_F²``).

    By default, compute all squared singular values as the eigenvalues of the
    smaller Gram matrix (``AAᵀ`` or ``AᵀA``) in float32, without centering or
    truncation. Clamp negative eigenvalues caused by roundoff to zero. Forming
    the Gram matrix can lose accuracy in very small spectral components.

    Args:
        residual_stream: ``(B, T, D)`` — residual stream activations from
            e.g. ``cache["resid_post", layer]``.
        num_components: If given, use a randomised truncated SVD retaining only
            the top ``num_components`` singular values (via
            :func:`torch.pca_lowrank`, with its default centering). This
            measures the centered spectrum and renormalizes the retained
            components; omitting the tail can substantially change entropy.
            ``None`` (default) uses the full uncentered Gram spectrum.

    Returns:
        ``(B,)`` — entropy of the normalised squared singular value spectrum.
    """
    logger.debug("get_residual_entropy: shape=%s", tuple(residual_stream.shape))
    rs = residual_stream.float()
    if num_components is not None:
        k = min(num_components, rs.shape[-2], rs.shape[-1])
        _, singular_values, _ = torch.pca_lowrank(rs, q=k, niter=2)  # (B, k)
        squared = singular_values.square()
    else:
        # Keep the Gram product in float32 even during mixed-precision inference.
        with torch.autocast(device_type=rs.device.type, enabled=False):
            gram = rs @ rs.mT if rs.shape[-2] <= rs.shape[-1] else rs.mT @ rs
            # Gram eigenvalues already equal sigma²; no additional squaring needed.
            squared = torch.linalg.eigvalsh(gram).clamp_min(0)  # (B, min(T, D))
    # Frobenius norm² = sum of squared singular values → normalise to a
    # probability distribution along the last dimension.
    frob_sq = squared.sum(dim=-1, keepdim=True)  # (B, 1)
    p_values = squared / frob_sq  # (B, K), sums to 1
    return shannon_entropy(p_values, dim=-1)


def get_residual_norm(
    residual_stream: torch.Tensor,
    position_index: int = 0,
    p: int = 2,
) -> torch.Tensor:
    """p-norm of the residual stream vector at a specific sequence position.

    Args:
        residual_stream: ``(B, T, D)`` — residual stream activations.
        position_index: Which token position to measure (default ``0`` = BOS).
        p: Norm order (default ``2`` = Euclidean norm).

    Returns:
        ``(B,)`` — p-norm of the residual vector at ``position_index``.
    """
    # TODO: generalise to many tokens
    return torch.norm(residual_stream.float()[:, position_index], p=p, dim=-1)


@dataclass(frozen=True)
class ResidualSequenceMeanComparisonMetrics(VectorComparisonMetrics):
    """Paired per-token comparisons to the sequence mean, plus aggregate spread."""

    @property
    def sequence_diff_frobenius_norm(self) -> torch.Tensor:
        """Sequence-wide Frobenius difference norm, shape ``(...,)``.

        This is the root sum of squared per-token difference norms.
        """
        return torch.linalg.vector_norm(self.diff_norm, dim=-1)


def compare_residuals_to_sequence_mean(
    residual_stream: torch.Tensor,
) -> ResidualSequenceMeanComparisonMetrics:
    """Compare each residual vector to its sequence mean, retaining both metrics.

    Input shape is ``(..., T, D)``; ``cosine_sim`` and ``diff_norm`` both have
    shape ``(..., T)``. ``sequence_diff_frobenius_norm`` reduces the per-token
    differences to one aggregate spread value per sequence. Compute the mean
    in float32 before comparing; original vectors remain unnormalized.
    """
    residual_stream = residual_stream.float()
    comparison = compare_residuals(
        residual_stream, residual_stream.mean(dim=-2, keepdim=True)
    )
    return ResidualSequenceMeanComparisonMetrics(
        cosine_sim=comparison.cosine_sim, diff_norm=comparison.diff_norm
    )
