"""Comprehensive tests for looped_internals.analysis.

Each test targets a specific mathematical property so that failures are
immediately informative.  All tests run on CPU with small tensors so that
no GPU is required and execution is fast.

Shape conventions used throughout:
    B = batch size
    H = number of attention heads
    T = sequence length
    D = model / residual-stream dimension
"""

import math

import pytest
import torch

from looped_internals.analysis import (
    compare_pairwise_residuals,
    compare_residuals,
    compare_residuals_to_final_state,
    compare_residuals_to_sequence_mean,
    compare_successive_recurrence_residuals,
    compute_pairwise_attn_pattern_diff_norm,
    compute_successive_recurrence_attn_pattern_diff_norm,
    compute_to_final_state_attn_pattern_diff_norm,
    get_colsum_concentrations,
    get_final_attention_patterns,
    get_final_residuals,
    get_legacy_mixing_scores,
    get_mean_sink_rate,
    get_mixing_scores,
    get_residual_entropy,
    get_residual_norm,
    get_sink_scores,
    shannon_entropy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, H, T, D = 2, 4, 8, 16  # default tensor dimensions for most tests


def _uniform_attention(b=B, h=H, t=T) -> torch.Tensor:
    """Return a uniform attention pattern (all rows equal 1/T)."""
    return torch.full((b, h, t, t), 1.0 / t)


def _full_sink_attention(b=B, h=H, t=T, key_index=0) -> torch.Tensor:
    """Return an attention pattern where every query attends only to key_index."""
    attn = torch.zeros(b, h, t, t)
    attn[:, :, :, key_index] = 1.0
    return attn


def _identity_attention(b=B, h=H, t=T) -> torch.Tensor:
    """Return an identity-like attention pattern (each query attends to itself)."""
    eye = torch.eye(t)
    return eye.expand(b, h, t, t).clone()


def _causal_uniform_attention(b=B, h=H, t=T) -> torch.Tensor:
    """Each row is uniform over positions 0..i (causal mask applied)."""
    attn = torch.zeros(b, h, t, t)
    for i in range(t):
        attn[:, :, i, : i + 1] = 1.0 / (i + 1)
    return attn


def _split_sink_attention(b=B, h=H, t=T) -> torch.Tensor:
    """First half of rows attends to token 0; second half to the midpoint token."""
    mid = t // 2
    attn = torch.zeros(b, h, t, t)
    attn[:, :, :mid, 0] = 1.0
    attn[:, :, mid:, mid] = 1.0
    return attn


def _prev_token_attention(b=B, h=H, t=T) -> torch.Tensor:
    """Row i attends to token i-1; row 0 attends to itself."""
    attn = torch.zeros(b, h, t, t)
    attn[:, :, 0, 0] = 1.0
    for i in range(1, t):
        attn[:, :, i, i - 1] = 1.0
    return attn


# ---------------------------------------------------------------------------
# shannon_entropy
# ---------------------------------------------------------------------------


class TestShannonEntropy:
    def test_uniform_is_maximum(self):
        """Uniform distribution maximises entropy: H = log(K)."""
        K = 8
        p = torch.full((K,), 1.0 / K)
        expected = math.log(K)
        result = shannon_entropy(p, dim=-1)
        assert torch.isclose(result, torch.tensor(expected), atol=1e-5)

    def test_delta_is_zero(self):
        """A point-mass distribution has zero entropy."""
        p = torch.zeros(8)
        p[3] = 1.0
        result = shannon_entropy(p, dim=-1)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_batched(self):
        """Entropy is computed independently along the specified dimension."""
        p = torch.full((3, 4), 1.0 / 4)
        result = shannon_entropy(p, dim=-1)
        assert result.shape == (3,)
        assert torch.allclose(result, torch.full((3,), math.log(4)), atol=1e-5)

    def test_zero_probability_handled(self):
        """Zero probability entries should not produce NaN (lim p→0 p log p = 0)."""
        p = torch.tensor([0.5, 0.5, 0.0, 0.0])
        result = shannon_entropy(p, dim=-1)
        assert not torch.isnan(result)
        assert torch.isclose(result, torch.tensor(math.log(2)), atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_causal_rows_match_float64_reference(self, dtype):
        """Keep masked zeros exact and reduce noncontiguous attention in float32."""
        generator = torch.Generator().manual_seed(42)
        p = torch.rand(2, 3, 17, 17, generator=generator).tril()
        p = (p / p.sum(dim=-1, keepdim=True)).to(dtype).transpose(0, 1)
        original = p.clone()
        reference = p.double()
        terms = torch.zeros_like(reference)
        positive = reference > 0
        terms[positive] = -reference[positive] * reference[positive].log()

        actual = shannon_entropy(p)

        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual.double(), terms.sum(-1), atol=1e-6, rtol=0)
        torch.testing.assert_close(p, original, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# get_sink_scores
# ---------------------------------------------------------------------------


class TestGetSinkScores:
    def test_full_sink_score_is_one(self):
        """When all queries attend to token 0, sink score = 1 for every head."""
        attn = _full_sink_attention()
        scores = get_sink_scores(attn, key_index=0)
        assert scores.shape == (B, H)
        assert torch.allclose(scores, torch.ones(B, H))

    def test_no_sink_score_is_low(self):
        """For an identity attention (each query attends only to itself), sink
        score at key 0 should be 1/T."""
        attn = _identity_attention()
        scores = get_sink_scores(attn, key_index=0)
        expected = 1.0 / T
        assert torch.allclose(scores, torch.full((B, H), expected), atol=1e-6)

    def test_uniform_sink_score(self):
        """For uniform attention, every key position gets the same sink score 1/T."""
        attn = _uniform_attention()
        scores = get_sink_scores(attn, key_index=0)
        assert torch.allclose(scores, torch.full((B, H), 1.0 / T), atol=1e-6)

    def test_custom_key_index(self):
        """Sink score can target an arbitrary key position."""
        attn = _full_sink_attention(key_index=3)
        scores_0 = get_sink_scores(attn, key_index=0)
        scores_3 = get_sink_scores(attn, key_index=3)
        assert torch.allclose(scores_0, torch.zeros(B, H), atol=1e-6)
        assert torch.allclose(scores_3, torch.ones(B, H))

    def test_output_shape(self):
        """Output shape is always (B, H)."""
        attn = torch.rand(3, 5, 12, 12)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        scores = get_sink_scores(attn)
        assert scores.shape == (3, 5)


# ---------------------------------------------------------------------------
# get_mean_sink_rate
# ---------------------------------------------------------------------------


class TestGetMeanSinkRate:
    def test_all_heads_exceed_threshold(self):
        """When all heads have sink score = 1, the rate should be 1."""
        attn = _full_sink_attention()
        rate = get_mean_sink_rate(attn, threshold=0.3)
        assert rate.shape == (B,)
        assert torch.allclose(rate, torch.ones(B))

    def test_no_head_exceeds_threshold(self):
        """When sink scores are all 1/T (uniform attention) < threshold, rate = 0."""
        attn = _uniform_attention()
        # 1/T = 1/8 = 0.125 < 0.3
        rate = get_mean_sink_rate(attn, threshold=0.3)
        assert torch.allclose(rate, torch.zeros(B))

    def test_half_heads_exceed_threshold(self):
        """Mix of sink and non-sink heads: rate should be 0.5."""
        assert H % 2 == 0, "Test assumes even H"
        attn = torch.zeros(B, H, T, T)
        # First half of heads are full sinks
        attn[:, : H // 2, :, 0] = 1.0
        # Second half are identity (each attends to itself, score = 1/T < 0.3)
        for i in range(H // 2, H):
            attn[:, i, :, :] = torch.eye(T)
        rate = get_mean_sink_rate(attn, threshold=0.3)
        assert torch.allclose(rate, torch.full((B,), 0.5))


# ---------------------------------------------------------------------------
# get_legacy_mixing_scores
# ---------------------------------------------------------------------------


class TestGetLegacyMixingScores:
    def test_full_sink_has_zero_mixing(self):
        """A full-sink pattern (deterministic attention) has zero row entropy."""
        attn = _full_sink_attention()
        scores = get_legacy_mixing_scores(attn)
        assert scores.shape == (B, H)
        assert torch.allclose(scores, torch.zeros(B, H), atol=1e-6)

    def test_identity_has_zero_mixing(self):
        """Identity attention (each query attends to one position) → zero entropy."""
        attn = _identity_attention()
        scores = get_legacy_mixing_scores(attn)
        assert torch.allclose(scores, torch.zeros(B, H), atol=1e-6)

    def test_uniform_is_maximum_mixing(self):
        """Uniform attention maximises normalised row entropy → score = 1.0."""
        attn = _uniform_attention()
        scores = get_legacy_mixing_scores(attn)
        assert torch.allclose(scores, torch.ones(B, H), atol=1e-5)

    def test_mixing_non_negative(self):
        """Mixing scores are always ≥ 0."""
        torch.manual_seed(0)
        attn = torch.rand(B, H, T, T)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        assert (get_legacy_mixing_scores(attn) >= 0).all()


# ---------------------------------------------------------------------------
# get_mixing_scores
# ---------------------------------------------------------------------------


class TestGetMixingScores:
    def test_full_sink_has_zero_mixing(self):
        """Full sink: every row is deterministic → mixing = 0."""
        attn = _full_sink_attention()
        scores = get_mixing_scores(attn)
        assert scores.shape == (B, H)
        assert torch.allclose(scores, torch.zeros(B, H), atol=1e-6)

    def test_split_sink_has_zero_mixing(self):
        """Split sink: rows are still deterministic → mixing = 0."""
        attn = _split_sink_attention()
        scores = get_mixing_scores(attn)
        assert torch.allclose(scores, torch.zeros(B, H), atol=1e-6)

    def test_uniform_causal_is_one(self):
        """Uniform causal: row i has entropy log(i+1), normalised by log(i+1) → 1.0."""
        attn = _causal_uniform_attention()
        scores = get_mixing_scores(attn)
        assert torch.allclose(scores, torch.ones(B, H), atol=1e-5)

    def test_identity_has_zero_mixing(self):
        """Identity: each row is a point mass → mixing = 0."""
        attn = _identity_attention()
        scores = get_mixing_scores(attn)
        assert torch.allclose(scores, torch.zeros(B, H), atol=1e-6)

    def test_prev_token_has_zero_mixing(self):
        """Previous-token: each row is a point mass → mixing = 0."""
        attn = _prev_token_attention()
        scores = get_mixing_scores(attn)
        assert torch.allclose(scores, torch.zeros(B, H), atol=1e-6)


# ---------------------------------------------------------------------------
# get_colsum_concentrations
# ---------------------------------------------------------------------------


class TestGetColsumConcentrations:
    def test_full_sink_is_one(self):
        """All attention to one key → concentration = 1."""
        attn = _full_sink_attention()
        conc = get_colsum_concentrations(attn)
        assert conc.shape == (B, H)
        assert torch.allclose(conc, torch.ones(B, H), atol=1e-5)

    def test_uniform_is_zero(self):
        """Uniform attention → each key receives equal attention → concentration ≈ 0."""
        attn = _uniform_attention()
        conc = get_colsum_concentrations(attn)
        assert torch.allclose(conc, torch.zeros(B, H), atol=1e-5)

    def test_range_in_zero_one(self):
        """Concentration should always be in [0, 1]."""
        torch.manual_seed(1)
        attn = torch.rand(B, H, T, T)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        conc = get_colsum_concentrations(attn)
        assert (conc >= 0).all() and (conc <= 1.0 + 1e-6).all()

    def test_split_sink_exact_value(self):
        """Split sink puts half the column mass on each of two positions.

        Normalised col sums: [0.5, 0, ..., 0, 0.5, 0, ...].
        Entropy = log(2), so concentration = 1 - log(2)/log(T) = 1 + log(0.5)/log(T).
        """
        attn = _split_sink_attention()
        conc = get_colsum_concentrations(attn)
        expected = 1.0 + math.log(0.5) / math.log(T)
        assert torch.allclose(conc, torch.full((B, H), expected), atol=1e-5)

    def test_identity_is_zero(self):
        """Identity attention distributes column mass perfectly uniformly → concentration = 0."""
        attn = _identity_attention()
        conc = get_colsum_concentrations(attn)
        assert torch.allclose(conc, torch.zeros(B, H), atol=1e-5)


# ---------------------------------------------------------------------------
# get_residual_entropy
# ---------------------------------------------------------------------------


class TestGetResidualEntropy:
    @pytest.mark.parametrize("shape", [(2, 12, 32), (2, 32, 12), (2, 16, 16)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("offset", [0.0, 20.0])
    @pytest.mark.parametrize("autocast", [False, True])
    def test_matches_float64_svd(self, shape, dtype, offset, autocast):
        """Match the uncentered spectrum for wide, tall, and square batches."""
        generator = torch.Generator().manual_seed(41)
        rs = (torch.randn(shape, generator=generator) + offset).to(dtype)
        original = rs.clone()
        squared = torch.linalg.svdvals(rs.double()).square()
        probabilities = squared / squared.sum(dim=-1, keepdim=True)
        expected = -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)

        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            actual = get_residual_entropy(rs)

        assert actual.dtype == torch.float32
        assert actual.device == rs.device
        torch.testing.assert_close(actual.double(), expected, atol=1e-5, rtol=0)
        torch.testing.assert_close(rs, original, atol=0, rtol=0)

    def test_known_squared_spectrum_with_zero_component(self):
        """Normalize sigma², not sigma or sigma⁴; zero components stay finite."""
        rs = torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.0])).unsqueeze(0)
        expected = -sum(p * math.log(p) for p in (16 / 21, 4 / 21, 1 / 21))
        torch.testing.assert_close(get_residual_entropy(rs), torch.tensor([expected]))

    def test_uniform_spectrum_keeps_all_components(self):
        rs = torch.eye(64).unsqueeze(0)
        expected = torch.tensor([math.log(64)])
        torch.testing.assert_close(get_residual_entropy(rs), expected)

    def test_zero_matrix_has_zero_entropy(self):
        entropy = get_residual_entropy(torch.zeros(B, T, D))
        torch.testing.assert_close(entropy, torch.zeros(B))

    def test_rank1_has_zero_entropy(self):
        """A rank-1 matrix has only one non-zero singular value → entropy = 0."""
        v = torch.randn(D)
        # Outer product: (B, T, D) where each 'matrix' is rank-1
        rs = torch.outer(torch.ones(T), v).unsqueeze(0).expand(B, T, D).clone()
        entropy = get_residual_entropy(rs)
        assert entropy.shape == (B,)
        assert torch.allclose(entropy, torch.zeros(B), atol=1e-4)

    def test_entropy_is_non_negative(self):
        """Entropy is always ≥ 0."""
        torch.manual_seed(2)
        rs = torch.randn(B, T, D)
        assert (get_residual_entropy(rs) >= 0).all()

    def test_entropy_increases_with_rank(self):
        """Higher-rank residual streams should (typically) have higher entropy."""
        torch.manual_seed(3)
        # Rank-1: only one direction of variation
        v = torch.randn(D)
        rs_low = (torch.randn(B, T, 1) * v.unsqueeze(0)).reshape(B, T, D)

        # Full-rank random: many independent directions
        rs_high = torch.randn(B, T, D)

        entropy_low = get_residual_entropy(rs_low).mean()
        entropy_high = get_residual_entropy(rs_high).mean()
        assert entropy_high > entropy_low

    def test_p_values_sum_to_one(self):
        """Sanity check: the normalised squared singular values must sum to 1."""
        torch.manual_seed(4)
        rs = torch.randn(B, T, D).float()
        sv = torch.linalg.svdvals(rs)
        sq = sv.pow(2)
        p = sq / sq.sum(dim=-1, keepdim=True)
        assert torch.allclose(p.sum(dim=-1), torch.ones(B), atol=1e-5)

    def test_num_components_output_shape(self):
        """Truncated SVD path still returns (B,)."""
        torch.manual_seed(15)
        rs = torch.randn(B, T, D)
        entropy = get_residual_entropy(rs, num_components=4)
        assert entropy.shape == (B,)

    def test_num_components_non_negative(self):
        """Truncated SVD entropy is non-negative."""
        torch.manual_seed(16)
        rs = torch.randn(B, T, D)
        assert (get_residual_entropy(rs, num_components=4) >= 0).all()

    def test_num_components_clamped_to_matrix_rank(self):
        """Requesting more components than min(T, D) should not raise."""
        torch.manual_seed(17)
        rs = torch.randn(B, T, D)
        # min(T, D) = min(8, 16) = 8; requesting 100 should silently clamp
        entropy = get_residual_entropy(rs, num_components=100)
        assert entropy.shape == (B,)

    def test_num_components_rank1_is_zero(self):
        """Rank-1 matrix: truncated SVD also gives entropy ≈ 0 (single direction)."""
        v = torch.randn(D)
        rs = torch.outer(torch.ones(T), v).unsqueeze(0).expand(B, T, D).clone()
        entropy = get_residual_entropy(rs, num_components=4)
        assert torch.allclose(entropy, torch.zeros(B), atol=1e-3)

    def test_num_components_agrees_with_full_svd_on_low_rank(self):
        """For a matrix with exactly k non-zero singular values, requesting
        num_components=k should give an entropy close to the full SVD.

        The randomised SVD (pca_lowrank) is an approximation, so we allow a
        loose tolerance of 0.5 nats — the point is that it captures the dominant
        directions, not that it matches to floating-point precision.
        """
        torch.manual_seed(18)
        k = 3
        # Build a rank-k matrix: U (T×k) @ S_diag @ V (k×D)
        U = torch.randn(T, k)
        S = torch.tensor([3.0, 2.0, 1.0])
        V = torch.randn(k, D)
        rs = (U * S.unsqueeze(0)) @ V  # (T, D), rank k
        rs = rs.unsqueeze(0).expand(B, T, D).clone()

        entropy_full = get_residual_entropy(rs)
        entropy_trunc = get_residual_entropy(rs, num_components=k)
        assert torch.allclose(entropy_full, entropy_trunc, atol=0.5)


# ---------------------------------------------------------------------------
# get_residual_norm
# ---------------------------------------------------------------------------


class TestGetResidualNorm:
    def test_known_norm(self):
        """Norm of a known vector is computed correctly."""
        rs = torch.zeros(B, T, D)
        # Set position 0 of every batch item to a vector of all-ones
        rs[:, 0, :] = 1.0
        norms = get_residual_norm(rs, position_index=0, p=2)
        assert norms.shape == (B,)
        expected = math.sqrt(D)
        assert torch.allclose(norms, torch.full((B,), expected), atol=1e-5)

    def test_zero_vector(self):
        rs = torch.zeros(B, T, D)
        norms = get_residual_norm(rs, position_index=0)
        assert torch.allclose(norms, torch.zeros(B))

    def test_position_index(self):
        """The function should measure the norm at the specified position only."""
        rs = torch.zeros(B, T, D)
        rs[:, 0, :] = 1.0  # position 0 has norm sqrt(D)
        rs[:, 3, :] = 2.0  # position 3 has norm 2*sqrt(D)
        assert not torch.allclose(
            get_residual_norm(rs, position_index=0),
            get_residual_norm(rs, position_index=3),
        )


# ---------------------------------------------------------------------------
# compare_residuals_to_sequence_mean
# ---------------------------------------------------------------------------


class TestGetMeanDifference:
    def test_identical_tokens_is_zero(self):
        """When all tokens are the same, the mean difference is 0."""
        v = torch.randn(D)
        rs = v.unsqueeze(0).unsqueeze(0).expand(B, T, D).clone()
        diff = compare_residuals_to_sequence_mean(rs).sequence_diff_frobenius_norm
        assert diff.shape == (B,)
        assert torch.allclose(diff, torch.zeros(B), atol=1e-5)

    def test_varied_tokens_is_positive(self):
        """Varied token representations give a positive mean difference."""
        torch.manual_seed(5)
        rs = torch.randn(B, T, D)
        diff = compare_residuals_to_sequence_mean(rs).sequence_diff_frobenius_norm
        assert (diff > 0).all()

    def test_larger_variation_gives_larger_difference(self):
        """Scaling the residual stream scales the Frobenius norm linearly."""
        torch.manual_seed(6)
        rs = torch.randn(B, T, D)
        diff1 = compare_residuals_to_sequence_mean(rs).sequence_diff_frobenius_norm
        diff2 = compare_residuals_to_sequence_mean(
            rs * 2.0
        ).sequence_diff_frobenius_norm
        assert torch.allclose(diff2, diff1 * 2.0, atol=1e-4)

    def test_batch_dimension_is_independent(self):
        """Each batch item should be measured independently."""
        rs = torch.zeros(2, T, D)
        rs[0] = torch.randn(T, D)  # varied
        rs[1] = torch.ones(T, D)  # uniform → difference = 0
        diff = compare_residuals_to_sequence_mean(rs).sequence_diff_frobenius_norm
        assert diff[0] > 0
        assert torch.isclose(diff[1], torch.tensor(0.0), atol=1e-5)


# ---------------------------------------------------------------------------
# compare_residuals
# ---------------------------------------------------------------------------


class TestGetAngularDifference:
    def test_identical_vectors_is_zero(self):
        """Identical residual streams → angle = 0."""
        rs = torch.randn(B, T, D)
        diff = compare_residuals(rs, rs).angular_diff_fraction
        assert diff.shape == (B, T)
        # acos near 0 has limited float32 precision; 5e-4 accounts for this
        assert torch.allclose(diff, torch.zeros(B, T), atol=5e-4)

    def test_opposite_vectors_is_one(self):
        """Anti-parallel residual streams → angle/π = 1."""
        rs = torch.randn(B, T, D)
        diff = compare_residuals(rs, -rs).angular_diff_fraction
        # acos near π has limited float32 precision; 5e-4 accounts for this
        assert torch.allclose(diff, torch.ones(B, T), atol=5e-4)

    def test_orthogonal_vectors_is_half(self):
        """Orthogonal residual streams → angle/π = 0.5."""
        # Construct orthogonal pairs: v and w s.t. v·w = 0
        assert D >= 2
        rs_a = torch.zeros(B, T, D)
        rs_b = torch.zeros(B, T, D)
        rs_a[:, :, 0] = 1.0  # e_0
        rs_b[:, :, 1] = 1.0  # e_1 — orthogonal to e_0
        diff = compare_residuals(rs_a, rs_b).angular_diff_fraction
        assert torch.allclose(diff, torch.full((B, T), 0.5), atol=1e-5)

    def test_range_zero_to_one(self):
        """Angular difference is always in [0, 1]."""
        torch.manual_seed(7)
        rs_a = torch.randn(B, T, D)
        rs_b = torch.randn(B, T, D)
        diff = compare_residuals(rs_a, rs_b).angular_diff_fraction
        assert (diff >= 0).all() and (diff <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# compute_pairwise_attn_pattern_diff_norm
# ---------------------------------------------------------------------------


class TestGetAttentionPatternNorms:
    def test_identical_layers_have_zero_distance(self):
        """The diagonal of the pairwise distance matrix should be 0."""
        # Use b=1 so squeeze(0) correctly removes the batch dimension
        attn = _full_sink_attention(b=1, h=H, t=T)  # (1, H, T, T)
        layer_attn = attn.squeeze(0)  # (H, T, T)
        stack = layer_attn.unsqueeze(0).expand(3, H, T, T).clone()  # (3, H, T, T)
        norms = compute_pairwise_attn_pattern_diff_norm(stack)
        assert norms.shape == (3, 3, H)
        # Diagonal entries: distance from layer i to itself must be 0
        # torch.diagonal over (dim1=0, dim2=1) for a (3,3,H) tensor → (H, 3)
        assert torch.allclose(
            norms.diagonal(dim1=0, dim2=1), torch.zeros(H, 3), atol=1e-5
        )

    def test_symmetry(self):
        """Pairwise distance matrix is symmetric."""
        torch.manual_seed(8)
        attn_stack = torch.rand(4, H, T, T)
        attn_stack = attn_stack / attn_stack.sum(dim=-1, keepdim=True)
        norms = compute_pairwise_attn_pattern_diff_norm(attn_stack)
        assert torch.allclose(norms, norms.permute(1, 0, 2), atol=1e-5)

    def test_with_batch_dim(self):
        """A (L, B, H, T, T) stack is handled by squeezing the batch dim."""
        attn_stack = torch.rand(3, 1, H, T, T)
        norms = compute_pairwise_attn_pattern_diff_norm(attn_stack)
        assert norms.shape == (3, 3, H)


# ---------------------------------------------------------------------------
# compare_pairwise_residuals
# ---------------------------------------------------------------------------


class TestGetResidualStreamCosineSimilarity:
    def test_identical_layers_have_similarity_one(self):
        """Identical residual streams across layers → all cosine similarities = 1."""
        rs = torch.randn(T, D)
        stack = rs.unsqueeze(0).expand(3, T, D).clone()
        sims = compare_pairwise_residuals(stack).cosine_sim
        assert sims.shape == (3, 3, T)
        assert torch.allclose(sims, torch.ones(3, 3, T), atol=1e-5)

    def test_symmetry(self):
        """Cosine similarity matrix is symmetric."""
        torch.manual_seed(9)
        stack = torch.randn(4, T, D)
        sims = compare_pairwise_residuals(stack).cosine_sim
        assert torch.allclose(sims, sims.permute(1, 0, 2), atol=1e-5)

    def test_range(self):
        """Cosine similarities are bounded in [-1, 1]."""
        torch.manual_seed(10)
        stack = torch.randn(5, T, D)
        sims = compare_pairwise_residuals(stack).cosine_sim
        assert (sims >= -1.0 - 1e-5).all() and (sims <= 1.0 + 1e-5).all()


# ---------------------------------------------------------------------------
# get_final_residuals
# ---------------------------------------------------------------------------


class TestGetFixedPointResiduals:
    def test_last_occurrence_wins(self):
        """For repeated layer indices, the final occurrence should be returned."""
        rs0 = torch.zeros(T, D)
        _rs1 = torch.ones(T, D)
        rs2 = torch.full((T, D), 2.0)

        # Layer 0 appears twice; final value is rs2
        layer_indices = [0, 0]
        streams = [rs0, rs2]

        fp = get_final_residuals(layer_indices, streams)
        assert fp.shape == (1, T, D)
        assert torch.allclose(fp[0], rs2)

    def test_multiple_unique_layers(self):
        """Each layer's fixed-point is placed at the correct index."""
        rs = {0: torch.zeros(T, D), 1: torch.ones(T, D), 2: torch.full((T, D), 2.0)}
        layer_indices = [0, 1, 2]
        streams = [rs[0], rs[1], rs[2]]

        fp = get_final_residuals(layer_indices, streams)
        assert fp.shape == (3, T, D)
        for i in range(3):
            assert torch.allclose(fp[i], rs[i])

    def test_missing_layer_is_zero(self):
        """Layers not appearing in layer_indices should be zero-filled."""
        rs0 = torch.ones(T, D)
        rs2 = torch.full((T, D), 2.0)
        # Layer 1 is never visited
        layer_indices = [0, 2]
        streams = [rs0, rs2]

        fp = get_final_residuals(layer_indices, streams)
        assert fp.shape == (3, T, D)
        assert torch.allclose(fp[1], torch.zeros(T, D))


# ---------------------------------------------------------------------------
# get_final_attention_patterns
# ---------------------------------------------------------------------------


class TestGetFixedPointAttentionPatterns:
    def test_last_occurrence_wins(self):
        attn0 = torch.zeros(H, T, T)
        attn1 = torch.ones(H, T, T)
        fp = get_final_attention_patterns([0, 0], [attn0, attn1])
        assert fp.shape == (1, H, T, T)
        assert torch.allclose(fp[0], attn1)


# ---------------------------------------------------------------------------
# compare_residuals_to_final_state
# ---------------------------------------------------------------------------


class TestGetFixedPointNorms:
    def test_zero_distance_to_self(self):
        """If each block's residual stream IS the fixed point, all norms are 0."""
        streams = [torch.randn(T, D) for _ in range(3)]
        # layer 0,1,2 each appears once → their fixed point IS themselves
        layer_indices = [0, 1, 2]
        norms = compare_residuals_to_final_state(layer_indices, streams).diff_norm
        # Shape: (n_layers=3, n_unique_layers=3, T)
        assert norms.shape == (3, 3, T)
        # Diagonal: block i vs fixed point of layer i → should be 0
        for i in range(3):
            assert torch.allclose(norms[i, i], torch.zeros(T), atol=1e-5)

    def test_non_zero_distance(self):
        """Fixed-point residuals different from block residuals give positive distances."""
        rs_early = torch.zeros(T, D)
        rs_late = torch.ones(T, D)
        # Layer 0 appears twice: early then late; fixed point is rs_late
        layer_indices = [0, 0]
        streams = [rs_early, rs_late]
        norms = compare_residuals_to_final_state(layer_indices, streams).diff_norm
        # Block 0 (rs_early) vs fixed point of layer 0 (rs_late) should be positive
        assert (norms[0, 0] > 0).all()
        # Block 1 (rs_late) vs fixed point of layer 0 (rs_late) should be 0
        assert torch.allclose(norms[1, 0], torch.zeros(T), atol=1e-5)


# ---------------------------------------------------------------------------
# compare_residuals_to_final_state
# ---------------------------------------------------------------------------


class TestGetFixedPointCosineSimilarities:
    def test_similarity_to_self_is_one(self):
        """Fixed-point residual vs itself should have cosine similarity = 1."""
        streams = [torch.randn(T, D) for _ in range(3)]
        layer_indices = [0, 1, 2]
        sims = compare_residuals_to_final_state(layer_indices, streams).cosine_sim
        # Block i vs fixed point of layer i (= streams[i]) should be 1
        for i in range(3):
            assert torch.allclose(sims[i, i], torch.ones(T), atol=1e-4)

    def test_opposite_vectors_similarity_is_minus_one(self):
        """Anti-parallel residuals should have cosine similarity ≈ -1."""
        rs_pos = torch.ones(T, D)
        rs_neg = -torch.ones(T, D)
        layer_indices = [0, 0]
        streams = [rs_pos, rs_neg]  # fixed point is rs_neg
        sims = compare_residuals_to_final_state(layer_indices, streams).cosine_sim
        # Block 0 (rs_pos) vs fixed point of layer 0 (rs_neg) → -1
        assert torch.allclose(sims[0, 0], torch.full((T,), -1.0), atol=1e-4)


# ---------------------------------------------------------------------------
# compute_to_final_state_attn_pattern_diff_norm
# ---------------------------------------------------------------------------


class TestGetFixedPointAttentionDifferences:
    def test_zero_distance_to_self(self):
        """Attention patterns that are already at their fixed point have distance 0."""
        attns = [torch.rand(H, T, T) for _ in range(3)]
        layer_indices = [0, 1, 2]
        diffs = compute_to_final_state_attn_pattern_diff_norm(layer_indices, attns)
        assert diffs.shape == (3, 3, H)
        for i in range(3):
            assert torch.allclose(diffs[i, i], torch.zeros(H), atol=1e-5)


# ---------------------------------------------------------------------------
# compare_successive_recurrence_residuals
# ---------------------------------------------------------------------------


class TestGetSuccessiveNormDifferences:
    def test_single_occurrence_has_empty_tensor(self):
        """Layers that appear only once have empty successive-recurrence tensors."""
        streams = [torch.randn(T, D), torch.randn(T, D)]
        result = [
            comparison.diff_norm
            for comparison in compare_successive_recurrence_residuals([0, 1], streams)
        ]
        assert len(result) == 2
        assert all(value.numel() == 0 for value in result)

    def test_constant_stream_is_zero(self):
        """If the residual stream does not change across recurrences, norm = 0."""
        rs = torch.randn(T, D)
        streams = [rs, rs, rs]  # layer 0 repeated three times
        result = [
            comparison.diff_norm
            for comparison in compare_successive_recurrence_residuals(
                [0, 0, 0], streams
            )
        ]
        assert len(result) == 1
        assert torch.allclose(result[0], torch.zeros(2, T), atol=1e-6)

    def test_changing_stream_is_positive(self):
        """Differing successive residuals produce positive norms."""
        torch.manual_seed(11)
        streams = [torch.randn(T, D), torch.randn(T, D)]
        result = [
            comparison.diff_norm
            for comparison in compare_successive_recurrence_residuals([0, 0], streams)
        ]
        assert (result[0] > 0).all()

    def test_output_shape(self):
        """Output shape is (n_recurrences - 1, T)."""
        streams = [torch.randn(T, D)] * 5
        result = [
            comparison.diff_norm
            for comparison in compare_successive_recurrence_residuals([0] * 5, streams)
        ]
        assert result[0].shape == (4, T)


# ---------------------------------------------------------------------------
# compare_successive_recurrence_residuals
# ---------------------------------------------------------------------------


class TestGetSuccessiveCosSimlarities:
    def test_single_occurrence_has_empty_tensor(self):
        result = [
            comparison.cosine_sim
            for comparison in compare_successive_recurrence_residuals(
                [0, 1], [torch.randn(T, D)] * 2
            )
        ]
        assert len(result) == 2
        assert all(value.numel() == 0 for value in result)

    def test_identical_streams_is_one(self):
        rs = torch.randn(T, D)
        result = [
            comparison.cosine_sim
            for comparison in compare_successive_recurrence_residuals(
                [0, 0, 0], [rs, rs, rs]
            )
        ]
        assert torch.allclose(result[0], torch.ones(2, T), atol=1e-5)

    def test_opposite_streams_is_minus_one(self):
        rs = torch.randn(T, D)
        result = [
            comparison.cosine_sim
            for comparison in compare_successive_recurrence_residuals([0, 0], [rs, -rs])
        ]
        assert torch.allclose(result[0], torch.full((1, T), -1.0), atol=1e-5)

    def test_range(self):
        """Cosine similarities are in [-1, 1]."""
        torch.manual_seed(12)
        streams = [torch.randn(T, D) for _ in range(4)]
        result = [
            comparison.cosine_sim
            for comparison in compare_successive_recurrence_residuals([0] * 4, streams)
        ]
        sims = result[0]
        assert (sims >= -1.0 - 1e-5).all() and (sims <= 1.0 + 1e-5).all()


# ---------------------------------------------------------------------------
# compute_successive_recurrence_attn_pattern_diff_norm
# ---------------------------------------------------------------------------


class TestGetSuccessiveAttentionDifferences:
    def test_single_occurrence_has_empty_tensor(self):
        attns = [torch.rand(H, T, T)] * 2
        result = compute_successive_recurrence_attn_pattern_diff_norm([0, 1], attns)
        assert len(result) == 2
        assert all(value.numel() == 0 for value in result)

    def test_identical_attention_is_zero(self):
        attn = torch.rand(H, T, T)
        result = compute_successive_recurrence_attn_pattern_diff_norm(
            [0, 0, 0], [attn, attn, attn]
        )
        assert torch.allclose(result[0], torch.zeros(2, H), atol=1e-5)

    def test_output_shape(self):
        """Output shape is (n_recurrences - 1, H)."""
        attns = [torch.rand(H, T, T) for _ in range(5)]
        result = compute_successive_recurrence_attn_pattern_diff_norm([0] * 5, attns)
        assert result[0].shape == (4, H)

    def test_positive_for_different_weights(self):
        torch.manual_seed(13)
        attns = [torch.rand(H, T, T) for _ in range(2)]
        result = compute_successive_recurrence_attn_pattern_diff_norm([0, 0], attns)
        assert (result[0] > 0).all()

    def test_multiple_layers(self):
        """Different layer indices are tracked independently."""
        torch.manual_seed(14)
        attns = [torch.rand(H, T, T) for _ in range(4)]
        layer_indices = [0, 1, 0, 1]
        result = compute_successive_recurrence_attn_pattern_diff_norm(
            layer_indices, attns
        )
        assert len(result) == 2
        assert result[0].shape == (1, H)  # layer 0: 2 recurrences → 1 diff
        assert result[1].shape == (1, H)  # layer 1: 2 recurrences → 1 diff
