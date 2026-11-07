"""Numerical and storage contracts for paired residual comparison metrics."""

from dataclasses import fields

import pytest
import torch
import torch.nn.functional as F

from looped_internals.analysis import (
    compare_pairwise_residuals,
    compare_residuals,
    compare_residuals_to_final_state,
    compare_residuals_to_sequence_mean,
    compare_successive_recurrence_residuals,
    compare_vector_sets,
    compute_pairwise_attn_pattern_diff_norm,
    compute_successive_recurrence_attn_pattern_diff_norm,
    compute_to_final_state_attn_pattern_diff_norm,
)
from looped_internals.analysis.neurons import compare_neuron_vocab_vectors
from looped_internals.capture.extraction import _comparisons
from looped_internals.results import (
    ExampleResidualComparisonData,
    ModelRunData,
    PerVirtualLayerMetrics,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_vector_metrics_use_raw_magnitudes_and_one_zero_vector_convention(dtype):
    residual_a = torch.tensor([[3, 4], [0, 0], [1, 0], [0, 2]], dtype=dtype)
    residual_b = torch.tensor([[6, 8], [0, 0], [-1, 0], [2, 0]], dtype=dtype)
    comparison = compare_residuals(residual_a, residual_b)
    torch.testing.assert_close(
        comparison.cosine_sim, torch.tensor([1.0, 0.0, -1.0, 0.0])
    )
    torch.testing.assert_close(
        comparison.diff_norm, torch.tensor([5.0, 0.0, 2.0, 8**0.5])
    )
    assert comparison.cosine_sim.dtype == comparison.diff_norm.dtype == torch.float32
    torch.testing.assert_close(
        comparison.angular_diff_fraction, torch.tensor([0.0, 0.5, 1.0, 0.5])
    )


def test_small_vectors_use_separately_clamped_norms_in_both_kernels():
    vectors = torch.tensor([[1e-10, 0], [1e-7, 0], [0, 0]])
    corresponding = compare_residuals(vectors, vectors)
    every_pair = compare_vector_sets(vectors, vectors)
    torch.testing.assert_close(corresponding.cosine_sim, torch.tensor([1e-4, 1.0, 0.0]))
    torch.testing.assert_close(
        every_pair.cosine_sim.diagonal(), corresponding.cosine_sim
    )
    torch.testing.assert_close(every_pair.diff_norm.diagonal(), torch.zeros(3))


def test_vector_sets_preserve_small_differences_between_large_vectors():
    vectors_a = torch.tensor([[10000.0, 10000.0], [10000.0, 10001.0]])
    vectors_b = torch.tensor([[10000.0, 10000.0], [-10000.0, 10000.0]])
    comparison = compare_vector_sets(vectors_a, vectors_b)
    torch.testing.assert_close(
        comparison.diff_norm,
        torch.linalg.vector_norm(vectors_a[:, None] - vectors_b[None], dim=-1),
    )
    torch.testing.assert_close(
        comparison.cosine_sim,
        F.cosine_similarity(vectors_a[:, None], vectors_b[None], dim=-1),
    )
    assert comparison.diff_norm[1, 0] == 1


@pytest.fixture
def activation_history():
    generator = torch.Generator().manual_seed(17)
    residuals = torch.randn(4, 3, 5, generator=generator)
    # Collinear, different magnitudes: cosine alone must not hide movement.
    residuals[2] = 2 * residuals[0]
    attention_patterns = torch.rand(4, 2, 3, 3, generator=generator)
    return residuals, attention_patterns, [0, 2, 0, 2]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_all_comparison_families_match_direct_formulas(activation_history, dtype):
    residuals, attention_patterns, layers = activation_history
    residuals = residuals.to(dtype)
    attention_patterns = attention_patterns.to(dtype)
    expected_residual_diff_norm = (
        residuals.float()[:, None] - residuals.float()[None]
    ).norm(dim=-1)
    expected_residual_cosine_sim = F.cosine_similarity(
        residuals.float()[:, None], residuals.float()[None], dim=-1
    )
    expected_attn_pattern_diff_norm = (
        attention_patterns.float()[:, None] - attention_patterns.float()[None]
    ).norm(dim=(-2, -1))

    pairwise = compare_pairwise_residuals(residuals.unsqueeze(1), output_device="cpu")
    torch.testing.assert_close(pairwise.cosine_sim, expected_residual_cosine_sim)
    torch.testing.assert_close(pairwise.diff_norm, expected_residual_diff_norm)
    torch.testing.assert_close(pairwise.diff_norm, pairwise.diff_norm.transpose(0, 1))
    torch.testing.assert_close(
        compute_pairwise_attn_pattern_diff_norm(attention_patterns.unsqueeze(1)),
        expected_attn_pattern_diff_norm,
    )

    final = compare_residuals_to_final_state(layers, residuals)
    final_attention = compute_to_final_state_attn_pattern_diff_norm(
        layers, attention_patterns
    )
    recurrence = compare_successive_recurrence_residuals(layers, residuals)
    recurrence_attention = compute_successive_recurrence_attn_pattern_diff_norm(
        layers, attention_patterns
    )
    for layer, reference in [(0, 2), (2, 3)]:
        torch.testing.assert_close(
            final.cosine_sim[:, layer], expected_residual_cosine_sim[:, reference]
        )
        torch.testing.assert_close(
            final.diff_norm[:, layer], expected_residual_diff_norm[:, reference]
        )
        torch.testing.assert_close(
            final_attention[:, layer], expected_attn_pattern_diff_norm[:, reference]
        )
        first = reference - 2
        torch.testing.assert_close(
            recurrence[layer].cosine_sim[0],
            expected_residual_cosine_sim[first, reference],
        )
        torch.testing.assert_close(
            recurrence[layer].diff_norm[0],
            expected_residual_diff_norm[first, reference],
        )
        torch.testing.assert_close(
            recurrence_attention[layer][0],
            expected_attn_pattern_diff_norm[first, reference],
        )
    assert final.cosine_sim[:, 1].isnan().all()
    assert final.diff_norm[:, 1].isnan().all()
    assert final_attention[:, 1].isnan().all()
    assert recurrence[1].cosine_sim.shape == recurrence[1].diff_norm.shape == (0, 3)
    assert recurrence_attention[1].shape == (0, 2)


@pytest.mark.parametrize("compute_pairwise", [False, True])
@pytest.mark.parametrize("compute_differences", [False, True])
@pytest.mark.parametrize("attention_availability", ["all", "partial", "none"])
def test_extraction_retains_both_metrics_for_each_enabled_family(
    activation_history, compute_pairwise, compute_differences, attention_availability
):
    residuals, attention_patterns, layers = activation_history
    patterns = list(attention_patterns.unbind())
    if attention_availability == "partial":
        patterns[1] = None
    elif attention_availability == "none":
        patterns = [None] * len(residuals)
    result = _comparisons(
        list(residuals.unbind()),
        patterns,
        layers,
        compute_pairwise,
        compute_differences,
    )
    families = (["pairwise"] if compute_pairwise else []) + (
        ["to_final_state", "successive_recurrence"] if compute_differences else []
    )
    assert set(result) == {
        f"{family}_{quantity}"
        for family in families
        for quantity in (
            "residual_cosine_sim",
            "residual_diff_norm",
            "attn_pattern_diff_norm",
        )
    }
    for family in families:
        cosine_sim = result[f"{family}_residual_cosine_sim"]
        diff_norm = result[f"{family}_residual_diff_norm"]
        cosine_tensors = cosine_sim if isinstance(cosine_sim, list) else [cosine_sim]
        norm_tensors = diff_norm if isinstance(diff_norm, list) else [diff_norm]
        for cosine, norm in zip(cosine_tensors, norm_tensors, strict=True):
            assert cosine.shape == norm.shape
            assert cosine.device.type == norm.device.type == "cpu"
            assert cosine.dtype == norm.dtype == torch.float32
        assert (result[f"{family}_attn_pattern_diff_norm"] is None) == (
            attention_availability != "all"
        )
    if compute_pairwise:
        torch.testing.assert_close(
            result["pairwise_residual_diff_norm"],
            (residuals[:, None] - residuals[None]).norm(dim=-1),
        )


def test_empty_and_single_observation_extraction_preserves_existing_contract():
    assert _comparisons([], [], [], True, True) == {}
    assert _comparisons([torch.ones(2, 3)], [None], [0], True, True) == {}
    comparison = compare_pairwise_residuals([torch.ones(2, 3)])
    torch.testing.assert_close(comparison.diff_norm, torch.zeros(1, 1, 2))
    torch.testing.assert_close(comparison.cosine_sim, torch.ones(1, 1, 2))


def test_mean_comparisons_retain_each_token_and_the_sequence_spread():
    residual = torch.tensor([[[3.0, 4.0], [0.0, 0.0], [-3.0, 4.0]]])
    comparison = compare_residuals_to_sequence_mean(residual)
    mean = residual.mean(dim=-2, keepdim=True)
    torch.testing.assert_close(
        comparison.cosine_sim, F.cosine_similarity(residual, mean, dim=-1)
    )
    torch.testing.assert_close(comparison.diff_norm, (residual - mean).norm(dim=-1))
    torch.testing.assert_close(
        comparison.sequence_diff_frobenius_norm, (residual - mean).norm(dim=(-2, -1))
    )


@pytest.mark.parametrize("neuron_chunk_size", [1, 3, 256])
def test_neuron_vocab_pairing_matches_original_weight_vectors(neuron_chunk_size):
    generator = torch.Generator().manual_seed(8)
    weights = torch.randn(2, 4, 3, generator=generator).bfloat16()
    unembedding = torch.randn(3, 5, generator=generator).bfloat16()
    comparison = compare_neuron_vocab_vectors(
        weights, unembedding, neuron_chunk_size=neuron_chunk_size
    )
    expected_cosine = F.cosine_similarity(
        weights.float()[:, :, None], unembedding.T.float()[None, None], dim=-1
    )
    expected_norm = (
        weights.float()[:, :, None] - unembedding.T.float()[None, None]
    ).norm(dim=-1)
    torch.testing.assert_close(comparison.cosine_sim, expected_cosine)
    torch.testing.assert_close(comparison.diff_norm, expected_norm)


def _run(activation_history):
    residuals, attention, layers = activation_history
    return ModelRunData(
        input_ids=torch.arange(3).unsqueeze(0),
        per_virtual_layer_metrics=[
            PerVirtualLayerMetrics(
                layer_index=layer,
                virtual_layer_index=i,
                sink_scores=None,
                mixing_scores=None,
                residual_norms=residuals[i, 0].norm().reshape(1),
                colsum_concentrations=None,
            )
            for i, layer in enumerate(layers)
        ],
        **_comparisons(list(residuals), list(attention), layers, True, True),
    )


def test_saved_runs_have_all_new_metrics_and_no_adjacent_layer_fields(
    activation_history, tmp_path
):
    run = _run(activation_history)
    path = tmp_path / "run.pt"
    run.save(path)
    stored = torch.load(path, weights_only=True)
    restored = ModelRunData.load(path)
    for family in ("pairwise", "to_final_state", "successive_recurrence"):
        for quantity in (
            "residual_cosine_sim",
            "residual_diff_norm",
            "attn_pattern_diff_norm",
        ):
            name = f"{family}_{quantity}"
            torch.testing.assert_close(
                getattr(restored, name), getattr(run, name), equal_nan=True
            )
    assert not any(name.startswith("successive_virtual_layer") for name in stored)
    assert not any(
        field.name.startswith("successive_virtual_layer")
        for field in fields(ModelRunData)
    )


def test_trajectory_roundtrip_retains_both_series_and_explicit_metric_selection(
    activation_history, tmp_path
):
    run = _run(activation_history)
    trajectory = ExampleResidualComparisonData.from_run(run, 7, ["a", "b", "c"])
    path = tmp_path / "trajectory.pt"
    trajectory.save(path)
    loaded = ExampleResidualComparisonData.load(path)
    for metric in ("cosine_sim", "diff_norm"):
        series = getattr(loaded, f"to_final_state_residual_{metric}_by_layer")
        for layer, indices in [(0, [0, 2]), (2, [1, 3])]:
            torch.testing.assert_close(
                series[layer],
                getattr(run, f"to_final_state_residual_{metric}")[indices, layer],
            )
        expected = run.classify_limiting_behavior_per_token(metric=metric)
        classified = loaded.classify(metric=metric)
        assert classified["behavior_type"].tolist() == [
            type(behavior).__name__ for layer in [0, 2] for behavior in expected[layer]
        ]
    with pytest.raises(ValueError, match="metric must"):
        loaded.classify(metric="distance")
