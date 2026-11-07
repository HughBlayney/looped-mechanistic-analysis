"""Tests for looped_internals.capture — compute_model_run_data().

Uses MinimalRecurrentLM (tiny, CPU-only) as a test fixture.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge

from looped_internals.capture.extraction import (
    _cached_virtual_layers,
    _get_attn_pattern,
    compute_model_run_data,
    metrics_names_filter,
)
from looped_internals.models import ModelSpec, load_model
from looped_internals.models.adapters.minimal import MinimalRecurrentAdapter
from looped_internals.models.minimal import (
    MinimalRecurrentConfig,
    MinimalRecurrentLM,
)
from looped_internals.results import ModelRunData
from looped_internals.structure import LayerGroup, ModelLayerStructure, VirtualLayer


def _make_minimal_bridge(
    num_cycles: int = 3,
) -> tuple[TransformerBridge, ModelLayerStructure]:
    """Create a tiny MinimalRecurrentLM bridge for testing."""
    config = MinimalRecurrentConfig(
        d_model=32,
        n_heads=2,
        n_layers=2,
        num_cycles=num_cycles,
        d_vocab=100,
        expansion_factor=2,
    )
    model = MinimalRecurrentLM(config)
    model.eval()

    tl_cfg = TransformerBridgeConfig(
        d_model=32,
        d_head=16,
        n_layers=2,
        n_ctx=32,
        n_heads=2,
        d_vocab=100,
        architecture="MinimalRecurrentLM",
    )
    adapter = MinimalRecurrentAdapter(tl_cfg)
    bridge = TransformerBridge(deepcopy(model), adapter, tokenizer=None)
    bridge.eval()

    # MinimalRecurrentLM uses "blocks" prefix, all recurrent
    structure = ModelLayerStructure(
        groups=[
            LayerGroup("blocks", 2, is_recurrent=True),
        ]
    )
    return bridge, structure


class TestCachedVirtualLayers:
    @pytest.fixture(scope="class")
    def bridge(self):
        return _make_minimal_bridge()[0]

    def test_forward_order_and_iteration_key_precedence(self, bridge):
        """Keep layer identities and numeric iteration order in a partial cache."""
        structure = ModelLayerStructure(
            [
                LayerGroup("prelude_blocks", 2, False),
                LayerGroup("core_blocks", 2, True),
                LayerGroup("coda_blocks", 1, False),
            ]
        )
        names = [
            "coda_blocks.0.hook_out.iter_0",
            "core_blocks.1.hook_out.iter_10",
            "core_blocks.0.hook_out.iter_2",
            "core_blocks.0.hook_out",
            "core_blocks.0.hook_out.iter_10",
            "core_blocks.1.hook_out.iter_2",
            "prelude_blocks.1.hook_out",
            "core_blocks.0.attn.q.hook_out.iter_2",
        ]
        cache = ActivationCache(dict.fromkeys(names, torch.empty(0)), bridge)

        assert _cached_virtual_layers(cache, structure) == [
            VirtualLayer(1, 0, "prelude_blocks.1", None),
            VirtualLayer(2, 1, "core_blocks.0", 2),
            VirtualLayer(3, 2, "core_blocks.1", 2),
            VirtualLayer(2, 3, "core_blocks.0", 10),
            VirtualLayer(3, 4, "core_blocks.1", 10),
            VirtualLayer(4, 5, "coda_blocks.0", 0),
        ]

    @pytest.mark.parametrize(
        "suffix",
        [
            ".iter_",
            ".iteration_0",
            ".iter.0",
            "_iter_0",
            ".iter_-1",
            ".iter_+1",
            ".iter_01",
            ".iter_1.0",
            ".iter_1.extra",
            ".iter_ 1",
            ".iter_\u0661",
            ".iter_foo",
        ],
    )
    @pytest.mark.parametrize("is_recurrent", [False, True])
    def test_rejects_malformed_iteration_keys(self, bridge, suffix, is_recurrent):
        """Malformed keys must not disappear behind valid cached entries."""
        key = "blocks.0.hook_out"
        cache = ActivationCache(
            dict.fromkeys([key, key + ".iter_0", key + suffix], torch.empty(0)),
            bridge,
        )
        structure = ModelLayerStructure([LayerGroup("blocks", 1, is_recurrent)])

        with pytest.raises(ValueError, match="Invalid iteration cache key") as error:
            _cached_virtual_layers(cache, structure)
        assert repr(key + suffix) in str(error.value)
        assert f"{key}.iter_N" in str(error.value)

    @pytest.mark.parametrize("accumulate", [False, True])
    @pytest.mark.parametrize("fused", [False, True])
    def test_raw_projections_use_the_virtual_layer_iteration(
        self, bridge, accumulate, fused
    ):
        """Residual, attention, and Q/K/V lookups must select the same application."""
        input_ids = torch.zeros((1, 3), dtype=torch.long)
        structure = ModelLayerStructure([LayerGroup("blocks", 1, True)])
        suffixes = [".iter_0", ".iter_2"] if accumulate else [""]
        cache_dict = {}
        expected = []
        for i, suffix in enumerate(suffixes):
            residual = torch.full((1, 3, 32), float(i + 1))
            attention = torch.full((1, 2, 3, 3), float(i + 1))
            q, k, v = [torch.randn(1, 3, 32) for _ in range(3)]
            cache_dict[f"blocks.0.hook_out{suffix}"] = residual
            cache_dict[f"blocks.0.attn.hook_pattern{suffix}"] = attention
            if fused:
                cache_dict[f"blocks.0.attn.qkv.hook_out{suffix}"] = torch.cat(
                    [q, k, v], dim=-1
                )
            else:
                for name, tensor in zip(("q", "k", "v"), (q, k, v), strict=True):
                    cache_dict[f"blocks.0.attn.{name}.hook_out{suffix}"] = tensor
            expected.append((residual, attention, q, k, v))
        if accumulate:
            # Unsuffixed values must never replace an accumulated application.
            cache_dict["blocks.0.hook_out"] = torch.full((1, 3, 32), float("nan"))
        cache = ActivationCache(cache_dict, bridge)

        result = compute_model_run_data(
            cache, input_ids, structure, compute_pairwise=False, store_raw_values=True
        )[0]

        for metrics, (residual, attention, q, k, v) in zip(
            result.per_virtual_layer_metrics, expected, strict=True
        ):
            torch.testing.assert_close(metrics.raw_residual, residual)
            torch.testing.assert_close(metrics.raw_attn_pattern, attention)
            for actual, projection in zip(
                (metrics.raw_attn_q, metrics.raw_attn_k, metrics.raw_attn_v),
                (q, k, v),
                strict=True,
            ):
                torch.testing.assert_close(
                    actual, projection.view(1, 3, 2, 16).permute(0, 2, 1, 3)
                )


class TestComputeModelRunData:
    @pytest.fixture(scope="class")
    def bridge_and_structure(self):
        return _make_minimal_bridge(num_cycles=3)

    @pytest.fixture(scope="class")
    def run_output(self, bridge_and_structure):
        bridge, structure = bridge_and_structure
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(input_ids, accumulate=True)
        return cache, input_ids, structure

    def test_returns_model_run_data(self, run_output):
        cache, input_ids, structure = run_output
        results = compute_model_run_data(cache, input_ids, structure)
        assert isinstance(results, list)
        assert isinstance(results[0], ModelRunData)

    def test_per_virtual_layer_metrics_count(self, run_output):
        """Should produce n_layers × num_cycles entries (2 blocks × 3 cycles = 6)."""
        cache, input_ids, structure = run_output
        result = compute_model_run_data(cache, input_ids, structure)[0]
        # 2 blocks × 3 recurrences = 6
        assert len(result.per_virtual_layer_metrics) == 6

    def test_layer_indices(self, run_output):
        """With iter-major ordering, layers interleave: [0,1, 0,1, 0,1]."""
        cache, input_ids, structure = run_output
        result = compute_model_run_data(cache, input_ids, structure)[0]
        layer_indices = [pbm.layer_index for pbm in result.per_virtual_layer_metrics]
        # Both layers should appear 3 times each
        assert layer_indices.count(0) == 3
        assert layer_indices.count(1) == 3
        # Execution order: iter-major — layers alternate within each cycle
        assert layer_indices == [0, 1, 0, 1, 0, 1]

    def test_virtual_layer_indices(self, run_output):
        """virtual_layer_index should be 0..5, one per execution step."""
        cache, input_ids, structure = run_output
        result = compute_model_run_data(cache, input_ids, structure)[0]
        seq_indices = [
            pbm.virtual_layer_index for pbm in result.per_virtual_layer_metrics
        ]
        assert seq_indices == list(range(6))

    def test_per_virtual_layer_metrics_execution_order(self, run_output):
        cache, input_ids, structure = run_output
        result = compute_model_run_data(cache, input_ids, structure)[0]
        seq_indices = [
            pbm.virtual_layer_index for pbm in result.per_virtual_layer_metrics
        ]
        assert seq_indices == sorted(seq_indices)

    def test_get_metric_by_layer_groups_by_layer(self, run_output):
        """get_metric(group_by='layer') should yield 2 layers × 3 iterations each."""
        cache, input_ids, structure = run_output
        result = compute_model_run_data(cache, input_ids, structure)[0]
        bd = result.get_metric("sink_scores", group_by="layer")
        assert list(bd.keys()) == ["L0", "L1"]
        assert bd["L0"].shape[0] == 3  # 3 iterations for layer 0
        assert bd["L1"].shape[0] == 3  # 3 iterations for layer 1

    def test_residual_norms_finite_positive(self, run_output):
        cache, input_ids, structure = run_output
        results = compute_model_run_data(cache, input_ids, structure)
        entries = _cached_virtual_layers(cache, structure)
        for batch, result in enumerate(results):
            for pbm, entry in zip(
                result.per_virtual_layer_metrics, entries, strict=True
            ):
                residual = cache.cache_dict[entry.residual_key][
                    batch : batch + 1
                ].float()
                norms = residual.norm(dim=-1)
                for actual, expected in (
                    (pbm.residual_norms, norms[:, 0]),
                    (pbm.residual_norms_last, norms[:, -1]),
                    (pbm.residual_norms_mean, norms.mean(-1)),
                ):
                    assert actual.shape == (1,)
                    assert torch.isfinite(actual).all() and (actual >= 0).all()
                    torch.testing.assert_close(actual, expected)

    def test_sink_scores_in_range(self, run_output):
        cache, input_ids, structure = run_output
        result = compute_model_run_data(cache, input_ids, structure)[0]
        for pbm in result.per_virtual_layer_metrics:
            if pbm.sink_scores is not None:
                assert pbm.sink_scores.min() >= -1e-4
                assert pbm.sink_scores.max() <= 1 + 1e-4

    def test_residual_entropy_populated(self, run_output):
        """residual_entropy is a finite (1,) tensor when compute_residual_entropy=True."""
        cache, input_ids, structure = run_output
        result = compute_model_run_data(
            cache, input_ids, structure, compute_residual_entropy=True
        )[0]
        for pbm in result.per_virtual_layer_metrics:
            assert pbm.residual_entropy is not None
            assert pbm.residual_entropy.shape == (1,)
            assert pbm.residual_entropy.isfinite().all()

    def test_residual_entropy_none_by_default(self, run_output):
        """residual_entropy is None when compute_residual_entropy is not set."""
        cache, input_ids, structure = run_output
        result = compute_model_run_data(cache, input_ids, structure)[0]
        for pbm in result.per_virtual_layer_metrics:
            assert pbm.residual_entropy is None

    def test_residual_entropy_truncated_svd(self, run_output):
        """residual_entropy_components passes through to truncated SVD and gives
        a finite, non-negative (1,) result."""
        cache, input_ids, structure = run_output
        result = compute_model_run_data(
            cache,
            input_ids,
            structure,
            compute_residual_entropy=True,
            residual_entropy_components=8,
        )[0]
        for pbm in result.per_virtual_layer_metrics:
            assert pbm.residual_entropy is not None
            assert pbm.residual_entropy.shape == (1,)
            assert pbm.residual_entropy.isfinite().all()
            assert (pbm.residual_entropy >= 0).all()

    def test_pairwise_residual_metrics_present_together(self, run_output):
        cache, input_ids, structure = run_output
        result = compute_model_run_data(
            cache, input_ids, structure, compute_pairwise=True
        )[0]
        assert result.pairwise_residual_cosine_sim is not None
        assert result.pairwise_residual_diff_norm is not None
        assert (
            result.pairwise_residual_diff_norm.shape
            == result.pairwise_residual_cosine_sim.shape
        )
        # (L, L) where L = n_layers × recurrences = 6
        assert result.pairwise_residual_cosine_sim.shape[0] == 6
        assert result.pairwise_residual_cosine_sim.shape[1] == 6

    def test_input_ids_preserved(self, run_output):
        cache, input_ids, structure = run_output
        result = compute_model_run_data(cache, input_ids, structure)[0]
        assert torch.equal(result.input_ids, input_ids.cpu())


class TestComputeModelRunDataNonRecurrent:
    """Test with a non-recurrent single-group structure."""

    @pytest.fixture(scope="class")
    def bridge_and_structure(self):
        bridge, _ = _make_minimal_bridge(num_cycles=1)
        # Override structure to be non-recurrent
        structure = ModelLayerStructure(
            groups=[
                LayerGroup("blocks", 2, is_recurrent=False),
            ]
        )
        return bridge, structure

    @pytest.mark.parametrize("accumulate", [False, True])
    def test_non_recurrent_single_entry_per_layer(
        self, bridge_and_structure, accumulate
    ):
        bridge, structure = bridge_and_structure
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(input_ids, accumulate=accumulate)
        result = compute_model_run_data(cache, input_ids, structure)[0]
        # 2 blocks × 1 recurrence = 2
        assert len(result.per_virtual_layer_metrics) == 2

    def test_rejects_multiple_iterations_in_non_recurrent_group(self):
        """Reject a cache whose recurrent applications contradict the layout."""
        bridge, structure = _make_minimal_bridge(num_cycles=2)
        structure.groups[0].is_recurrent = False
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(input_ids, accumulate=True)

        with pytest.raises(
            ValueError,
            match=(
                r"Non-recurrent group 'blocks' has multiple cached iterations "
                r"for layer 0: \[0, 1\]"
            ),
        ):
            compute_model_run_data(cache, input_ids, structure)

    def test_non_recurrent_virtual_layer_indices(self, bridge_and_structure):
        bridge, structure = bridge_and_structure
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(input_ids, accumulate=True)
        result = compute_model_run_data(cache, input_ids, structure)[0]
        seq_indices = [
            pbm.virtual_layer_index for pbm in result.per_virtual_layer_metrics
        ]
        assert seq_indices == [0, 1]


class TestDifferenceMetrics:
    def test_differences_computed_when_requested(self):
        bridge, structure = _make_minimal_bridge(num_cycles=4)
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(input_ids, accumulate=True)
        result = compute_model_run_data(
            cache, input_ids, structure, compute_differences=True
        )[0]
        assert result.to_final_state_residual_diff_norm is not None
        assert result.successive_recurrence_residual_diff_norm is not None

    def test_differences_not_computed_when_not_requested(self):
        bridge, structure = _make_minimal_bridge(num_cycles=3)
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(input_ids, accumulate=True)
        result = compute_model_run_data(
            cache, input_ids, structure, compute_differences=False
        )[0]
        assert result.to_final_state_residual_diff_norm is None


class TestAttnPatternAvailable:
    """Attention patterns must be non-None for the Minimal model.

    MinimalAttentionBridge fires hook_pattern so _get_attn_pattern should
    always return a tensor for the Minimal model.
    """

    @pytest.fixture(scope="class")
    def cache_and_structure(self):
        bridge, structure = _make_minimal_bridge(num_cycles=2)
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(input_ids, accumulate=True)
        return cache, structure

    def test_attn_pattern_not_none_via_helper(self, cache_and_structure):
        """_get_attn_pattern should return a tensor, not None, for the Minimal model."""
        cache, _ = cache_and_structure
        pattern = _get_attn_pattern(cache, "blocks.0")
        assert pattern is not None, (
            "_get_attn_pattern returned None for Minimal model — "
            "MinimalAttentionBridge should fire hook_pattern"
        )
        assert pattern.dim() == 4, (
            f"Expected 4D (B,H,T,T) pattern, got shape {pattern.shape}"
        )

    def test_attn_pattern_iter_not_none(self, cache_and_structure):
        """Per-iteration attention patterns should also be non-None."""
        cache, _ = cache_and_structure
        from looped_internals.capture.extraction import _get_attn_pattern_iter

        pattern = _get_attn_pattern_iter(cache, "blocks.0", iter_idx=0)
        assert pattern is not None, "_get_attn_pattern_iter returned None for iter 0"

    def test_sink_scores_not_none(self, cache_and_structure):
        """Attention-based metrics (sink, mixing, colsum) must be populated."""
        cache, structure = cache_and_structure
        input_ids = torch.randint(0, 100, (1, 8))
        result = compute_model_run_data(cache, input_ids, structure)[0]
        for pbm in result.per_virtual_layer_metrics:
            assert pbm.sink_scores is not None, (
                f"sink_scores None at layer {pbm.layer_index} seq {pbm.virtual_layer_index}"
            )
            assert pbm.mixing_scores is not None
            assert pbm.colsum_concentrations is not None


class TestMetricsNamesFilter:
    """metrics_names_filter passes only hook_out and hook_pattern."""

    def test_passes_block_hook_out(self):
        assert metrics_names_filter("blocks.0.hook_out")
        assert metrics_names_filter("core_blocks.3.hook_out")
        assert metrics_names_filter("prelude_blocks.0.hook_out")
        assert metrics_names_filter("coda_blocks.1.hook_out")

    def test_passes_attn_hook_pattern(self):
        assert metrics_names_filter("blocks.0.attn.hook_pattern")
        assert metrics_names_filter("core_blocks.2.attn.hook_pattern")

    def test_rejects_non_metric_hooks(self):
        """All other hooks should be rejected — use names_filter=None for raw values."""
        assert not metrics_names_filter("blocks.0.attn.hook_q")
        assert not metrics_names_filter("blocks.0.attn.hook_k")
        assert not metrics_names_filter("blocks.0.attn.q.hook_out")
        assert not metrics_names_filter("blocks.0.attn.k.hook_out")
        assert not metrics_names_filter("blocks.0.attn.qkv.hook_out")
        assert not metrics_names_filter("blocks.0.attn.hook_in")
        assert not metrics_names_filter("blocks.0.attn.o.hook_out")
        assert not metrics_names_filter("blocks.0.attn.v.hook_out")
        assert not metrics_names_filter("blocks.0.mlp.hook_out")
        assert not metrics_names_filter("blocks.0.ln1.hook_out")
        assert not metrics_names_filter("blocks.0.hook_in")


class TestLoadMinimal:
    """End-to-end test for load_model(ModelSpec(config)) → run_with_cache → compute_model_run_data.

    Regression test for a bug where the minimal loader used
    LayerGroup("transformer_layers")
    instead of LayerGroup("blocks"), causing all cache.get_iterations() calls to raise
    KeyError and produce empty per_virtual_layer_metrics.
    """

    N_LAYERS = 2
    N_CYCLES = 3

    @pytest.fixture(scope="class")
    def result(self):
        config = MinimalRecurrentConfig(
            d_model=32,
            n_heads=2,
            n_layers=self.N_LAYERS,
            num_cycles=self.N_CYCLES,
            d_vocab=100,
            expansion_factor=2,
        )
        bridge, structure = load_model(ModelSpec(config), device="cpu")
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(input_ids, accumulate=True)
        return compute_model_run_data(cache, input_ids, structure)[0]

    def test_per_virtual_layer_metrics_not_empty(self, result):
        """Regression: was empty when LayerGroup prefix did not match TL hook prefix."""
        assert len(result.per_virtual_layer_metrics) == self.N_LAYERS * self.N_CYCLES

    def test_residual_norms_finite(self, result):
        for pbm in result.per_virtual_layer_metrics:
            assert torch.isfinite(pbm.residual_norms).all()


class TestLoadMinimalFeedForward:
    """Regression: gated_mlp=False / FeedForward path must also produce metrics."""

    def test_feedforward_per_virtual_layer_metrics(self):
        config = MinimalRecurrentConfig(
            d_model=32,
            n_heads=2,
            n_layers=2,
            num_cycles=2,
            d_vocab=100,
            expansion_factor=2,
            gated_mlp=False,
            activation="relu",
        )
        bridge, structure = load_model(ModelSpec(config), device="cpu")
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(input_ids, accumulate=True)
        result = compute_model_run_data(cache, input_ids, structure)[0]
        assert (
            len(result.per_virtual_layer_metrics) == config.n_layers * config.num_cycles
        )


def test_reduction_does_not_stack_or_copy_raw_history(monkeypatch, tmp_path):
    """Compare against direct tensor formulas, and forbid full-history stacks."""
    from looped_internals.results._serialization import state_dict

    bridge, structure = _make_minimal_bridge(num_cycles=3)
    ids = torch.randint(0, 100, (2, 5))
    with torch.no_grad():
        _, cache = bridge.run_with_cache(
            ids, accumulate=True, names_filter=metrics_names_filter, device="cpu"
        )
    before = dict(cache.cache_dict)
    executions = list(structure.virtual_layers(3))
    residual = torch.stack([cache[e.residual_key][0].float() for e in executions])
    attention = torch.stack([cache[e.attn_key][0].float() for e in executions])
    expected_cosine = torch.nn.functional.cosine_similarity(
        residual[:, None], residual[None, :], dim=-1
    )
    expected_attention = (attention[:, None] - attention[None, :]).norm(dim=(-2, -1))

    original_stack = torch.stack

    def bounded_stack(tensors, *args, **kwargs):
        assert not tensors or tensors[0].ndim < 3, (
            "Raw activation history must not be stacked"
        )
        return original_stack(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "stack", bounded_stack)
    runs = compute_model_run_data(
        cache, ids, structure, compute_differences=True, store_raw_values=True
    )
    run = runs[0]
    torch.testing.assert_close(
        run.pairwise_residual_cosine_sim, expected_cosine, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        run.pairwise_residual_diff_norm,
        (residual[:, None] - residual[None, :]).norm(dim=-1),
    )
    torch.testing.assert_close(run.pairwise_attn_pattern_diff_norm, expected_attention)
    for layer in range(2):
        final = residual[4 + layer]
        torch.testing.assert_close(
            run.to_final_state_residual_diff_norm[:, layer],
            (residual - final).norm(dim=-1),
        )
    for key, tensor in before.items():
        assert cache.cache_dict[key] is tensor
    assert (
        run.per_virtual_layer_metrics[0].raw_residual.data_ptr()
        == before[executions[0].residual_key].data_ptr()
    )
    assert (
        state_dict(run)["per_virtual_layer_metrics"][0]["raw_residual"]
        is run.per_virtual_layer_metrics[0].raw_residual
    )
    run.save(tmp_path / "run.pt")
    restored = ModelRunData.load(tmp_path / "run.pt")
    torch.testing.assert_close(
        restored.to_final_state_residual_cosine_sim,
        run.to_final_state_residual_cosine_sim,
    )
    assert all(
        b.residual_norms.device.type == "cpu" for b in run.per_virtual_layer_metrics
    )
