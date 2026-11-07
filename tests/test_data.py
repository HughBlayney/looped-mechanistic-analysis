"""Tests for looped_internals.results — serialisation round-trip."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

from looped_internals.results import (
    DepthAccuracy,
    DepthEvaluationResult,
    DepthValues,
    ExampleResidualComparisonData,
    FixedPoint,
    LimitingBehavior,
    ModelRunData,
    Orbit,
    PerVirtualLayerMetrics,
    Slider,
    Unknown,
    aggregate_values,
)


def _make_per_virtual_layer(
    layer_index: int, virtual_layer_index: int
) -> PerVirtualLayerMetrics:
    return PerVirtualLayerMetrics(
        layer_index=layer_index,
        virtual_layer_index=virtual_layer_index,
        sink_scores=torch.rand(2, 4),  # (batch, heads)
        mixing_scores=torch.rand(2, 4),
        residual_norms=torch.rand(2),  # (batch,)
        colsum_concentrations=torch.rand(2, 4),
    )


def _make_per_virtual_layer_b1(
    layer_index: int, virtual_layer_index: int
) -> PerVirtualLayerMetrics:
    """Like _make_per_virtual_layer but with batch size 1, as required by get_metric."""
    return PerVirtualLayerMetrics(
        layer_index=layer_index,
        virtual_layer_index=virtual_layer_index,
        sink_scores=torch.rand(1, 4),
        mixing_scores=torch.rand(1, 4),
        residual_norms=torch.rand(1),
        colsum_concentrations=torch.rand(1, 4),
    )


def _make_model_run_data() -> ModelRunData:
    # Simulates: layer 0 fires at seq 0, layer 1 fires at seq 1 and 2.
    # S=3 virtual layers, L=2 unique layers, T=16 tokens, H=4 heads.
    return ModelRunData(
        input_ids=torch.randint(0, 100, (2, 16)),
        per_virtual_layer_metrics=[
            _make_per_virtual_layer(0, 0),
            _make_per_virtual_layer(1, 1),
            _make_per_virtual_layer(1, 2),
        ],
        pairwise_attn_pattern_diff_norm=torch.rand(3, 3, 4),  # (S, S, H)
        pairwise_residual_cosine_sim=torch.rand(3, 3, 16),
        pairwise_residual_diff_norm=torch.rand(3, 3, 16),  # (S, S, T)
        to_final_state_residual_diff_norm=torch.rand(3, 2, 16),  # (S, L, T)
        to_final_state_residual_cosine_sim=torch.rand(3, 2, 16),
        to_final_state_attn_pattern_diff_norm=torch.rand(3, 2, 4),  # (S, L, H)
        successive_recurrence_residual_diff_norm=[
            torch.empty(0, 16),
            torch.rand(1, 16),
        ],
        successive_recurrence_residual_cosine_sim=[
            torch.empty(0, 16),
            torch.rand(1, 16),
        ],
        successive_recurrence_attn_pattern_diff_norm=[
            torch.empty(0, 4),
            torch.rand(1, 4),
        ],
    )


def _make_model_run_data_b1() -> ModelRunData:
    """ModelRunData with batch-1 PerVirtualLayerMetrics: layer 0 fires once, layer 1 fires twice."""
    return ModelRunData(
        input_ids=torch.randint(0, 100, (1, 16)),
        per_virtual_layer_metrics=[
            _make_per_virtual_layer_b1(0, 0),
            _make_per_virtual_layer_b1(1, 1),
            _make_per_virtual_layer_b1(1, 2),
        ],
    )


def _make_recurrent_model_data_b1() -> ModelRunData:
    """4-layer pure-recurrent model, 2 iterations: 0,1,2,3,0,1,2,3."""
    return ModelRunData(
        input_ids=torch.randint(0, 100, (1, 8)),
        per_virtual_layer_metrics=[
            _make_per_virtual_layer_b1(0, 0),
            _make_per_virtual_layer_b1(1, 1),
            _make_per_virtual_layer_b1(2, 2),
            _make_per_virtual_layer_b1(3, 3),
            _make_per_virtual_layer_b1(0, 4),
            _make_per_virtual_layer_b1(1, 5),
            _make_per_virtual_layer_b1(2, 6),
            _make_per_virtual_layer_b1(3, 7),
        ],
    )


def _make_raven_model_data_b1() -> ModelRunData:
    """Raven-style model: prelude [0], core [1,2] × 2, coda [3].

    Execution order: 0, 1, 2, 1, 2, 3  (virtual layer indices 0–5).
    """
    return ModelRunData(
        input_ids=torch.randint(0, 100, (1, 6)),
        per_virtual_layer_metrics=[
            _make_per_virtual_layer_b1(0, 0),  # prelude
            _make_per_virtual_layer_b1(1, 1),  # core iter 0
            _make_per_virtual_layer_b1(2, 2),  # core iter 0
            _make_per_virtual_layer_b1(1, 3),  # core iter 1
            _make_per_virtual_layer_b1(2, 4),  # core iter 1
            _make_per_virtual_layer_b1(3, 5),  # coda
        ],
    )


class TestPerVirtualLayerMetrics:
    def test_fields_exist(self):
        pbm = _make_per_virtual_layer(0, 0)
        assert pbm.layer_index == 0
        assert pbm.virtual_layer_index == 0
        assert pbm.sink_scores.shape == (2, 4)
        assert pbm.residual_norms.shape == (2,)

    def test_none_fields_allowed(self):
        pbm = PerVirtualLayerMetrics(
            layer_index=0,
            virtual_layer_index=0,
            sink_scores=None,
            mixing_scores=None,
            residual_norms=torch.rand(2),
            colsum_concentrations=None,
        )
        assert pbm.sink_scores is None
        assert pbm.residual_norms.shape == (2,)

    def test_repr_summarizes_tensors(self):
        pbm = PerVirtualLayerMetrics(
            layer_index=0,
            virtual_layer_index=0,
            sink_scores=torch.arange(12).reshape(3, 4),
            mixing_scores=None,
            residual_norms=torch.arange(3),
            colsum_concentrations=torch.arange(12).reshape(3, 4),
            raw_residual=torch.arange(120).reshape(3, 5, 8),
        )

        text = repr(pbm)

        assert "PerVirtualLayerMetrics(" in text
        assert "sink_scores=Tensor(shape=(3, 4), dtype=torch.int64)" in text
        assert "raw_residual=Tensor(shape=(3, 5, 8), dtype=torch.int64)" in text
        assert "tensor(" not in text
        assert "119" not in text


class TestModelRunData:
    def test_construction(self):
        d = _make_model_run_data()
        assert d.input_ids.shape == (2, 16)
        assert len(d.per_virtual_layer_metrics) == 3
        assert d.pairwise_attn_pattern_diff_norm.shape == (3, 3, 4)
        assert d.pairwise_residual_cosine_sim.shape == (3, 3, 16)

    def test_optional_fields_default_none(self):
        d = ModelRunData(
            input_ids=torch.zeros(1, 4, dtype=torch.long),
            per_virtual_layer_metrics=[_make_per_virtual_layer(0, 0)],
        )
        assert d.pairwise_attn_pattern_diff_norm is None
        assert d.to_final_state_residual_diff_norm is None
        assert d.successive_recurrence_residual_diff_norm is None

    def test_recurrent_layer_indices_unique(self):
        """recurrent_layer_indices should be unique even when layers fire 3+ times."""
        d = ModelRunData(
            input_ids=torch.zeros(1, 2, dtype=torch.long),
            per_virtual_layer_metrics=[
                _make_per_virtual_layer(0, 0),
                _make_per_virtual_layer(0, 1),
                _make_per_virtual_layer(0, 2),
            ],
        )
        assert d.recurrent_layer_indices == [0]  # unique, not [0, 0]

    def test_get_metric_ungrouped_shape(self):
        d = _make_model_run_data_b1()  # S=3
        result = d.get_metric("sink_scores")
        assert result.shape == (3, 4)  # S=3, heads=4

    def test_get_metric_ungrouped_values(self):
        d = _make_model_run_data_b1()
        result = d.get_metric("sink_scores")
        for i, pbm in enumerate(d.per_virtual_layer_metrics):
            assert torch.equal(result[i], pbm.sink_scores.squeeze(0))

    def test_get_metric_by_layer_keys_and_shapes(self):
        d = (
            _make_model_run_data_b1()
        )  # layer 0: 1 virtual layer, layer 1: 2 virtual layers
        result = d.get_metric("sink_scores", group_by="layer")
        assert list(result.keys()) == ["L0", "L1"]
        assert result["L0"].shape == (1, 4)
        assert result["L1"].shape == (2, 4)

    def test_get_metric_by_layer_values(self):
        d = _make_model_run_data_b1()
        result = d.get_metric("sink_scores", group_by="layer")
        # L0: virtual layer at seq 0
        assert torch.equal(
            result["L0"][0], d.per_virtual_layer_metrics[0].sink_scores.squeeze(0)
        )
        # L1: virtual layers at seq 1 and 2
        assert torch.equal(
            result["L1"][0], d.per_virtual_layer_metrics[1].sink_scores.squeeze(0)
        )
        assert torch.equal(
            result["L1"][1], d.per_virtual_layer_metrics[2].sink_scores.squeeze(0)
        )

    def test_get_metric_by_layer_keys_ascending(self):
        d = _make_recurrent_model_data_b1()  # 4 layers × 2 virtual layers
        result = d.get_metric("sink_scores", group_by="layer")
        assert list(result.keys()) == ["L0", "L1", "L2", "L3"]

    def test_get_metric_by_recurrence_pure_keys(self):
        d = _make_recurrent_model_data_b1()  # pure recurrent: no prelude/coda
        result = d.get_metric("sink_scores", group_by="recurrence")
        assert list(result.keys()) == ["R0", "R1"]

    def test_get_metric_by_recurrence_pure_shapes(self):
        d = _make_recurrent_model_data_b1()
        result = d.get_metric("sink_scores", group_by="recurrence")
        assert result["R0"].shape == (4, 4)  # 4 layers, 4 heads
        assert result["R1"].shape == (4, 4)

    def test_get_metric_by_recurrence_raven_keys(self):
        d = _make_raven_model_data_b1()
        result = d.get_metric("sink_scores", group_by="recurrence")
        assert list(result.keys()) == ["Prelude", "R0", "R1", "Coda"]

    def test_get_metric_by_recurrence_raven_shapes(self):
        d = _make_raven_model_data_b1()
        result = d.get_metric("sink_scores", group_by="recurrence")
        assert result["Prelude"].shape == (1, 4)
        assert result["R0"].shape == (2, 4)
        assert result["R1"].shape == (2, 4)
        assert result["Coda"].shape == (1, 4)

    def test_get_metric_raises_on_none(self):
        d = _make_model_run_data_b1()
        d.per_virtual_layer_metrics[1] = PerVirtualLayerMetrics(
            layer_index=1,
            virtual_layer_index=1,
            sink_scores=None,
            mixing_scores=None,
            residual_norms=torch.rand(1),
            colsum_concentrations=None,
        )
        with pytest.raises(ValueError, match="sink_scores is None"):
            d.get_metric("sink_scores")

    def test_get_metric_raises_on_batch_gt_1(self):
        d = _make_model_run_data_b1()
        d.per_virtual_layer_metrics[0] = PerVirtualLayerMetrics(
            layer_index=0,
            virtual_layer_index=0,
            sink_scores=torch.rand(2, 4),  # batch size 2
            mixing_scores=None,
            residual_norms=torch.rand(2),
            colsum_concentrations=None,
        )
        with pytest.raises(ValueError, match="batch size 2"):
            d.get_metric("sink_scores")

    def test_save_load_roundtrip(self):
        original = _make_model_run_data()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)

        try:
            original.save(path)
            loaded = ModelRunData.load(path)

            assert loaded.input_ids.shape == original.input_ids.shape
            assert torch.equal(loaded.input_ids, original.input_ids)

            assert len(loaded.per_virtual_layer_metrics) == len(
                original.per_virtual_layer_metrics
            )
            for orig_pbm, load_pbm in zip(
                original.per_virtual_layer_metrics,
                loaded.per_virtual_layer_metrics,
                strict=True,
            ):
                assert orig_pbm.layer_index == load_pbm.layer_index
                assert orig_pbm.virtual_layer_index == load_pbm.virtual_layer_index
                assert torch.allclose(orig_pbm.residual_norms, load_pbm.residual_norms)
                if orig_pbm.sink_scores is not None:
                    assert torch.allclose(orig_pbm.sink_scores, load_pbm.sink_scores)

            assert torch.allclose(
                loaded.pairwise_attn_pattern_diff_norm,
                original.pairwise_attn_pattern_diff_norm,
            )
            assert torch.allclose(
                loaded.pairwise_residual_cosine_sim,
                original.pairwise_residual_cosine_sim,
            )
            assert torch.allclose(
                loaded.to_final_state_residual_diff_norm,
                original.to_final_state_residual_diff_norm,
            )
            for orig_t, load_t in zip(
                original.successive_recurrence_residual_diff_norm,
                loaded.successive_recurrence_residual_diff_norm,
                strict=True,
            ):
                assert torch.allclose(orig_t, load_t)
        finally:
            path.unlink(missing_ok=True)

    def test_repr_summarizes_nested_data(self):
        data = _make_model_run_data()

        text = repr(data)

        assert "ModelRunData(" in text
        assert "input_ids=Tensor(shape=(2, 16), dtype=torch.int64)" in text
        assert "per_virtual_layer_metrics=list(len=3)" in text
        assert "successive_recurrence_residual_diff_norm=list(len=2, items=[" in text
        assert "Tensor(shape=(0, 16), dtype=torch.float32)" in text
        assert "tensor(" not in text

    def test_save_load_with_none_fields(self):
        d = ModelRunData(
            input_ids=torch.zeros(1, 4, dtype=torch.long),
            per_virtual_layer_metrics=[_make_per_virtual_layer(0, 0)],
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)
        try:
            d.save(path)
            loaded = ModelRunData.load(path)
            assert loaded.pairwise_attn_pattern_diff_norm is None
            assert loaded.pairwise_residual_cosine_sim is None
            assert loaded.to_final_state_residual_diff_norm is None
            assert loaded.successive_recurrence_residual_diff_norm is None
        finally:
            path.unlink(missing_ok=True)


class TestFastReprs:
    def test_depth_values_repr_summarizes_tensors(self):
        values = DepthValues(
            positions=torch.arange(4),
            position_label="Depth",
            mean=torch.arange(4, dtype=torch.float32),
            std=torch.arange(4, dtype=torch.float32),
        )

        text = repr(values)

        assert text == (
            "DepthValues(positions=Tensor(shape=(4,), dtype=torch.int64), "
            "position_label='Depth', mean=Tensor(shape=(4,), dtype=torch.float32), "
            "std=Tensor(shape=(4,), dtype=torch.float32))"
        )

    def test_example_similarity_repr_summarizes_tokens_and_to_final_state_residual_cosine_sim_by_layer(
        self,
    ):
        data = ExampleResidualComparisonData(
            example_idx=2,
            token_strs=["a", "b", "c"],
            question_start=1,
            to_final_state_residual_cosine_sim_by_layer={
                4: torch.arange(12).reshape(3, 4)
            },
            to_final_state_residual_diff_norm_by_layer={4: torch.zeros(3, 4)},
        )

        text = repr(data)

        assert "token_strs=list(len=3)" in text
        assert (
            "to_final_state_residual_cosine_sim_by_layer=dict(len=1, keys=[4], values=Tensor)"
            in text
        )
        assert "tensor(" not in text

    def test_b3_eval_repr_summarizes_per_virtual_layer_metrics(self):
        data = DepthEvaluationResult(
            per_virtual_layer_metrics=[
                _make_per_virtual_layer_b1(0, 0),
                _make_per_virtual_layer_b1(0, 1),
            ],
            model_name="model",
            dataset_name="dataset",
            n_test=10,
            n_train=20,
        )

        text = repr(data)

        assert "DepthEvaluationResult(" in text
        assert "per_virtual_layer_metrics=list(len=2)" in text
        assert "model_name='model'" in text
        assert "tensor(" not in text


class TestAggregateValues:
    """Tests for the aggregate_values module-level helper."""

    # ------------------------------------------------------------------
    # group_by=None (ungrouped / virtual layer depth)
    # ------------------------------------------------------------------

    def test_group_by_none_key(self):
        d = _make_recurrent_model_data_b1()
        result = aggregate_values([d, d], "sink_scores")
        assert list(result.keys()) == ["all"]

    def test_group_by_none_shape(self):
        d = _make_recurrent_model_data_b1()  # S=8 virtual layers
        result = aggregate_values([d, d], "sink_scores")
        assert result["all"].mean.shape == (8,)
        assert result["all"].std.shape == (8,)
        assert result["all"].positions.shape == (8,)

    def test_group_by_none_std_zero_identical_models(self):
        d = _make_recurrent_model_data_b1()
        result = aggregate_values([d, d], "sink_scores")
        assert torch.allclose(result["all"].std, torch.zeros(8))

    def test_group_by_none_mean_correct(self):
        d = _make_model_run_data_b1()  # S=3
        result = aggregate_values([d], "sink_scores")
        expected = d.get_metric("sink_scores").float().mean(dim=-1)
        assert torch.allclose(result["all"].mean, expected)

    # ------------------------------------------------------------------
    # group_by="layer"
    # ------------------------------------------------------------------

    def test_group_by_layer_keys(self):
        d = _make_model_run_data_b1()  # L=2 layers
        result = aggregate_values([d, d], "sink_scores", group_by="layer")
        assert list(result.keys()) == ["L0", "L1"]

    def test_group_by_layer_shapes(self):
        d = (
            _make_model_run_data_b1()
        )  # layer 0: 1 virtual layer, layer 1: 2 virtual layers
        result = aggregate_values([d, d], "sink_scores", group_by="layer")
        assert result["L0"].mean.shape == (1,)
        assert result["L1"].mean.shape == (2,)

    def test_group_by_layer_std_zero_identical_models(self):
        d = _make_model_run_data_b1()
        result = aggregate_values([d, d], "sink_scores", group_by="layer")
        for dv in result.values():
            assert torch.allclose(dv.std, torch.zeros_like(dv.std))

    def test_group_by_layer_keys_ascending(self):
        # Keys must be sorted by layer index regardless of execution order
        d = _make_recurrent_model_data_b1()  # 4 layers × 2 virtual layers
        result = aggregate_values([d], "sink_scores", group_by="layer")
        assert list(result.keys()) == ["L0", "L1", "L2", "L3"]

    # ------------------------------------------------------------------
    # group_by="recurrence"
    # ------------------------------------------------------------------

    def test_group_by_recurrence_pure_keys(self):
        d = _make_recurrent_model_data_b1()  # pure recurrent, no prelude/coda
        result = aggregate_values([d, d], "sink_scores", group_by="recurrence")
        assert list(result.keys()) == ["R0", "R1"]

    def test_group_by_recurrence_pure_shapes(self):
        d = _make_recurrent_model_data_b1()
        result = aggregate_values([d, d], "sink_scores", group_by="recurrence")
        assert result["R0"].mean.shape == (4,)
        assert result["R1"].mean.shape == (4,)

    def test_group_by_recurrence_raven_keys(self):
        # Raven: prelude layer 0, core [1,2]×2, coda layer 3 — auto-detected
        d = _make_raven_model_data_b1()
        result = aggregate_values([d, d], "sink_scores", group_by="recurrence")
        assert list(result.keys()) == ["Prelude", "R0", "R1", "Coda"]

    def test_group_by_recurrence_raven_shapes(self):
        d = _make_raven_model_data_b1()
        result = aggregate_values([d, d], "sink_scores", group_by="recurrence")
        assert result["Prelude"].mean.shape == (1,)
        assert result["R0"].mean.shape == (2,)
        assert result["R1"].mean.shape == (2,)
        assert result["Coda"].mean.shape == (1,)

    def test_group_by_recurrence_std_zero_identical_models(self):
        d = _make_raven_model_data_b1()
        result = aggregate_values([d, d], "sink_scores", group_by="recurrence")
        for dv in result.values():
            assert torch.allclose(dv.std, torch.zeros_like(dv.std))

    # ------------------------------------------------------------------
    # scale_by
    # ------------------------------------------------------------------

    def test_scale_by_none_raw_positions(self):
        d = _make_recurrent_model_data_b1()  # S=8, seq indices 0..7
        result = aggregate_values([d], "sink_scores", scale_by=None)
        assert torch.allclose(
            result["all"].positions, torch.arange(8, dtype=torch.float)
        )

    def test_scale_by_virtual_layer_range(self):
        d = _make_recurrent_model_data_b1()
        result = aggregate_values([d], "sink_scores", scale_by="virtual_layer")
        assert result["all"].positions.min().item() == pytest.approx(0.0)
        assert result["all"].positions.max().item() == pytest.approx(100.0)

    def test_scale_by_virtual_layer_global_denominator(self):
        # With a global denominator, positions in different groups are NOT
        # independently scaled to [0, 100].
        d = _make_model_run_data_b1()  # layer 0 at seq 0, layer 1 at seq 1 and 2
        result = aggregate_values(
            [d], "sink_scores", group_by="layer", scale_by="virtual_layer"
        )
        # S=3, max seq = 2 → L0 → [0/2*100=0], L1 → [1/2*100=50, 2/2*100=100]
        assert torch.allclose(result["L0"].positions, torch.tensor([0.0]))
        assert torch.allclose(result["L1"].positions, torch.tensor([50.0, 100.0]))

    def test_scale_by_layer_range(self):
        d = _make_recurrent_model_data_b1()  # layers 0,1,2,3 → max=3
        result = aggregate_values([d], "sink_scores", scale_by="layer")
        assert result["all"].positions.min().item() == pytest.approx(0.0)
        assert result["all"].positions.max().item() == pytest.approx(100.0)

    def test_scale_by_layer_values(self):
        # In virtual layer order: 0,1,2,3,0,1,2,3 → layer indices / 3 * 100
        d = _make_recurrent_model_data_b1()
        result = aggregate_values([d], "sink_scores", scale_by="layer")
        expected = (
            torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.float) / 3.0 * 100.0
        )
        assert torch.allclose(result["all"].positions, expected)

    # ------------------------------------------------------------------
    # position_label
    # ------------------------------------------------------------------

    def test_position_label_none(self):
        d = _make_recurrent_model_data_b1()
        result = aggregate_values([d], "sink_scores", scale_by=None)
        assert result["all"].position_label == "Virtual layer"

    def test_position_label_virtual_layer(self):
        d = _make_recurrent_model_data_b1()
        result = aggregate_values([d], "sink_scores", scale_by="virtual_layer")
        assert result["all"].position_label == "Virtual layer depth (%)"

    def test_position_label_layer(self):
        d = _make_recurrent_model_data_b1()
        result = aggregate_values([d], "sink_scores", scale_by="layer")
        assert result["all"].position_label == "Layer depth (%)"

    def test_position_label_consistent_across_groups(self):
        d = _make_raven_model_data_b1()
        result = aggregate_values(
            [d], "sink_scores", group_by="recurrence", scale_by="virtual_layer"
        )
        labels = {dv.position_label for dv in result.values()}
        assert labels == {"Virtual layer depth (%)"}

    # ------------------------------------------------------------------
    # return type
    # ------------------------------------------------------------------

    def test_returns_depthvalues(self):
        d = _make_model_run_data_b1()
        result = aggregate_values([d], "sink_scores", group_by="layer")
        for dv in result.values():
            assert isinstance(dv, DepthValues)


# ---------------------------------------------------------------------------
# Helpers for limiting-behavior tests
# ---------------------------------------------------------------------------

# Threshold used throughout these tests
_THRESH = 0.01


def _make_norm_tensor(patterns: list[list[float]], T: int, L: int) -> torch.Tensor:
    """Build a (S, L, T) to_final_state_residual_diff_norm tensor.

    ``patterns`` has one entry per virtual layer (length S). Each entry is a list of
    T values for layer 0 (column 0). All other layers are set to 1.0 (far from
    final state, irrelevant to the test).
    """
    S = len(patterns)
    tensor = torch.ones(S, L, T)
    for s, row in enumerate(patterns):
        for t, v in enumerate(row):
            tensor[s, 0, t] = v
    return tensor


def _make_sim_tensor(patterns: list[list[float]], T: int, L: int) -> torch.Tensor:
    """Build a (S, L, T) to_final_state_residual_cosine_sim tensor (cosine sim, range [-1,1])."""
    S = len(patterns)
    tensor = torch.zeros(S, L, T)  # far from 1.0 by default
    for s, row in enumerate(patterns):
        for t, v in enumerate(row):
            tensor[s, 0, t] = v
    return tensor


def _single_layer_mrd(
    n_virtual_layers: int,
    norm_tensor: torch.Tensor | None = None,
    sim_tensor: torch.Tensor | None = None,
) -> ModelRunData:
    """ModelRunData with a single recurrent layer (layer_index=0) virtual layer n_virtual_layers times."""
    pbms = [_make_per_virtual_layer(0, s) for s in range(n_virtual_layers)]
    return ModelRunData(
        input_ids=torch.zeros(
            1, norm_tensor.shape[2] if norm_tensor is not None else 1, dtype=torch.long
        ),
        per_virtual_layer_metrics=pbms,
        to_final_state_residual_diff_norm=norm_tensor,
        to_final_state_residual_cosine_sim=sim_tensor,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassifyLimitingBehavior:
    """Tests for ModelRunData.classify_limiting_behavior_per_token."""

    # ------------------------------------------------------------------
    # fixed point
    # ------------------------------------------------------------------

    def test_fixed_point_all_tokens(self):
        # 5 virtual layers; tokens 0-2 converge from the very first virtual layer
        T = 3
        # pattern: all virtual layers very close to 0 (final virtual layer included)
        patterns = [
            [0.001, 0.001, 0.001],
            [0.001, 0.001, 0.001],
            [0.001, 0.001, 0.001],
            [0.001, 0.001, 0.001],
            [0.0, 0.0, 0.0],  # final virtual layer, trivially excluded
        ]
        norm = _make_norm_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=5, norm_tensor=norm)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH, metric="diff_norm"
        )
        assert 0 in result
        assert len(result[0]) == T
        for b in result[0]:
            assert isinstance(b, FixedPoint)

    def test_fixed_point_use_sim(self):
        # Cosine-sim version: values close to 1.0 = fixed point
        T = 2
        patterns = [
            [0.999, 0.999],
            [0.999, 0.999],
            [1.0, 1.0],  # final virtual layer
        ]
        sim = _make_sim_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=3, sim_tensor=sim)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH, metric="cosine_sim"
        )
        for b in result[0]:
            assert isinstance(b, FixedPoint)

    # ------------------------------------------------------------------
    # slider
    # ------------------------------------------------------------------

    def test_slider_all_tokens(self):
        # Norm ramps down from 1.0 to 0 over 8 virtual layers → large negative gradient
        # → convergence_gradient (flipped) >> threshold → Slider
        T = 3
        n_analysis = 8
        patterns = [[1.0 - i / (n_analysis - 1)] * T for i in range(n_analysis)]
        patterns.append([0.0] * T)  # final virtual layer
        norm = _make_norm_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=n_analysis + 1, norm_tensor=norm)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH, metric="diff_norm"
        )
        for b in result[0]:
            assert isinstance(b, Slider)

    def test_slider_flat_is_unknown(self):
        # Flat signal far from 0: gradient ≈ 0, never close → Unknown
        T = 3
        patterns = [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],  # final virtual layer
        ]
        norm = _make_norm_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=4, norm_tensor=norm)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH, metric="diff_norm"
        )
        for b in result[0]:
            assert isinstance(b, Unknown)

    def test_slider_use_sim(self):
        # Sim ramps up from 0.0 to 1.0 → large positive gradient → Slider
        T = 2
        n_analysis = 8
        patterns = [[i / (n_analysis - 1)] * T for i in range(n_analysis)]
        patterns.append([1.0] * T)  # final virtual layer
        sim = _make_sim_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=n_analysis + 1, sim_tensor=sim)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH, metric="cosine_sim"
        )
        for b in result[0]:
            assert isinstance(b, Slider)

    # ------------------------------------------------------------------
    # orbit
    # ------------------------------------------------------------------

    def test_orbit_period2(self):
        # Alternating: close, far, close, far, … + final=0
        T = 1
        n_analysis = 8  # virtual layers before the final
        patterns = [[0.001] if i % 2 == 0 else [1.0] for i in range(n_analysis)]
        patterns.append([0.0])  # final virtual layer
        norm = _make_norm_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=len(patterns), norm_tensor=norm)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH, metric="diff_norm"
        )
        b = result[0][0]
        assert isinstance(b, Orbit)

    def test_orbit_frequency_period4(self):
        """FFT should detect period-4 orbit (frequency ≈ 0.25 cycles/loop).

        Uses a sinusoidal similarity signal so the fundamental clearly dominates
        after Hann windowing (square waves have harmonics that can confuse the peak
        bin after windowing).
        """
        T = 1
        period = 4
        n_analysis = 16
        # Cosine-sim signal oscillating sinusoidally between 0.3 and 1.0
        # so the fundamental at 1/period dominates the windowed FFT.
        patterns = [
            [0.65 + 0.35 * math.cos(2 * math.pi * i / period)]
            for i in range(n_analysis)
        ]
        patterns.append([1.0])  # final virtual layer
        sim = _make_sim_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=len(patterns), sim_tensor=sim)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH, metric="cosine_sim"
        )
        b = result[0][0]
        assert isinstance(b, Orbit)
        expected_freq = 1.0 / period
        assert math.isclose(b.frequency, expected_freq, abs_tol=1.0 / n_analysis + 1e-9)

    def test_orbit_use_sim(self):
        # Sim version: alternating between close-to-1 and close-to-0
        T = 1
        n_analysis = 8
        patterns = [[0.999] if i % 2 == 0 else [0.0] for i in range(n_analysis)]
        patterns.append([1.0])  # final virtual layer
        sim = _make_sim_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=len(patterns), sim_tensor=sim)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH, metric="cosine_sim"
        )
        assert isinstance(result[0][0], Orbit)

    # ------------------------------------------------------------------
    # mixed tokens
    # ------------------------------------------------------------------

    def test_mixed_token_behaviors(self):
        """Different tokens in the same layer can have different behaviors."""
        T = 3
        n_analysis = 8
        patterns = []
        for i in range(n_analysis):
            row = [
                0.001,  # token 0: fixed point (always close to 0)
                1.0 - i / (n_analysis - 1),  # token 1: slider (ramping down)
                0.001 if i % 2 == 0 else 1.0,  # token 2: orbit period 2
            ]
            patterns.append(row)
        patterns.append([0.0, 0.0, 0.0])  # final virtual layer
        norm = _make_norm_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=len(patterns), norm_tensor=norm)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH, metric="diff_norm"
        )
        assert isinstance(result[0][0], FixedPoint)
        assert isinstance(result[0][1], Slider)
        assert isinstance(result[0][2], Orbit)

    # ------------------------------------------------------------------
    # recurrence_lookback_distance clipping
    # ------------------------------------------------------------------

    def test_lookback_clips_to_available(self):
        """lookback_distance larger than available virtual layers uses all available."""
        T = 1
        # 3 analysis virtual layers + 1 final; lookback=100 should just use all 3
        patterns = [[0.001], [0.001], [0.001], [0.0]]
        norm = _make_norm_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=4, norm_tensor=norm)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH,
            recurrence_lookback_distance=100,
            metric="diff_norm",
        )
        assert isinstance(result[0][0], FixedPoint)

    def test_lookback_limits_window(self):
        """lookback_distance=2 should only look at the last 2 analysis virtual layers."""
        T = 1
        # Early virtual layers are sliders but last 2 analysis virtual layers are close → fixed point
        patterns = [[1.0], [1.0], [1.0], [0.001], [0.001], [0.0]]
        norm = _make_norm_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=6, norm_tensor=norm)
        # Without lookback limit this would be Orbit (mix of close/far)
        # With lookback=2 we only see the last 2 analysis virtual layers (both close)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH,
            recurrence_lookback_distance=2,
            metric="diff_norm",
        )
        assert isinstance(result[0][0], FixedPoint)

    # ------------------------------------------------------------------
    # return type and structure
    # ------------------------------------------------------------------

    def test_returns_dict_keyed_by_layer_index(self):
        T = 2
        patterns = [[0.001, 0.001], [0.001, 0.001], [0.0, 0.0]]
        norm = _make_norm_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=3, norm_tensor=norm)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH, metric="diff_norm"
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {0}
        assert len(result[0]) == T

    def test_result_elements_are_limiting_behavior(self):
        T = 1
        patterns = [[1.0], [0.0]]
        norm = _make_norm_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=2, norm_tensor=norm)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH, metric="diff_norm"
        )
        for b in result[0]:
            assert isinstance(b, LimitingBehavior)

    def test_empty_for_non_recurrent_model(self):
        """Non-recurrent model (each layer fires once) returns empty dict."""
        pbms = [_make_per_virtual_layer(0, 0), _make_per_virtual_layer(1, 1)]
        norm = torch.rand(2, 2, 4)
        mrd = ModelRunData(
            input_ids=torch.zeros(1, 4, dtype=torch.long),
            per_virtual_layer_metrics=pbms,
            to_final_state_residual_diff_norm=norm,
        )
        result = mrd.classify_limiting_behavior_per_token(metric="diff_norm")
        assert result == {}

    def test_raises_when_norm_is_none(self):
        T = 2
        pbms = [_make_per_virtual_layer(0, 0), _make_per_virtual_layer(0, 1)]
        mrd = ModelRunData(
            input_ids=torch.zeros(1, T, dtype=torch.long),
            per_virtual_layer_metrics=pbms,
        )
        with pytest.raises(
            ValueError, match="to_final_state_residual_diff_norm is None"
        ):
            mrd.classify_limiting_behavior_per_token(metric="diff_norm")

    def test_raises_when_sim_is_none(self):
        T = 2
        pbms = [_make_per_virtual_layer(0, 0), _make_per_virtual_layer(0, 1)]
        mrd = ModelRunData(
            input_ids=torch.zeros(1, T, dtype=torch.long),
            per_virtual_layer_metrics=pbms,
        )
        with pytest.raises(
            ValueError, match="to_final_state_residual_cosine_sim is None"
        ):
            mrd.classify_limiting_behavior_per_token(metric="cosine_sim")

    # ------------------------------------------------------------------
    # Regression: noisy FixedPoint must not be classified as Orbit
    # ------------------------------------------------------------------

    def test_noisy_fixed_point_not_orbit(self):
        """7/8 values close, 1/8 slightly above threshold → FixedPoint, not Orbit."""
        T = 1
        # 7 virtual layers clearly close, 1 virtual layer just above threshold (noisy FixedPoint)
        patterns = [
            [0.001],
            [0.001],
            [0.001],
            [0.001],
            [0.001],
            [0.001],
            [0.001],
            [0.015],  # last analysis virtual layer: slightly > 0.01
            [0.0],  # final virtual layer
        ]
        norm = _make_norm_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=len(patterns), norm_tensor=norm)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH, metric="diff_norm"
        )
        assert isinstance(result[0][0], FixedPoint), (
            "Noisy FixedPoint (7/8 close) must not be classified as Orbit"
        )

    # ------------------------------------------------------------------
    # Regression: monotonic Slider must not be classified as Orbit
    # ------------------------------------------------------------------

    def test_slider_ramp_not_orbit(self):
        """Linear similarity ramp 0→0.999 over 16 steps: last value crosses
        threshold, but it is still a Slider (not an Orbit)."""
        T = 1
        n_analysis = 16
        # cosine-sim ramp from 0.0 to 0.999 (last analysis value >= 1-0.01=0.99 → close)
        ramp = [i / (n_analysis - 1) * 0.999 for i in range(n_analysis)]
        patterns = [[v] for v in ramp]
        patterns.append([1.0])  # final virtual layer
        sim = _make_sim_tensor(patterns, T=T, L=1)
        mrd = _single_layer_mrd(n_virtual_layers=len(patterns), sim_tensor=sim)
        result = mrd.classify_limiting_behavior_per_token(
            closeness_threshold=_THRESH, metric="cosine_sim"
        )
        assert isinstance(result[0][0], Slider), (
            "Monotonic similarity ramp must not be classified as Orbit"
        )


# ---------------------------------------------------------------------------
# TestGetPcaResidualTrajectories
# ---------------------------------------------------------------------------

# Raven-style fixture with raw_residual populated:
#   prelude [L0], core [L1, L2] × 2, coda [L3]
#   Execution order: 0,1,2,1,2,3  (S=6, T=5, D=8)

_T_PCA = 5
_D_PCA = 8


def _make_pbm_with_residual(
    layer_index: int, virtual_layer_index: int
) -> PerVirtualLayerMetrics:
    return PerVirtualLayerMetrics(
        layer_index=layer_index,
        virtual_layer_index=virtual_layer_index,
        sink_scores=torch.rand(1, 4),
        mixing_scores=torch.rand(1, 4),
        residual_norms=torch.rand(1),
        colsum_concentrations=torch.rand(1, 4),
        raw_residual=torch.randn(1, _T_PCA, _D_PCA),
    )


def _make_raven_data_with_residual() -> ModelRunData:
    """Raven-style: prelude [L0], core [L1,L2]×2, coda [L3]. S=6."""
    return ModelRunData(
        input_ids=torch.randint(0, 100, (1, _T_PCA)),
        per_virtual_layer_metrics=[
            _make_pbm_with_residual(0, 0),  # prelude
            _make_pbm_with_residual(1, 1),  # core R0
            _make_pbm_with_residual(2, 2),  # core R0
            _make_pbm_with_residual(1, 3),  # core R1
            _make_pbm_with_residual(2, 4),  # core R1
            _make_pbm_with_residual(3, 5),  # coda
        ],
    )


class TestGetPcaResidualTrajectories:
    def test_ungrouped_shape(self):
        mrd = _make_raven_data_with_residual()
        result = mrd.get_pca_residual_trajectories(num_dimensions=2)
        assert result.shape == (6, _T_PCA, 2)

    def test_num_dimensions(self):
        mrd = _make_raven_data_with_residual()
        result = mrd.get_pca_residual_trajectories(num_dimensions=3)
        assert result.shape == (6, _T_PCA, 3)

    def test_grouped_by_recurrence_keys_and_shapes(self):
        mrd = _make_raven_data_with_residual()
        result = mrd.get_pca_residual_trajectories(group_by="recurrence")
        assert list(result.keys()) == ["Prelude", "R0", "R1", "Coda"]
        assert result["Prelude"].shape == (1, _T_PCA, 2)  # 1 prelude virtual layer
        assert result["R0"].shape == (2, _T_PCA, 2)  # 2 core blocks per recurrence
        assert result["R1"].shape == (2, _T_PCA, 2)
        assert result["Coda"].shape == (1, _T_PCA, 2)  # 1 coda virtual layer

    def test_grouped_by_layer_keys_and_shapes(self):
        mrd = _make_raven_data_with_residual()
        result = mrd.get_pca_residual_trajectories(group_by="layer")
        assert list(result.keys()) == ["L0", "L1", "L2", "L3"]
        assert result["L0"].shape == (1, _T_PCA, 2)  # 1 virtual layer
        assert result["L1"].shape == (2, _T_PCA, 2)  # 2 virtual layers
        assert result["L2"].shape == (2, _T_PCA, 2)
        assert result["L3"].shape == (1, _T_PCA, 2)

    def test_separate_across_tokens_shape(self):
        mrd = _make_raven_data_with_residual()
        result = mrd.get_pca_residual_trajectories(separate_across_tokens=True)
        assert result.shape == (6, _T_PCA, 2)

    def test_separate_across_tokens_grouped(self):
        mrd = _make_raven_data_with_residual()
        result = mrd.get_pca_residual_trajectories(
            separate_across_tokens=True, group_by="recurrence"
        )
        assert list(result.keys()) == ["Prelude", "R0", "R1", "Coda"]
        assert result["R0"].shape == (2, _T_PCA, 2)

    def test_raises_if_raw_residual_none(self):
        mrd = _make_raven_model_data_b1()  # no raw_residual set
        with pytest.raises(ValueError, match="raw_residual"):
            mrd.get_pca_residual_trajectories()


def test_depth_result_round_trip_preserves_virtual_layer_coordinates(tmp_path):
    result = DepthEvaluationResult(
        per_virtual_layer_metrics=[
            DepthAccuracy(
                layer_index=layer,
                virtual_layer_index=virtual_layer,
                logitlens_accuracy=torch.tensor([virtual_layer / 10]),
                logitlens_norm_accuracy=torch.tensor([virtual_layer / 20]),
                probe_accuracy=torch.tensor([virtual_layer / 30]),
            )
            for virtual_layer, layer in enumerate([0, 1, 0, 1])
        ],
        model_name="recurrent",
        dataset_name="local",
        n_test=10,
        n_train=20,
    )
    path = tmp_path / "depth.pt"
    result.save(path)
    stored = torch.load(path, weights_only=True)
    assert [m["virtual_layer_index"] for m in stored["per_virtual_layer_metrics"]] == [
        0,
        1,
        2,
        3,
    ]
    restored = DepthEvaluationResult.load(path)
    grouped = restored.get_metric("probe_accuracy", group_by="layer")
    torch.testing.assert_close(grouped["L0"], torch.tensor([0.0, 2 / 30]))
    torch.testing.assert_close(grouped["L1"], torch.tensor([1 / 30, 3 / 30]))
    assert restored.recurrent_layer_indices == [0, 1]
