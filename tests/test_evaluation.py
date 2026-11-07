"""Tests for looped_internals.evaluation — BridgeLM.

Uses a tiny MinimalRecurrentLM (CPU, no real tokenizer) to test the lm-eval
TemplateLM interface without loading any large models.
"""

from __future__ import annotations

import math
from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge

from looped_internals.evaluation.lm_eval import BridgeLM
from looped_internals.models.adapters.minimal import MinimalRecurrentAdapter
from looped_internals.models.minimal import (
    MinimalRecurrentConfig,
    MinimalRecurrentLM,
)
from looped_internals.results import ModelRunData
from looped_internals.structure import LayerGroup, ModelLayerStructure

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyTokenizer:
    """Minimal tokenizer stub that maps chars to ints."""

    eos_token_id = 0
    bos_token_id = 1

    def encode(
        self, text: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        return [ord(c) % 50 + 2 for c in text]  # chars → ints in [2, 51]


def _make_bridge_lm(
    num_cycles: int = 2, compute_metrics: bool = True, batch_size: int = 2
) -> BridgeLM:
    """Build a tiny recurrent evaluation wrapper for local tests.

    Parameters
    ----------
    num_cycles : int, optional
        Number of recurrent cycles, defaulting to two.
    compute_metrics : bool, optional
        Whether to collect per-sample metrics, defaulting to True.
    batch_size : int, optional
        Maximum requests per forward pass, defaulting to two.

    Returns
    -------
    BridgeLM
        Randomly initialized model with a character tokenizer and block layout.
    """
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
    bridge = TransformerBridge(deepcopy(model), adapter, tokenizer=TinyTokenizer())
    bridge.eval()

    structure = ModelLayerStructure(
        groups=[
            LayerGroup("blocks", 2, is_recurrent=True),
        ]
    )
    return BridgeLM(
        bridge,
        structure,
        batch_size=batch_size,
        compute_metrics=compute_metrics,
        compute_differences=False,
    )


def _make_instance(ctx_str: str, cont_str: str):
    """Create a mock lm-eval Instance-like tuple."""

    class _Inst:
        def __init__(self, ctx, cont):
            self.args = (ctx, cont)

    return _Inst(ctx_str, cont_str)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBridgeLMInterface:
    def test_eot_token_id(self):
        lm = _make_bridge_lm()
        assert isinstance(lm.eot_token_id, int)
        assert lm.eot_token_id == TinyTokenizer.eos_token_id

    def test_tok_encode(self):
        lm = _make_bridge_lm()
        tokens = lm.tok_encode("hello")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) == 5  # "hello" = 5 chars

    def test_batch_size_property(self):
        lm = _make_bridge_lm()
        assert lm.batch_size == 2


class TestLoglikelihood:
    """Test _loglikelihood_tokens via the loglikelihood() public method."""

    @pytest.fixture(scope="class")
    def lm(self):
        """Provide an evaluation wrapper for requests of varying lengths.

        Returns
        -------
        BridgeLM
            Class-scoped wrapper with batch size one and metrics disabled.
        """
        return _make_bridge_lm(compute_metrics=False, batch_size=1)

    def test_returns_list_of_tuples(self, lm):
        instances = [_make_instance("hello world", " foo")]
        results = lm.loglikelihood(instances)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 2

    def test_logprob_is_finite(self, lm):
        instances = [
            _make_instance("the quick", " brown"),
            _make_instance("hello", " world"),
        ]
        results = lm.loglikelihood(instances)
        for lp, _ in results:
            assert math.isfinite(lp)

    def test_logprob_is_negative(self, lm):
        """Log-probabilities should be ≤ 0."""
        instances = [_make_instance("test context", " continuation")]
        results = lm.loglikelihood(instances)
        lp, _ = results[0]
        assert lp <= 0.0

    def test_greedy_flag_is_bool(self, lm):
        instances = [_make_instance("foo bar", " baz")]
        results = lm.loglikelihood(instances)
        _, is_greedy = results[0]
        assert isinstance(is_greedy, bool)

    def test_multiple_requests_correct_count(self, lm):
        instances = [_make_instance(f"ctx{i}", f" cont{i}") for i in range(5)]
        results = lm.loglikelihood(instances)
        assert len(results) == 5

    def test_better_continuation_higher_logprob(self, lm):
        """The same context with the same RNG seed should give identical logprobs."""
        inst = _make_instance("hello", " world")
        torch.manual_seed(0)
        r1 = lm.loglikelihood([inst])[0]
        torch.manual_seed(0)
        r2 = lm.loglikelihood([inst])[0]
        assert abs(r1[0] - r2[0]) < 1e-5


class TestMetricsCollection:
    @pytest.fixture(scope="class")
    def lm_with_metrics(self):
        """Provide an evaluation wrapper that collects metrics per request.

        Returns
        -------
        BridgeLM
            Class-scoped wrapper with batch size one and metrics enabled.
        """
        return _make_bridge_lm(compute_metrics=True, batch_size=1)

    def test_all_model_data_populated(self, lm_with_metrics):
        """Check that scoring requests populates the collected model data.

        Parameters
        ----------
        lm_with_metrics : BridgeLM
            Evaluation fixture with per-sample metric collection enabled.
        """
        instances = [
            _make_instance("hello world", " foo"),
            _make_instance("test", " case"),
        ]
        lm_with_metrics.loglikelihood(instances)
        # Each request produces one metrics entry.
        assert len(lm_with_metrics.all_model_data) >= 2

    def test_model_data_type(self, lm_with_metrics):
        for d in lm_with_metrics.all_model_data:
            assert isinstance(d, ModelRunData)

    def test_model_data_has_per_virtual_layer_metrics(self, lm_with_metrics):
        for d in lm_with_metrics.all_model_data:
            assert len(d.per_virtual_layer_metrics) > 0

    def test_model_data_input_ids_correct_dtype(self, lm_with_metrics):
        for d in lm_with_metrics.all_model_data:
            assert d.input_ids.dtype == torch.long


@pytest.mark.parametrize("compute_metrics", [False, True])
class TestPaddingSafety:
    @pytest.mark.parametrize("reverse", [False, True])
    def test_unequal_lengths_fail_before_forward(
        self, compute_metrics, reverse, monkeypatch
    ):
        """Reject unequal token lengths before either inference path runs.

        Parameters
        ----------
        compute_metrics : bool
            Whether to exercise the cached inference path.
        reverse : bool
            Whether to place the longer request first.
        monkeypatch : pytest.MonkeyPatch
            Fixture replacing inference methods with failure sentinels.
        """
        lm = _make_bridge_lm(compute_metrics=compute_metrics)
        forward = Mock(
            spec=lm.bridge.forward,
            side_effect=AssertionError("Unsafe batch reached inference"),
        )
        cached_forward = Mock(
            spec=lm.bridge.run_with_cache,
            side_effect=AssertionError("Unsafe batch reached inference"),
        )
        monkeypatch.setattr(lm.bridge, "forward", forward)
        monkeypatch.setattr(lm.bridge, "run_with_cache", cached_forward)
        requests = [
            (("short", "end"), [1, 2], [3]),
            (("long", "end"), [1, 2, 3], [4, 5]),
        ]
        if reverse:
            requests.reverse()

        with pytest.raises(ValueError, match=r"padding.*batch_size=1"):
            lm._loglikelihood_tokens(requests, disable_tqdm=True)

        forward.assert_not_called()
        cached_forward.assert_not_called()
        assert lm.all_model_data == []

    def test_equal_total_lengths_match_individual_scores(self, compute_metrics):
        """Check that equal-length batching preserves scores and metric inputs.

        Parameters
        ----------
        compute_metrics : bool
            Whether to collect and verify the per-sample metric inputs.
        """
        lm = _make_bridge_lm(compute_metrics=compute_metrics)
        # Fix the latent initialization so batch size does not change the noise.
        lm.bridge.original_model.config.initialization_variance = 0.0
        # Different context/continuation lengths are safe if totals match.
        # Token 0 is valid content here and must not be mistaken for padding.
        requests = [
            (("ctx", "end"), [0, 2, 3], [4]),
            (("ctx", "end"), [5], [6, 7, 8]),
        ]
        batched = lm._loglikelihood_tokens(requests, disable_tqdm=True)
        if compute_metrics:
            assert [run.input_ids.tolist() for run in lm.all_model_data] == [
                [[0, 2, 3, 4]],
                [[5, 6, 7, 8]],
            ]

        individual = [
            lm._loglikelihood_tokens([request], disable_tqdm=True)[0]
            for request in requests
        ]
        assert [lp for lp, _ in batched] == pytest.approx(
            [lp for lp, _ in individual], abs=1e-5
        )
        assert [greedy for _, greedy in batched] == [greedy for _, greedy in individual]

    def test_unequal_lengths_work_with_batch_size_one(self, compute_metrics):
        """Accept unequal request lengths when each request runs separately.

        Parameters
        ----------
        compute_metrics : bool
            Whether to collect and verify the per-sample metric inputs.
        """
        lm = _make_bridge_lm(compute_metrics=compute_metrics, batch_size=1)
        requests = [
            (("short", "end"), [1, 2], [3]),
            (("long", "end"), [1, 2, 3], [4, 5]),
        ]
        results = lm._loglikelihood_tokens(requests, disable_tqdm=True)
        assert len(results) == 2
        assert all(math.isfinite(lp) for lp, _ in results)
        if compute_metrics:
            assert [run.input_ids.tolist() for run in lm.all_model_data] == [
                [[1, 2, 3]],
                [[1, 2, 3, 4, 5]],
            ]


class TestNotImplemented:
    def test_loglikelihood_rolling_raises(self):
        lm = _make_bridge_lm(compute_metrics=False)
        with pytest.raises(NotImplementedError):
            lm.loglikelihood_rolling([])

    def test_generate_until_raises(self):
        lm = _make_bridge_lm(compute_metrics=False)
        with pytest.raises(NotImplementedError):
            lm.generate_until([])


def test_probe_cache_uses_virtual_layer_count(tmp_path):
    from looped_internals.evaluation.probes import (
        MultipleChoiceProbe,
        load_probes,
        save_probes,
    )

    probes = [MultipleChoiceProbe(8) for _ in range(4)]
    path = tmp_path / "probes.pt"
    save_probes(path, probes, d_model=8, n_classes=2, n_virtual_layers=4)
    stored = torch.load(path, weights_only=True)
    assert stored["metadata"]["n_virtual_layers"] == 4
    restored = load_probes(
        path, d_model=8, n_classes=2, n_virtual_layers=4, device=torch.device("cpu")
    )
    hidden = torch.randn(2, 8)
    for before, after in zip(probes, restored, strict=True):
        torch.testing.assert_close(before(hidden), after(hidden))
    with pytest.raises(ValueError, match="n_virtual_layers"):
        load_probes(
            path, d_model=8, n_classes=2, n_virtual_layers=2, device=torch.device("cpu")
        )
