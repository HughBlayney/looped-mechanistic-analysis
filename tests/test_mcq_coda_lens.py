"""Coda-lens choice scoring shares activations and preserves causal inputs."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from looped_internals.evaluation.datasets import MCQExample
from looped_internals.structure import LayerGroup, ModelLayerStructure


def test_coda_scoring_reuses_cache_trims_padding_and_omits_observed_coda(monkeypatch):
    module = importlib.import_module("looped_internals.evaluation.logit_lens")
    coda_module = importlib.import_module("looped_internals.evaluation.coda_lens")
    structure = ModelLayerStructure(
        [
            LayerGroup("prelude", 1, False),
            LayerGroup("core", 2, True),
            LayerGroup("coda", 2, False),
        ]
    )
    layout = list(structure.virtual_layers(2))
    forwards, coda_inputs = [], []

    class Tokenizer:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return ([1] if add_special_tokens else []) + [
                {"a": 2, "b": 3, "c": 4, "d": 5}[c] for c in text
            ]

    def run_with_cache(input_ids, **kwargs):
        forwards.append(input_ids.clone())
        return None, {
            layer.residual_key: input_ids.float().unsqueeze(-1) for layer in layout
        }

    def logit_lens(bridge, hidden):
        logits = torch.zeros(*hidden.shape[:2], 6)
        logits[0, 1, 3] = 10  # Ordinary readout prefers the first option.
        return logits

    def coda_lens(bridge, hidden, *, input_ids, from_core):
        coda_inputs.append(hidden.clone())
        assert hidden.shape[0] == 1
        assert not (hidden == 0).any()  # The right-padding must never reach the coda.
        torch.testing.assert_close(input_ids, hidden[..., 0].long())
        assert from_core == (len(coda_inputs) > 2)
        logits = torch.zeros(*hidden.shape[:2], 6)
        if hidden.shape[1] == 4:
            logits[0, 1, 4] = logits[0, 2, 5] = 10  # Coda prefers the second option.
        return logits

    monkeypatch.setattr(module, "logit_lens", logit_lens)
    monkeypatch.setattr(coda_module, "coda_lens", coda_lens)
    bridge = SimpleNamespace(run_with_cache=run_with_cache)
    example = MCQExample("a", ["b", "cd"], 1)
    ordinary = module.score_logitlens(
        bridge, Tokenizer(), example, structure, 2, torch.device("cpu")
    )
    assert len(forwards) == 1 and not coda_inputs and ordinary[2] is None
    scored = module.score_logitlens(
        bridge,
        Tokenizer(),
        example,
        structure,
        2,
        torch.device("cpu"),
        compute_coda=True,
    )
    assert len(forwards) == 2  # One forward per call, regardless of coda readout.
    for before, after in zip(ordinary[:2], scored[:2], strict=True):
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(after, np.zeros(7))
    np.testing.assert_array_equal(scored[2][:5], np.ones(5))
    assert np.isnan(scored[2][5:]).all()
    assert [value.shape[1] for value in coda_inputs] == [3, 4] * 5
    for value in coda_inputs:
        assert value[0, :2, 0].tolist() == [1, 2]  # Keep the complete prefix.


def test_projected_readout_applies_c_before_the_coda_only(monkeypatch):
    module = importlib.import_module("looped_internals.evaluation.logit_lens")
    structure = ModelLayerStructure(
        [
            LayerGroup("prelude", 1, False),
            LayerGroup("core", 1, True),
            LayerGroup("coda", 1, False),
        ]
    )
    layout = list(structure.virtual_layers(2))
    projected_inputs = []

    class Tokenizer:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return ([1] if add_special_tokens else []) + [
                {"a": 2, "b": 3, "c": 4}[c] for c in text
            ]

    def run_with_cache(input_ids, **kwargs):
        return None, {layer.residual_key: torch.zeros(2, 2, 1) for layer in layout}

    def core_projection(hidden):
        projected_inputs.append(hidden.shape)
        return hidden + 1

    def logit_lens(bridge, hidden):
        logits = torch.zeros(*hidden.shape[:2], 5)
        # Raw states prefer option "b"; projected states prefer option "c".
        logits[0 if hidden.sum() == 0 else 1, 1, 3 if hidden.sum() == 0 else 4] = 10
        return logits

    monkeypatch.setattr(module, "logit_lens", logit_lens)
    bridge = SimpleNamespace(
        run_with_cache=run_with_cache, core_projection=core_projection
    )
    predictions = module.score_logitlens(
        bridge,
        Tokenizer(),
        MCQExample("a", ["b", "c"], 1),
        structure,
        2,
        torch.device("cpu"),
        project_pre_coda=True,
    )
    np.testing.assert_array_equal(predictions[0], np.zeros(4))
    # Prelude and both recurrences use C; the observed coda state does not.
    np.testing.assert_array_equal(predictions[3], [1, 1, 1, 0])
    assert len(projected_inputs) == 3


def test_native_huginn_coda_readout_matches_observed_final_state():
    from test_match_raven import HAS_RECPRE, _make_tiny_raven
    from transformer_lens.config import TransformerBridgeConfig
    from transformer_lens.model_bridge.bridge import TransformerBridge

    from looped_internals.evaluation.coda_lens import coda_lens
    from looped_internals.evaluation.logit_lens import logit_lens, score_logitlens
    from looped_internals.models.adapters.raven import RavenArchitectureAdapter

    if not HAS_RECPRE:
        pytest.skip("recpre package is unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = TransformerBridgeConfig(
        d_model=128,
        d_head=32,
        n_layers=4,
        n_ctx=64,
        n_heads=4,
        d_vocab=256,
        architecture="RavenForCausalLM",
    )
    bridge = TransformerBridge(
        _make_tiny_raven(), RavenArchitectureAdapter(cfg), tokenizer=None
    )
    bridge.eval()
    structure = ModelLayerStructure(
        [
            LayerGroup("prelude_blocks", 1, False),
            LayerGroup("core_blocks", 2, True),
            LayerGroup("coda_blocks", 1, False),
        ]
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        _, cache = bridge.run_with_cache(
            tokens,
            accumulate=True,
            names_filter=structure.hook_names(),
            num_steps=2,
        )
    expected = logit_lens(bridge, cache["coda_blocks.0.hook_out"])
    actual = coda_lens(bridge, cache["core_blocks.1.hook_out.iter_1"])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    class Tokenizer:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return ([1] if add_special_tokens else []) + [ord(c) % 32 + 2 for c in text]

    # Different option lengths exercise native coda attention after removing padding.
    predictions = score_logitlens(
        bridge,
        Tokenizer(),
        MCQExample("Question", ["A", "BCD"], 0),
        structure,
        2,
        torch.device("cpu"),
        {"num_steps": 2},
        compute_coda=True,
    )
    assert predictions[2][-2] == predictions[0][-1]
    assert np.isnan(predictions[2][-1])
