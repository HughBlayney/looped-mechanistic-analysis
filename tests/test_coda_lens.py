"""Coda readouts preserve native suffix semantics and full causal context."""

from types import SimpleNamespace

import pytest
import torch
from test_match_raven import _make_tiny_raven
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge

from looped_internals.evaluation import coda_hidden_states, coda_lens
from looped_internals.models.adapters.raven import RavenArchitectureAdapter


def tiny_raven(steps=2):
    model = _make_tiny_raven()
    model.config.mean_recurrence = steps
    cfg = TransformerBridgeConfig(
        d_model=128,
        d_head=32,
        n_layers=4,
        n_ctx=64,
        n_heads=4,
        d_vocab=256,
        architecture="RavenForCausalLM",
    )
    return TransformerBridge(
        model, RavenArchitectureAdapter(cfg), tokenizer=None
    ).eval()


def test_huginn_coda_lens_matches_final_prediction_and_is_causal():
    torch.set_num_threads(1)
    bridge = tiny_raven()
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    with torch.no_grad():
        output, cache = bridge.run_with_cache(
            tokens, accumulate=True, names_filter=["core_blocks.1.hook_out"]
        )
        state = cache["core_blocks.1.hook_out.iter_1"]
        original = state.clone()
        readout = coda_lens(bridge, state)
        torch.testing.assert_close(readout, output)
        torch.testing.assert_close(state, original)
        changed = state.clone()
        changed[:, -1] += 100
        torch.testing.assert_close(coda_lens(bridge, changed)[:, :-1], readout[:, :-1])
        assert readout.dtype == torch.float32 and not readout.requires_grad


def test_retrofit_coda_uses_native_rotary_positions_without_huginn_prenorm():
    """Synthetic causal blocks make a spurious pre-coda norm detectable."""
    torch.manual_seed(7)
    cfg = TransformerBridgeConfig(d_model=8, d_head=4, n_heads=2, n_layers=2, n_ctx=8)
    positions_seen = []

    def rotary(hidden, positions):
        positions_seen.append(positions.clone())
        return positions.cos(), positions.sin()

    class CausalBlock(torch.nn.Module):
        def forward(self, h, freqs, step, mask, cache):
            assert isinstance(freqs, tuple) and mask is None and cache is None
            assert step.item() in {-1, -2}
            denominator = torch.arange(1, h.shape[1] + 1).reshape(1, -1, 1)
            return h + h.cumsum(dim=1) / denominator

    bridge = SimpleNamespace(
        adapter=RavenArchitectureAdapter(cfg, n_norms=2),
        original_model=SimpleNamespace(rotary_emb=rotary),
        coda_blocks=[CausalBlock(), CausalBlock()],
        ln_final=torch.nn.RMSNorm(8),
        unembed=torch.nn.Linear(8, 11),
    )
    state = torch.randn(2, 5, 8) * 3
    denominator = torch.arange(1, 6).reshape(1, -1, 1)
    expected = state + state.cumsum(dim=1) / denominator
    expected = expected + expected.cumsum(dim=1) / denominator
    torch.testing.assert_close(coda_hidden_states(bridge, state), expected)
    torch.testing.assert_close(
        coda_lens(bridge, state), bridge.unembed(bridge.ln_final(expected))
    )
    assert all(torch.equal(p, torch.arange(5).unsqueeze(0)) for p in positions_seen)


def test_coda_lens_rejects_models_without_a_coda():
    with pytest.raises(ValueError, match="Coda lens requires"):
        coda_lens(SimpleNamespace(adapter=None), torch.zeros(1, 2, 8))
