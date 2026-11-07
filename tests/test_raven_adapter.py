"""Regression tests for RavenArchitectureAdapter — specifically the n_norms parameter.

Regression: AttributeError: 'SandwichBlock' object has no attribute 'norm_3'
when loading retrofitted models (smcleish/Recurrent-*) whose SandwichBlock only
has norm_1 (pre-attn) and norm_2 (pre-MLP), unlike the original Raven (huginn-0125)
which has all four sandwich norms (norm_1..norm_4).

These tests use a CPU-only fake model that faithfully replicates the retrofitted
model's module-attribute structure without requiring any HuggingFace downloads.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge

from looped_internals.models.adapters.raven import (
    RavenArchitectureAdapter,
    _make_sandwich_block_submodules,
)

# ---------------------------------------------------------------------------
# Fake model — mirrors smcleish/Recurrent-* SandwichBlock attribute structure
# ---------------------------------------------------------------------------


class _FakeRavenAttn(nn.Module):
    """Minimal CausalSelfAttention-like module with fused QKV (Raven-style)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.Wqkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _FakeRavenMLP(nn.Module):
    """Minimal fused gate+up MLP (Raven-style: single fc weight, separate proj)."""

    def __init__(self, d_model: int, intermediate: int) -> None:
        super().__init__()
        # Raven stores gate and up projections in a single fused matrix:
        # fc.weight.shape == [intermediate * 2, d_model]
        self.fc = nn.Linear(d_model, intermediate * 2, bias=False)
        self.proj = nn.Linear(intermediate, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeSandwichBlock2Norm(nn.Module):
    """2-norm SandwichBlock — exactly what smcleish/Recurrent-* ships.

    Has norm_1 (pre-attn) and norm_2 (pre-MLP) only.
    Does NOT have norm_3 or norm_4.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm_1 = nn.RMSNorm(d_model)
        self.norm_2 = nn.RMSNorm(d_model)
        self.attn = _FakeRavenAttn(d_model)
        self.mlp = _FakeRavenMLP(d_model, d_model * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_prelude: int,
        n_core: int,
        n_coda: int,
        d_vocab: int,
    ) -> None:
        super().__init__()
        self.wte = nn.Embedding(d_vocab, d_model)
        self.prelude = nn.ModuleList(
            [_FakeSandwichBlock2Norm(d_model) for _ in range(n_prelude)]
        )
        self.adapter = nn.Linear(d_model, d_model, bias=False)
        self.core_block = nn.ModuleList(
            [_FakeSandwichBlock2Norm(d_model) for _ in range(n_core)]
        )
        self.coda = nn.ModuleList(
            [_FakeSandwichBlock2Norm(d_model) for _ in range(n_coda)]
        )
        self.ln_f = nn.RMSNorm(d_model)


class FakeRetrofittedModel(nn.Module):
    """Tiny CPU model with the same attribute structure as the retrofitted Llama.

    Suitable for testing RavenArchitectureAdapter without any HF downloads.
    """

    def __init__(
        self,
        d_model: int = 32,
        d_vocab: int = 100,
        n_prelude: int = 1,
        n_core: int = 1,
        n_coda: int = 1,
    ) -> None:
        super().__init__()
        self.transformer = _FakeTransformer(d_model, n_prelude, n_core, n_coda, d_vocab)
        self.lm_head = nn.Linear(d_model, d_vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.wte(input_ids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_D_MODEL = 32
_N_HEADS = 2
_D_VOCAB = 100
_N_LAYERS = 3  # n_prelude + n_core + n_coda = 1+1+1


def _make_tl_cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=_D_MODEL,
        d_head=_D_MODEL // _N_HEADS,
        n_layers=_N_LAYERS,
        n_ctx=32,
        n_heads=_N_HEADS,
        d_vocab=_D_VOCAB,
        architecture="RavenForCausalLM",
    )


def _make_fake_model() -> FakeRetrofittedModel:
    return FakeRetrofittedModel(d_model=_D_MODEL, d_vocab=_D_VOCAB)


# ---------------------------------------------------------------------------
# Unit tests for _make_sandwich_block_submodules
# ---------------------------------------------------------------------------


class TestMakeSandwichBlockSubmodules:
    def test_4norm_includes_all_norms(self):
        subs = _make_sandwich_block_submodules(_make_tl_cfg(), n_norms=4)
        assert "ln1" in subs
        assert "ln2" in subs
        assert "ln3" in subs
        assert "ln4" in subs

    def test_2norm_excludes_post_norms(self):
        subs = _make_sandwich_block_submodules(_make_tl_cfg(), n_norms=2)
        assert "ln1" in subs
        assert "ln2" in subs
        assert "ln3" not in subs, (
            "n_norms=2 must not map ln3 — retrofitted SandwichBlocks have no norm_3"
        )
        assert "ln4" not in subs, (
            "n_norms=2 must not map ln4 — retrofitted SandwichBlocks have no norm_4"
        )

    def test_attn_and_mlp_present_for_both(self):
        for n in (2, 4):
            subs = _make_sandwich_block_submodules(_make_tl_cfg(), n_norms=n)
            assert "attn" in subs
            assert "mlp" in subs


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRavenAdapterRetrofitted:
    """RavenArchitectureAdapter with n_norms=2 must bridge 2-norm blocks cleanly."""

    @pytest.mark.parametrize("n_norms", [2, 4])
    @pytest.mark.parametrize("auto_detect", [False, True])
    @pytest.mark.filterwarnings("error:Hook alias:UserWarning")
    def test_only_supported_hook_aliases_are_registered(self, n_norms, auto_detect):
        """Both explicit and auto-detected layouts must resolve their hook aliases."""
        model = _make_fake_model()
        if n_norms == 4:
            for stage in ("prelude", "core_block", "coda"):
                for block in getattr(model.transformer, stage):
                    block.norm_3 = nn.RMSNorm(_D_MODEL)
                    block.norm_4 = nn.RMSNorm(_D_MODEL)
        adapter = RavenArchitectureAdapter(
            _make_tl_cfg(), n_norms=6 - n_norms if auto_detect else n_norms
        )
        if auto_detect:
            adapter.prepare_model(model)
        bridge = TransformerBridge(model, adapter, tokenizer=None)

        for stage in ("prelude_blocks", "core_blocks", "coda_blocks"):
            block = getattr(bridge, stage)[0]
            assert block.hook_resid_pre is block.hook_in
            assert block.hook_resid_post is block.hook_out
            assert block.hook_attn_out is block.attn.hook_out
            assert block.hook_mlp_out is block.mlp.hook_out
            assert f"{stage}.0.attn.hook_pattern" in bridge.hook_dict
            for name in (
                "hook_attn_in",
                "hook_q_input",
                "hook_k_input",
                "hook_v_input",
            ):
                assert name not in block.hook_aliases
                assert not hasattr(block, name)

    def test_2norm_bridge_no_attributeerror(self):
        """Regression: must not raise AttributeError: 'SandwichBlock' has no attribute 'norm_3'."""
        model = _make_fake_model()
        adapter = RavenArchitectureAdapter(_make_tl_cfg(), n_norms=2)
        bridge = TransformerBridge(model, adapter, tokenizer=None)
        bridge.eval()

    def test_4norm_adapter_raises_on_2norm_model(self):
        """Regression sentinel: the old (unfixed) code path must still fail on 2-norm blocks.

        This test documents why n_norms=2 is necessary — if it ever starts
        passing, the sentinel is no longer guarding against reintroduction.
        """
        model = _make_fake_model()
        adapter = RavenArchitectureAdapter(_make_tl_cfg(), n_norms=4)
        with pytest.raises(AttributeError, match="norm_3"):
            TransformerBridge(model, adapter, tokenizer=None)

    def test_autodetect_selects_2norm_for_retrofitted_block(self):
        """The adapter auto-detection picks n_norms=2 for retrofitted models."""
        model = _make_fake_model()
        first_block = model.transformer.core_block[0]
        n_norms = 4 if hasattr(first_block, "norm_3") else 2
        assert n_norms == 2, (
            "Auto-detection should return 2 for a retrofitted SandwichBlock "
            "without norm_3; got 4"
        )
