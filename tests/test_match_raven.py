"""Integration test for Raven (Huginn) architecture adapter.

Tests that the TransformerBridge wrapping of a Raven model produces identical outputs
to the raw HuggingFace model. Requires the `recpre` package to be installed.
"""

import gc
import importlib.util
import sys
import types
from copy import deepcopy

import pytest
import torch


def _import_recpre_minimal():
    """Import only the minimal Raven config/model files from recpre.

    The recpre package's __init__.py imports modules that depend on torch internals
    which may not exist in all torch versions. We bypass __init__.py by injecting a
    stub package into sys.modules and then importing the specific submodules directly.
    """
    # Check if recpre is installed at all
    spec = importlib.util.find_spec("recpre")
    if spec is None or spec.submodule_search_locations is None:
        return None, None

    # Install a stub package to bypass the real __init__.py
    if "recpre" not in sys.modules:
        stub = types.ModuleType("recpre")
        stub.__path__ = list(spec.submodule_search_locations)
        stub.__package__ = "recpre"
        sys.modules["recpre"] = stub

    from recpre.raven_config_minimal import RavenConfig
    from recpre.raven_modeling_minimal import RavenForCausalLM

    return RavenConfig, RavenForCausalLM


try:
    RavenConfig, RavenForCausalLM = _import_recpre_minimal()
    HAS_RECPRE = RavenConfig is not None
except (ImportError, ModuleNotFoundError):
    HAS_RECPRE = False

pytestmark = pytest.mark.skipif(
    not HAS_RECPRE, reason="recpre package required for Raven tests"
)

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge

from looped_internals.capture.extraction import (
    _get_attn_pattern,
    _get_attn_pattern_iter,
)
from looped_internals.models.adapters.raven import RavenArchitectureAdapter


def _make_tiny_raven():
    """Create a tiny Raven model for testing."""
    config = RavenConfig(
        n_embd=128,
        n_heads=4,
        n_layers=4,
        block_size=64,
        vocab_size=256,
        padding_multiple=1,
        intermediate_size=256,
        bias=False,
        n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2,
        n_layers_in_coda=1,
        mean_recurrence=1,
        mean_backprop_depth=0,
        qk_bias=False,
        tie_embeddings=False,
        rope_base=10000.0,
    )
    model = RavenForCausalLM(config)
    model.eval()
    return model


class TestMatchRaven:
    @pytest.fixture(scope="class")
    def hf_model(self):
        model = _make_tiny_raven()
        yield model
        del model
        gc.collect()

    @pytest.fixture(scope="class")
    def bridge_model(self, hf_model):
        cfg = TransformerBridgeConfig(
            d_model=128,
            d_head=32,
            n_layers=4,
            n_ctx=64,
            n_heads=4,
            d_vocab=256,
            architecture="RavenForCausalLM",
        )
        adapter = RavenArchitectureAdapter(cfg)
        # We need to clone the model to avoid in-place modifications by the bridge affecting the
        # original HF model used for comparison
        cloned_hf_model = deepcopy(hf_model)
        bridge = TransformerBridge(cloned_hf_model, adapter, tokenizer=None)
        bridge.eval()
        yield bridge
        del bridge
        gc.collect()

    @pytest.fixture(scope="class")
    def sample_input(self):
        torch.manual_seed(42)
        return torch.randint(0, 256, (2, 16))

    def test_full_model_output_match(self, hf_model, bridge_model, sample_input):
        """Test that bridge forward produces identical logits to raw HF forward."""
        fixed_seed = 42
        with torch.no_grad():
            with torch.random.fork_rng():
                torch.manual_seed(fixed_seed)
                hf_output = hf_model(sample_input, num_steps=1)
            hf_logits = hf_output.logits

            with torch.random.fork_rng():
                torch.manual_seed(fixed_seed)
                bridge_logits = bridge_model(
                    sample_input, return_type="logits", num_steps=1
                )

        assert hf_logits.shape == bridge_logits.shape, (
            f"Shape mismatch: HF {hf_logits.shape} vs Bridge {bridge_logits.shape}"
        )
        assert torch.allclose(hf_logits, bridge_logits, atol=1e-4), (
            f"Max diff: {(hf_logits - bridge_logits).abs().max().item()}"
        )

    def test_full_model_output_match_recurrence(
        self, hf_model, bridge_model, sample_input
    ):
        """Test that bridge forward produces identical logits to raw HF forward."""
        fixed_seed = 42
        n_recurrences = 8
        with torch.no_grad():
            with torch.random.fork_rng():
                torch.manual_seed(fixed_seed)
                hf_output = hf_model(sample_input, num_steps=n_recurrences)
            hf_logits = hf_output.logits

            with torch.random.fork_rng():
                torch.manual_seed(fixed_seed)
                bridge_logits = bridge_model(
                    sample_input, return_type="logits", num_steps=n_recurrences
                )

        assert hf_logits.shape == bridge_logits.shape, (
            f"Shape mismatch: HF {hf_logits.shape} vs Bridge {bridge_logits.shape}"
        )
        assert torch.allclose(hf_logits, bridge_logits, atol=1e-4), (
            f"Max diff: {(hf_logits - bridge_logits).abs().max().item()}"
        )

    def test_recurrent_hook_access(self, bridge_model, sample_input):
        n_recurrences = 4
        n_core_blocks = 2  # n_layers_in_recurrent_block in _make_tiny_raven

        with torch.no_grad():
            with torch.random.fork_rng():
                torch.manual_seed(42)
                _raven_logits, raven_cache = bridge_model.run_with_cache(
                    sample_input, num_steps=n_recurrences, accumulate=True
                )

        # Every core block should have exactly n_recurrences iteration entries
        hook_suffixes = ["hook_in", "hook_out", "attn.hook_out", "mlp.hook_out"]
        for block_idx in range(n_core_blocks):
            for suffix in hook_suffixes:
                base = f"core_blocks.{block_idx}.{suffix}"
                for it in range(n_recurrences):
                    key = f"{base}.iter_{it}"
                    assert key in raven_cache, f"Missing iteration entry {key}"
                # No extra iterations beyond n_recurrences
                assert f"{base}.iter_{n_recurrences}" not in raven_cache, (
                    f"Unexpected extra iteration for {base}"
                )
                # Plain key holds the final iteration's value
                assert base in raven_cache, f"Missing plain key {base}"
                final_iter = raven_cache[f"{base}.iter_{n_recurrences - 1}"]
                assert torch.equal(raven_cache[base], final_iter), (
                    f"Plain key {base} should match the last iteration"
                )

        # get_iterations should stack all iterations with correct shape
        test_hook = "core_blocks.0.attn.hook_out"
        stacked = raven_cache.get_iterations(test_hook)
        assert stacked.shape[0] == n_recurrences, (
            f"Expected {n_recurrences} iterations, got {stacked.shape[0]}"
        )
        assert stacked.shape[1:] == raven_cache[test_hook].shape, (
            "Stacked iteration shape should match single activation shape"
        )

        # Prelude/coda blocks should NOT have iteration entries (they run once)
        assert "prelude_blocks.0.hook_out.iter_1" not in raven_cache, (
            "Prelude blocks should only fire once"
        )
        assert "coda_blocks.0.hook_out.iter_1" not in raven_cache, (
            "Coda blocks should only fire once"
        )

    def test_component_access_embed(self, bridge_model):
        """Test that we can access the embedding component through the bridge."""
        embed = bridge_model.adapter.get_component(bridge_model.original_model, "embed")
        assert embed is not None
        assert hasattr(embed, "weight")

    def test_component_access_prelude_block(self, bridge_model):
        """Test that we can access prelude block components."""
        ln1 = bridge_model.adapter.get_component(
            bridge_model.original_model, "prelude_blocks.0.ln1"
        )
        assert ln1 is not None

    def test_component_access_core_block(self, bridge_model):
        """Test that we can access core block components."""
        attn = bridge_model.adapter.get_component(
            bridge_model.original_model, "core_blocks.0.attn"
        )
        assert attn is not None

    def test_component_access_coda_block(self, bridge_model):
        """Test that we can access coda block components."""
        mlp = bridge_model.adapter.get_component(
            bridge_model.original_model, "coda_blocks.0.mlp"
        )
        assert mlp is not None

    def test_component_access_ln_final(self, bridge_model):
        """Test that we can access the final layer norm."""
        ln_final = bridge_model.adapter.get_component(
            bridge_model.original_model, "ln_final"
        )
        assert ln_final is not None

    def test_component_access_unembed(self, bridge_model):
        """Test that we can access the unembedding head."""
        unembed = bridge_model.adapter.get_component(
            bridge_model.original_model, "unembed"
        )
        assert unembed is not None
        assert hasattr(unembed, "weight")

    def test_path_translation(self, bridge_model):
        """Test that TransformerLens paths translate to correct HF paths."""
        adapter = bridge_model.adapter
        assert adapter.translate_transformer_lens_path("embed") == "transformer.wte"
        assert adapter.translate_transformer_lens_path("ln_final") == "transformer.ln_f"
        assert adapter.translate_transformer_lens_path("unembed") == "lm_head"
        assert (
            adapter.translate_transformer_lens_path("prelude_blocks.0.ln1")
            == "transformer.prelude.0.norm_1"
        )
        assert (
            adapter.translate_transformer_lens_path("core_blocks.0.attn")
            == "transformer.core_block.0.attn"
        )
        assert (
            adapter.translate_transformer_lens_path("coda_blocks.0.mlp")
            == "transformer.coda.0.mlp"
        )

    def test_output_attentions_correctness(self, hf_model, bridge_model, sample_input):
        """The monkey-patched output_attentions=True path must be numerically correct.

        Concretely:
        1. run_with_cache must capture attn.hook_pattern for every block type.
        2. hook_pattern must be 4D (B, H, T, T) with every row summing to 1
           (valid attention distribution, confirming softmax was applied).
        3. Logits from run_with_cache must agree with hf_model logits to
           atol=1e-3, confirming attention capture preserves the forward pass.
        """
        fixed_seed = 42
        n_steps = 1
        with torch.no_grad():
            with torch.random.fork_rng():
                torch.manual_seed(fixed_seed)
                hf_output = hf_model(sample_input, num_steps=n_steps)
            hf_logits = hf_output.logits

            with torch.random.fork_rng():
                torch.manual_seed(fixed_seed)
                bridge_logits, cache = bridge_model.run_with_cache(
                    sample_input, num_steps=n_steps, accumulate=True
                )

        # Capturing attention patterns must preserve the model's logits.
        assert torch.allclose(hf_logits, bridge_logits, atol=1e-3), (
            f"Logit mismatch when capturing attention: "
            f"max diff = {(hf_logits - bridge_logits).abs().max().item()}"
        )

        # hook_pattern must be present and valid for every block type.
        for prefix, n_layers in [
            ("prelude_blocks", 1),
            ("core_blocks", 2),
            ("coda_blocks", 1),
        ]:
            for block_i in range(n_layers):
                layer_key = f"{prefix}.{block_i}"
                pattern_key = f"{layer_key}.attn.hook_pattern"
                assert pattern_key in cache.cache_dict, (
                    f"hook_pattern not found for {layer_key}. "
                    f"Available attn keys: "
                    f"{[k for k in cache.cache_dict if 'attn' in k and layer_key in k]}"
                )
                pattern = cache.cache_dict[pattern_key]
                assert pattern is not None, f"hook_pattern is None for {layer_key}"
                assert pattern.dim() == 4, (
                    f"Expected 4D (B,H,T,T) pattern for {layer_key}, got {pattern.shape}"
                )
                assert pattern.shape[1] == 4, (  # n_heads=4 from _make_tiny_raven
                    f"Expected 4 heads for {layer_key}, got {pattern.shape}"
                )
                # Every row must sum to 1 — valid probability distribution.
                row_sums = pattern.sum(dim=-1)
                assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
                    f"Attention rows don't sum to 1 for {layer_key}: "
                    f"max deviation = {(row_sums - 1).abs().max().item()}"
                )

    def test_attn_pattern_available_for_all_blocks(self, bridge_model, sample_input):
        """Attention patterns must be retrievable for prelude, core, and coda blocks.

        Raven uses RavenAttentionBridge with capture_sdpa_patterns(), which
        fires hook_pattern with the post-softmax attention patterns.  _get_attn_pattern
        reads hook_pattern directly so patterns are exact.
        """

        n_recurrences = 2
        with torch.no_grad():
            with torch.random.fork_rng():
                torch.manual_seed(42)
                _, cache = bridge_model.run_with_cache(
                    sample_input, num_steps=n_recurrences, accumulate=True
                )

        # prelude=1 block, core=2 blocks, coda=1 block (from _make_tiny_raven)
        for prefix, n_layers in [
            ("prelude_blocks", 1),
            ("core_blocks", 2),
            ("coda_blocks", 1),
        ]:
            for block_i in range(n_layers):
                layer_key = f"{prefix}.{block_i}"
                pattern = _get_attn_pattern(cache, layer_key)
                assert pattern is not None, (
                    f"_get_attn_pattern returned None for {layer_key}. "
                    f"Available attn keys: "
                    f"{[k for k in cache.cache_dict if 'attn' in k and layer_key in k]}"
                )
                assert pattern.dim() == 4, (
                    f"Expected 4D pattern (B,H,T,T) for {layer_key}, got {pattern.shape}"
                )
                assert pattern.shape[1] == 4, (  # n_heads=4 in _make_tiny_raven
                    f"Expected 4 heads for {layer_key}, got {pattern.shape}"
                )

    def test_attn_pattern_iter_available_for_core_blocks(
        self, bridge_model, sample_input
    ):
        """Per-iteration attention patterns must be available for recurrent core blocks."""

        n_recurrences = 3
        with torch.no_grad():
            with torch.random.fork_rng():
                torch.manual_seed(42)
                _, cache = bridge_model.run_with_cache(
                    sample_input, num_steps=n_recurrences, accumulate=True
                )

        for block_i in range(2):  # n_layers_in_recurrent_block=2
            for iter_i in range(n_recurrences):
                pattern = _get_attn_pattern_iter(
                    cache, f"core_blocks.{block_i}", iter_i
                )
                assert pattern is not None, (
                    f"_get_attn_pattern_iter returned None for core_blocks.{block_i} iter {iter_i}"
                )
                assert pattern.dim() == 4
