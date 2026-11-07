"""Integration test for Ouro (LoopLM) architecture adapter.

Tests that the TransformerBridge wrapping of an Ouro model produces identical outputs
to the raw HuggingFace model. Requires the Ouro model code to be available via
HuggingFace's trust_remote_code mechanism.
"""

import gc
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge

from looped_internals.capture.extraction import (
    _get_attn_pattern,
    _get_attn_pattern_iter,
)
from looped_internals.models._compat.ouro import (
    prepare_ouro_config,
    prepare_ouro_source,
)
from looped_internals.models.adapters.ouro import OuroArchitectureAdapter


def _make_tiny_ouro():
    """Create a tiny Ouro model for testing."""
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        prepare_ouro_source("ByteDance/Ouro-1.4B")
        config = AutoConfig.from_pretrained(
            "ByteDance/Ouro-1.4B", trust_remote_code=True
        )
    except OSError as exc:
        pytest.skip(f"Ouro config/source unavailable: {exc}")
    config.vocab_size = 256
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.max_position_embeddings = 64
    config.total_ut_steps = 2
    config.rms_norm_eps = 1e-6
    config.head_dim = 32

    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config._attn_implementation = "eager"
    config.pad_token_id = 0
    config.bos_token_id = 1
    config.eos_token_id = 2
    prepare_ouro_config(config)
    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, dtype=torch.float32
    )
    model.eval()
    return model


@pytest.mark.parametrize("model_name", ["ByteDance/Ouro-1.4B", "ByteDance/Ouro-2.6B"])
@pytest.mark.filterwarnings("error:`rope_config_validation`:FutureWarning")
def test_source_preparation_preserves_validation_and_documents_cache(
    model_name, capsys
):
    from transformers import AutoConfig, modeling_rope_utils, utils

    original_docstring = utils.auto_docstring
    original_validator = modeling_rope_utils.rope_config_validation
    try:
        prepare_ouro_source(model_name)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except OSError as exc:
        pytest.skip(f"Ouro config/source unavailable: {exc}")
    model_class = prepare_ouro_config(config)
    module = sys.modules[model_class.__module__]
    assert "[ERROR]" not in capsys.readouterr().out
    for cls in (module.OuroModel, model_class):
        assert "cache_position" in cls.forward.__doc__
        assert "Positions of the input tokens" in cls.forward.__doc__
    assert utils.auto_docstring is original_docstring
    assert modeling_rope_utils.rope_config_validation is original_validator
    assert config.rope_parameters["rope_theta"] == config.rope_theta
    with pytest.raises(KeyError, match="factor"):
        type(config)(rope_scaling={"rope_type": "linear"})


def test_failed_import_restores_docstring_decorator(monkeypatch):
    from transformers import utils

    from looped_internals.models._compat import ouro

    original = utils.auto_docstring

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate import failure")

    monkeypatch.setattr(ouro, "get_class_from_dynamic_module", fail)
    config = SimpleNamespace(
        model_type="ouro",
        name_or_path="test-ouro",
        auto_map={"AutoModelForCausalLM": "modeling_ouro.OuroForCausalLM"},
    )
    with pytest.raises(RuntimeError, match="deliberate import failure"):
        prepare_ouro_config(config)
    assert utils.auto_docstring is original


class TestMatchOuro:
    @pytest.fixture(scope="class")
    def hf_model(self):
        model = _make_tiny_ouro()
        yield model
        del model
        gc.collect()

    @pytest.fixture(scope="class")
    def bridge_model(self, hf_model):
        cfg = TransformerBridgeConfig(
            d_model=128,
            d_head=32,
            n_layers=2,
            n_ctx=64,
            n_heads=4,
            d_vocab=256,
            architecture="OuroForCausalLM",
        )
        adapter = OuroArchitectureAdapter(cfg)
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
        with torch.no_grad():
            hf_output = hf_model(sample_input)
            hf_logits = hf_output.logits

            bridge_logits = bridge_model(sample_input, return_type="logits")

        assert hf_logits.shape == bridge_logits.shape, (
            f"Shape mismatch: HF {hf_logits.shape} vs Bridge {bridge_logits.shape}"
        )
        assert torch.allclose(hf_logits, bridge_logits, atol=1e-4), (
            f"Max diff: {(hf_logits - bridge_logits).abs().max().item()}"
        )

    def test_cached_decode_matches_full_prefix(self, hf_model, sample_input):
        tokens = sample_input[:1, :4]
        with torch.no_grad():
            output = hf_model(tokens, use_cache=True)
            cache = output.past_key_values
            for token in (7, 11, 19):
                next_token = torch.tensor([[token]])
                tokens = torch.cat([tokens, next_token], dim=1)
                decoded = hf_model(next_token, past_key_values=cache, use_cache=True)
                full = hf_model(tokens, use_cache=False)
                torch.testing.assert_close(
                    decoded.logits[:, -1], full.logits[:, -1], atol=1e-5, rtol=1e-4
                )
                cache = decoded.past_key_values

    def test_generation_matches_without_cache(self, hf_model, sample_input):
        with torch.no_grad():
            options = dict(max_new_tokens=3, do_sample=False)
            cached = hf_model.generate(sample_input[:1, :4], use_cache=True, **options)
            full = hf_model.generate(sample_input[:1, :4], use_cache=False, **options)
        assert torch.equal(cached, full)

    def test_boot_saved_checkpoint(self, hf_model, tmp_path):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast

        from looped_internals.models import ModelSpec, load_model

        hf_model.save_pretrained(tmp_path)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                WordLevel({f"t{i}": i for i in range(256)}, unk_token="t3")
            ),
            pad_token="t0",
            bos_token="t1",
            eos_token="t2",
            unk_token="t3",
        )
        tokenizer.save_pretrained(tmp_path)
        bridge, structure = load_model(
            ModelSpec(
                str(tmp_path),
                layout="recurrent",
                recurrence_parameter="total_ut_steps",
                prepare_source=prepare_ouro_source,
                trust_remote_code=True,
            ),
            device="cpu",
            dtype=torch.float32,
            num_steps=2,
        )
        tokens = torch.tensor([[1, 4, 5, 6]])
        with torch.no_grad():
            expected = hf_model(tokens, use_cache=False).logits
            actual, cache = bridge.run_with_cache(tokens, accumulate=True)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        assert structure.groups[0].n_layers == 2
        assert cache.get_iterations("blocks.0.hook_out").shape[0] == 2

    def test_compatibility_does_not_patch_transformers_globals(self, hf_model):
        import sys

        from transformers import masking_utils, modeling_rope_utils

        module = sys.modules[type(hf_model).__module__]
        assert module.ROPE_INIT_FUNCTIONS is not modeling_rope_utils.ROPE_INIT_FUNCTIONS
        assert "default" not in modeling_rope_utils.ROPE_INIT_FUNCTIONS
        assert module.create_causal_mask is not masking_utils.create_causal_mask

    def test_recurrent_hook_access(self, bridge_model, sample_input):
        """Test that all blocks have iteration entries when accumulate=True."""
        n_layers = 2  # num_hidden_layers in _make_tiny_ouro
        total_ut_steps = 2  # total_ut_steps in _make_tiny_ouro

        with torch.no_grad():
            _, cache = bridge_model.run_with_cache(sample_input, accumulate=True)

        # Every block should have exactly total_ut_steps iteration entries
        hook_suffixes = ["hook_in", "hook_out", "attn.hook_out", "mlp.hook_out"]
        for block_idx in range(n_layers):
            for suffix in hook_suffixes:
                base = f"blocks.{block_idx}.{suffix}"
                for it in range(total_ut_steps):
                    key = f"{base}.iter_{it}"
                    assert key in cache, f"Missing iteration entry {key}"
                # No extra iterations beyond total_ut_steps
                assert f"{base}.iter_{total_ut_steps}" not in cache, (
                    f"Unexpected extra iteration for {base}"
                )
                # Plain key holds the final iteration's value
                assert base in cache, f"Missing plain key {base}"
                final_iter = cache[f"{base}.iter_{total_ut_steps - 1}"]
                assert torch.equal(cache[base], final_iter), (
                    f"Plain key {base} should match the last iteration"
                )

        # get_iterations should stack all iterations with correct shape
        test_hook = "blocks.0.attn.hook_out"
        stacked = cache.get_iterations(test_hook)
        assert stacked.shape[0] == total_ut_steps, (
            f"Expected {total_ut_steps} iterations, got {stacked.shape[0]}"
        )
        assert stacked.shape[1:] == cache[test_hook].shape, (
            "Stacked iteration shape should match single activation shape"
        )

    def test_component_access_embed(self, bridge_model):
        """Test that we can access the embedding component through the bridge."""
        embed = bridge_model.adapter.get_component(bridge_model.original_model, "embed")
        assert embed is not None
        assert hasattr(embed, "weight")

    def test_component_access_block_norms(self, bridge_model):
        """Test that we can access all 4 sandwich norms in a block."""
        for ln_name in ("ln1", "ln2", "ln3", "ln4"):
            component = bridge_model.adapter.get_component(
                bridge_model.original_model, f"blocks.0.{ln_name}"
            )
            assert component is not None, f"Could not access blocks.0.{ln_name}"

    def test_component_access_block_attn(self, bridge_model):
        """Test that we can access attention in a block."""
        attn = bridge_model.adapter.get_component(
            bridge_model.original_model, "blocks.0.attn"
        )
        assert attn is not None

    def test_component_access_block_mlp(self, bridge_model):
        """Test that we can access MLP in a block."""
        mlp = bridge_model.adapter.get_component(
            bridge_model.original_model, "blocks.0.mlp"
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
        assert adapter.translate_transformer_lens_path("embed") == "model.embed_tokens"
        assert adapter.translate_transformer_lens_path("ln_final") == "model.norm"
        assert adapter.translate_transformer_lens_path("unembed") == "lm_head"
        assert (
            adapter.translate_transformer_lens_path("blocks.0.ln1")
            == "model.layers.0.input_layernorm"
        )
        assert (
            adapter.translate_transformer_lens_path("blocks.0.ln2")
            == "model.layers.0.input_layernorm_2"
        )
        assert (
            adapter.translate_transformer_lens_path("blocks.0.ln3")
            == "model.layers.0.post_attention_layernorm"
        )
        assert (
            adapter.translate_transformer_lens_path("blocks.0.ln4")
            == "model.layers.0.post_attention_layernorm_2"
        )
        assert (
            adapter.translate_transformer_lens_path("blocks.0.attn")
            == "model.layers.0.self_attn"
        )
        assert (
            adapter.translate_transformer_lens_path("blocks.0.mlp")
            == "model.layers.0.mlp"
        )

    def test_attn_pattern_available_for_all_blocks(self, bridge_model, sample_input):
        """Attention patterns must be retrievable for all Ouro blocks.

        Ouro uses PositionEmbeddingsAttentionBridge which exposes hook_pattern
        directly, and also has q.hook_out / k.hook_out as fallbacks.
        """
        n_layers = 2  # num_hidden_layers in _make_tiny_ouro
        with torch.no_grad():
            _, cache = bridge_model.run_with_cache(sample_input, accumulate=True)

        for block_i in range(n_layers):
            layer_key = f"blocks.{block_i}"
            pattern = _get_attn_pattern(cache, layer_key)
            assert pattern is not None, (
                f"_get_attn_pattern returned None for {layer_key}. "
                f"Available attn keys: "
                f"{[k for k in cache.cache_dict if 'attn' in k and layer_key in k]}"
            )
            assert pattern.dim() == 4, (
                f"Expected 4D pattern (B,H,T,T) for {layer_key}, got {pattern.shape}"
            )

    def test_attn_pattern_iter_available(self, bridge_model, sample_input):
        """Per-iteration attention patterns must be available for Ouro (all blocks recurrent)."""
        total_ut_steps = 2  # total_ut_steps in _make_tiny_ouro
        with torch.no_grad():
            _, cache = bridge_model.run_with_cache(sample_input, accumulate=True)

        for block_i in range(2):
            for iter_i in range(total_ut_steps):
                pattern = _get_attn_pattern_iter(cache, f"blocks.{block_i}", iter_i)
                assert pattern is not None, (
                    f"_get_attn_pattern_iter returned None for blocks.{block_i} iter {iter_i}"
                )
                assert pattern.dim() == 4
