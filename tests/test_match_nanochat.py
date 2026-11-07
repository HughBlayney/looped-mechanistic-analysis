"""Tests for the nanochat TrellisGPT2 bridge."""

from __future__ import annotations

import gc
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

NANOCHAT_REPO = Path("/slurm-storage/hugbla/repos/external/looped")


def _try_import_nanochat():
    if not NANOCHAT_REPO.exists():
        return None, None
    repo = str(NANOCHAT_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from nanochat.gpt import GPT, GPTConfig  # type: ignore[unresolved-import]
    except Exception:
        return None, None
    return GPT, GPTConfig


GPT, GPTConfig = _try_import_nanochat()
HAS_NANOCHAT = GPT is not None and GPTConfig is not None

pytestmark = pytest.mark.skipif(
    not HAS_NANOCHAT, reason="external nanochat repo required"
)

from looped_internals.capture import compute_model_run_data, metrics_names_filter
from looped_internals.models.nanochat import (
    NanochatTokenizerAdapter,
    make_nanochat_bridge,
)


def _make_tiny_nanochat(recurrent: bool = True):
    assert GPT is not None and GPTConfig is not None
    config = GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        n_prelude=1 if recurrent else 4,
        n_recur_block=2 if recurrent else 0,
        n_coda=1 if recurrent else 0,
        train_recur_mean=4.0,
        train_recur_max=4,
        inject_mode="concat_linear",
    )
    model = GPT(config)
    model.init_weights()
    model.eval()
    return model


class TestMatchNanochat:
    @pytest.fixture(scope="class")
    def sample_input(self):
        torch.manual_seed(42)
        return torch.randint(0, 256, (2, 16))

    def test_full_model_output_match_feedforward(self, sample_input):
        hf_model = _make_tiny_nanochat(recurrent=False)
        bridge, _structure = make_nanochat_bridge(deepcopy(hf_model), tokenizer=None)
        with torch.no_grad():
            torch.manual_seed(42)
            raw_logits, _ = hf_model(sample_input)
            torch.manual_seed(42)
            bridge_logits = bridge(sample_input, return_type="logits")

        assert raw_logits.shape == bridge_logits.shape
        assert torch.allclose(raw_logits, bridge_logits, atol=1e-4)
        del hf_model, bridge
        gc.collect()

    def test_full_model_output_match_recurrent(self, sample_input):
        hf_model = _make_tiny_nanochat(recurrent=True)
        bridge, _structure = make_nanochat_bridge(deepcopy(hf_model), tokenizer=None)
        with torch.no_grad():
            torch.manual_seed(42)
            raw_logits, _ = hf_model(sample_input, num_recur=4)
            torch.manual_seed(42)
            bridge_logits = bridge(sample_input, return_type="logits", num_recur=4)

        assert raw_logits.shape == bridge_logits.shape
        assert torch.allclose(raw_logits, bridge_logits, atol=1e-4)
        del hf_model, bridge
        gc.collect()

    def test_recurrent_hook_access(self, sample_input):
        model = _make_tiny_nanochat(recurrent=True)
        bridge, _structure = make_nanochat_bridge(model, tokenizer=None)

        with torch.no_grad():
            torch.manual_seed(42)
            _, cache = bridge.run_with_cache(
                sample_input,
                num_recur=4,
                accumulate=True,
                names_filter=metrics_names_filter,
            )

        assert "prelude_blocks.0.hook_out" in cache
        assert "coda_blocks.0.hook_out" in cache
        assert "prelude_blocks.0.hook_out.iter_1" not in cache
        assert "coda_blocks.0.hook_out.iter_1" not in cache

        for block_idx in range(2):
            for suffix in ["hook_out", "attn.hook_pattern"]:
                base = f"core_blocks.{block_idx}.{suffix}"
                for it in range(4):
                    assert f"{base}.iter_{it}" in cache
                assert f"{base}.iter_4" not in cache

        stacked = getattr(cache, "get_iterations")("core_blocks.0.hook_out")
        assert stacked.shape[0] == 4

    def test_compute_model_run_data_has_attention_metrics(self, sample_input):
        model = _make_tiny_nanochat(recurrent=True)
        bridge, structure = make_nanochat_bridge(model, tokenizer=None)

        with torch.no_grad():
            torch.manual_seed(42)
            _, cache = bridge.run_with_cache(
                sample_input[:1],
                num_recur=4,
                accumulate=True,
                names_filter=metrics_names_filter,
            )

        data = compute_model_run_data(
            cache=cache,
            input_ids=sample_input[:1],
            structure=structure,
            compute_pairwise=False,
            compute_differences=False,
        )
        assert len(data) == 1
        assert data[0].per_virtual_layer_metrics
        assert all(
            pbm.sink_scores is not None for pbm in data[0].per_virtual_layer_metrics
        )
        assert all(
            pbm.mixing_scores is not None for pbm in data[0].per_virtual_layer_metrics
        )

    def test_tokenizer_adapter_add_special_tokens(self):
        class TinyNanochatTokenizer:
            def get_bos_token_id(self):
                return 7

            def encode(self, text, prepend=None):
                ids = [ord(c) for c in text]
                if prepend is not None:
                    return [prepend, *ids]
                return ids

            def decode(self, ids):
                return "".join(chr(i) for i in ids if i != 7)

        tokenizer = NanochatTokenizerAdapter(TinyNanochatTokenizer())

        assert tokenizer.bos_token_id == 7
        assert tokenizer.pad_token_id == 0
        assert tokenizer.encode("ab", add_special_tokens=True) == [7, 97, 98]
        assert tokenizer.encode("ab", add_special_tokens=False) == [97, 98]
        assert tokenizer.decode([7, 97, 98]) == "ab"

    def test_ln_final_matches_rms_norm(self):
        model = _make_tiny_nanochat(recurrent=True)
        bridge, _structure = make_nanochat_bridge(model, tokenizer=None)
        hidden_states = torch.randn(2, 5, model.config.n_embd)

        with torch.no_grad():
            actual = bridge.ln_final(hidden_states)
            expected = torch.nn.functional.rms_norm(
                hidden_states, (model.config.n_embd,)
            )

        assert torch.allclose(actual, expected, atol=1e-6)
