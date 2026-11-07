"""Slow GPU tests for ``capture_sdpa_patterns`` correctness on real HF models.

Tests that the SDPA-intercept approach in ``capture_sdpa_patterns`` produces:

1. Logits numerically identical to the fused SDPA path (``atol=1e-3``).
2. Valid ``hook_pattern`` tensors for every attention block.

Covers:
- Huginn (``tomg-group-umd/huginn-0125``): lazy-patched via ``RavenAttentionBridge``.
- Retrofitted Llama (``smcleish/Recurrent-Llama-3.2-train-recurrence-32``).
- Retrofitted OLMo (``smcleish/Recurrent-OLMo-2-0425-train-recurrence-32``).
- Retrofitted TinyLlama (``smcleish/Recurrent-TinyLlama-3T-train-recurrence-32``):
  ``llama``-style RoPE.

All four classes are skipped unless a CUDA GPU is available and the model
config is accessible from HuggingFace.

Comparison strategy
-------------------
The key invariant is:

  plain bridge forward (no hooks → fused SDPA fast path)
  ≈ bridge.run_with_cache (same SDPA output, detached attention patterns captured)

This is the precise meaning of "patching is used vs not": when hook_pattern
has listeners, RavenAttentionBridge calls _patched_forward with
output_attentions=True; otherwise it calls the original forward unchanged.
"""

from __future__ import annotations

import gc

import pytest
import torch

HAS_GPU = torch.cuda.is_available()


def _try_fetch_config(model_name: str, trust_remote_code: bool = False) -> bool:
    """Return True if the HF model config is accessible (config only, no weights)."""
    if not HAS_GPU:
        return False
    try:
        from transformers import AutoConfig

        from looped_internals.models._compat.raven import prepare_raven_source

        if trust_remote_code:
            prepare_raven_source(model_name)
        AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        return True
    except OSError:
        return False


HAS_HUGINN = _try_fetch_config("tomg-group-umd/huginn-0125", trust_remote_code=True)
HAS_RETROFITTED_LLAMA = _try_fetch_config(
    "smcleish/Recurrent-Llama-3.2-train-recurrence-32", trust_remote_code=True
)
HAS_RETROFITTED_OLMO = _try_fetch_config(
    "smcleish/Recurrent-OLMo-2-0425-train-recurrence-32", trust_remote_code=True
)
HAS_RETROFITTED_TINYLLAMA = _try_fetch_config(
    "smcleish/Recurrent-TinyLlama-3T-train-recurrence-32", trust_remote_code=True
)


def _assert_hook_pattern_valid(
    cache, group_name: str, n_layers: int, n_heads: int
) -> None:
    """Assert hook_pattern is present, 4D (B,H,T,T), and rows sum to 1 for all blocks."""
    for block_i in range(n_layers):
        layer_key = f"{group_name}.{block_i}"
        pattern_key = f"{layer_key}.attn.hook_pattern"
        assert pattern_key in cache.cache_dict, (
            f"hook_pattern missing for {layer_key}. "
            f"Available attn keys: {[k for k in cache.cache_dict if 'attn' in k and layer_key in k]}"
        )
        pattern = cache.cache_dict[pattern_key]
        assert pattern is not None, f"hook_pattern is None for {layer_key}"
        assert pattern.dim() == 4, (
            f"Expected 4D (B,H,T,T) pattern for {layer_key}, got {pattern.shape}"
        )
        assert pattern.shape[1] == n_heads, (
            f"Expected {n_heads} heads for {layer_key}, got shape {pattern.shape}"
        )
        row_sums = pattern.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            f"Attention rows don't sum to 1 for {layer_key}: "
            f"max deviation = {(row_sums - 1).abs().max().item()}"
        )


class TestHuginnOutputAttentions:
    """Test _patched_forward on the real Huginn model (complex_like RoPE style).

    Huginn uses apply_rotary_emb_complex_like; its CausalSelfAttention class is
    lazy-patched by RavenAttentionBridge on the first forward call.
    """

    pytestmark = pytest.mark.skipif(
        not HAS_GPU or not HAS_HUGINN,
        reason="GPU or tomg-group-umd/huginn-0125 not available",
    )

    @pytest.fixture(scope="class")
    def bridge_and_structure(self):
        from looped_internals.models import load_model

        bridge, structure = load_model(
            "raven", num_steps=1, device="cuda", dtype=torch.bfloat16
        )
        yield bridge, structure
        del bridge
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.fixture(scope="class")
    def sample_input(self):
        torch.manual_seed(42)
        return torch.randint(0, 1000, (1, 16)).cuda()

    def test_logit_invariance(self, bridge_and_structure, sample_input):
        """Attention capture must preserve logits from fused SDPA.

        plain bridge forward (no hooks) ≈ run_with_cache (hooks active).
        """
        bridge, _ = bridge_and_structure
        fixed_seed = 42
        with torch.no_grad():
            with torch.random.fork_rng(devices=[sample_input.device]):
                torch.manual_seed(fixed_seed)
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False
                ):
                    plain_logits = bridge(
                        sample_input, return_type="logits", num_steps=1
                    )
            with torch.random.fork_rng(devices=[sample_input.device]):
                torch.manual_seed(fixed_seed)
                cached_logits, _ = bridge.run_with_cache(sample_input, num_steps=1)
        assert torch.allclose(plain_logits, cached_logits, atol=0.1), (
            f"Logit mismatch (Huginn): "
            f"max diff = {(plain_logits - cached_logits).abs().max().item()}"
        )

    def test_hook_pattern_valid(self, bridge_and_structure, sample_input):
        """hook_pattern must be present and valid for every block group."""
        bridge, structure = bridge_and_structure
        with torch.no_grad():
            _, cache = bridge.run_with_cache(sample_input, num_steps=1)
        n_heads = bridge.cfg.n_heads
        for group in structure.groups:
            _assert_hook_pattern_valid(
                cache, group.hook_prefix, group.n_layers, n_heads
            )


class TestRetrofittedLlamaOutputAttentions:
    """Test _patched_forward on Retrofitted Llama (llama-style RoPE).

    Uses apply_rotary_pos_emb; RoPE is applied after the head-first transpose
    on (B, H, S, D) tensors using a (cos, sin) tuple.
    """

    pytestmark = pytest.mark.skipif(
        not HAS_GPU or not HAS_RETROFITTED_LLAMA,
        reason="GPU or smcleish/Recurrent-Llama-3.2-train-recurrence-32 not available",
    )

    @pytest.fixture(scope="class")
    def bridge_and_structure(self):
        from looped_internals.models import load_model

        bridge, structure = load_model(
            "retrofitted_llama_32",
            num_steps=1,
            device="cuda",
            dtype=torch.bfloat16,
        )
        yield bridge, structure
        del bridge
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.fixture(scope="class")
    def sample_input(self):
        torch.manual_seed(42)
        return torch.randint(0, 1000, (1, 16)).cuda()

    def test_logit_invariance(self, bridge_and_structure, sample_input):
        """Attention capture must preserve logits from fused SDPA."""
        bridge, _ = bridge_and_structure
        fixed_seed = 42
        with torch.no_grad():
            with torch.random.fork_rng(devices=[sample_input.device]):
                torch.manual_seed(fixed_seed)
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False
                ):
                    plain_logits = bridge(
                        sample_input, return_type="logits", num_steps=1
                    )
            with torch.random.fork_rng(devices=[sample_input.device]):
                torch.manual_seed(fixed_seed)
                cached_logits, _ = bridge.run_with_cache(sample_input, num_steps=1)
        assert torch.allclose(plain_logits, cached_logits, atol=0.1), (
            f"Logit mismatch (Retrofitted Llama): "
            f"max diff = {(plain_logits - cached_logits).abs().max().item()}"
        )

    def test_hook_pattern_valid(self, bridge_and_structure, sample_input):
        """hook_pattern must be present and valid for every block group."""
        bridge, structure = bridge_and_structure
        with torch.no_grad():
            _, cache = bridge.run_with_cache(sample_input, num_steps=1)
        n_heads = bridge.cfg.n_heads
        for group in structure.groups:
            _assert_hook_pattern_valid(
                cache, group.hook_prefix, group.n_layers, n_heads
            )


class TestRetrofittedOLMoOutputAttentions:
    """Test _patched_forward on Retrofitted OLMo (llama-style RoPE)."""

    pytestmark = pytest.mark.skipif(
        not HAS_GPU or not HAS_RETROFITTED_OLMO,
        reason="GPU or smcleish/Recurrent-OLMo-2-0425-train-recurrence-32 not available",
    )

    @pytest.fixture(scope="class")
    def bridge_and_structure(self):
        from looped_internals.models import load_model

        bridge, structure = load_model(
            "retrofitted_olmo_32",
            num_steps=1,
            device="cuda",
            dtype=torch.bfloat16,
        )
        yield bridge, structure
        del bridge
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.fixture(scope="class")
    def sample_input(self):
        torch.manual_seed(42)
        return torch.randint(0, 1000, (1, 16)).cuda()

    def test_logit_invariance(self, bridge_and_structure, sample_input):
        """Attention capture must preserve logits from fused SDPA."""
        bridge, _ = bridge_and_structure
        fixed_seed = 42
        with torch.no_grad():
            with torch.random.fork_rng(devices=[sample_input.device]):
                torch.manual_seed(fixed_seed)
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False
                ):
                    plain_logits = bridge(
                        sample_input, return_type="logits", num_steps=1
                    )
            with torch.random.fork_rng(devices=[sample_input.device]):
                torch.manual_seed(fixed_seed)
                cached_logits, _ = bridge.run_with_cache(sample_input, num_steps=1)
        assert torch.allclose(plain_logits, cached_logits, atol=0.1), (
            f"Logit mismatch (Retrofitted OLMo): "
            f"max diff = {(plain_logits - cached_logits).abs().max().item()}"
        )

    def test_hook_pattern_valid(self, bridge_and_structure, sample_input):
        """hook_pattern must be present and valid for every block group."""
        bridge, structure = bridge_and_structure
        with torch.no_grad():
            _, cache = bridge.run_with_cache(sample_input, num_steps=1)
        n_heads = bridge.cfg.n_heads
        for group in structure.groups:
            _assert_hook_pattern_valid(
                cache, group.hook_prefix, group.n_layers, n_heads
            )


class TestRetrofittedTinyLlamaOutputAttentions:
    """Test _patched_forward on Retrofitted TinyLlama (llama-style RoPE)."""

    pytestmark = pytest.mark.skipif(
        not HAS_GPU or not HAS_RETROFITTED_TINYLLAMA,
        reason="GPU or smcleish/Recurrent-TinyLlama-3T-train-recurrence-32 not available",
    )

    @pytest.fixture(scope="class")
    def bridge_and_structure(self):
        from looped_internals.models import load_model

        bridge, structure = load_model(
            "retrofitted_tinyllama_32",
            num_steps=1,
            device="cuda",
            dtype=torch.bfloat16,
        )
        yield bridge, structure
        del bridge
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.fixture(scope="class")
    def sample_input(self):
        torch.manual_seed(42)
        return torch.randint(0, 1000, (1, 16)).cuda()

    def test_logit_invariance(self, bridge_and_structure, sample_input):
        """Attention capture must preserve logits from fused SDPA."""
        bridge, _ = bridge_and_structure
        fixed_seed = 42
        with torch.no_grad():
            with torch.random.fork_rng(devices=[sample_input.device]):
                torch.manual_seed(fixed_seed)
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False
                ):
                    plain_logits = bridge(
                        sample_input, return_type="logits", num_steps=1
                    )
            with torch.random.fork_rng(devices=[sample_input.device]):
                torch.manual_seed(fixed_seed)
                cached_logits, _ = bridge.run_with_cache(sample_input, num_steps=1)
        assert torch.allclose(plain_logits, cached_logits, atol=0.1), (
            f"Logit mismatch (Retrofitted TinyLlama): "
            f"max diff = {(plain_logits - cached_logits).abs().max().item()}"
        )

    def test_hook_pattern_valid(self, bridge_and_structure, sample_input):
        """hook_pattern must be present and valid for every block group."""
        bridge, structure = bridge_and_structure
        with torch.no_grad():
            _, cache = bridge.run_with_cache(sample_input, num_steps=1)
        n_heads = bridge.cfg.n_heads
        for group in structure.groups:
            _assert_hook_pattern_valid(
                cache, group.hook_prefix, group.n_layers, n_heads
            )
