"""Raven (Huginn) architecture adapter.

Supports the recurrent transformer architecture from recurrent-pretraining.
This adapter also covers retrofitted recurrence models (e.g.
smcleish/Recurrent-Llama-3.2-train-recurrence-32) which use the same
RavenForCausalLM architecture class and identical module structure.

The Raven model has a 3-stage structure:
  - Prelude: non-recurrent initial processing layers
  - Core block: recurrent layers (run once in single-pass mode)
  - Coda: final refinement layers

Each stage uses SandwichBlocks with RMSNorms, fused QKV attention (with GQA
support), and fused gate-up MLP projections.
"""

from typing import Any, ClassVar

import torch
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    JointGateUpMLPBridge,
    LinearBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)

from looped_internals.capture.attention import capture_sdpa_patterns


class RavenAttentionBridge(GeneralizedComponent):
    """Attention bridge for Raven that exposes post-RoPE attention patterns.

    Extends :class:`GeneralizedComponent` to fire ``hook_pattern`` with the
    post-softmax attention pattern captured via
    :func:`~looped_internals.capture.attention.capture_sdpa_patterns`.

    ``hook_pattern`` is only activated when the hook has listeners (i.e. during
    ``run_with_cache``).  Plain forward passes use the original fused
    ``F.scaled_dot_product_attention`` kernel unchanged.
    """

    def __init__(self, name: Any, config: Any, submodules=None):
        super().__init__(name, config=config, submodules=submodules or {})
        self.hook_pattern = HookPoint()

    def forward(self, *args, **kwargs):
        original_component = self._modules.get("_original_component", None)
        if original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        # hook_in on the first positional argument (normed hidden state from SandwichBlock).
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            args = (self.hook_in(args[0]), *args[1:])

        # Only intercept SDPA when hook_pattern has listeners; plain forward
        # passes use the original fused kernel unchanged.
        want_pattern = bool(getattr(self.hook_pattern, "fwd_hooks", None))

        if want_pattern:
            with capture_sdpa_patterns() as captured:
                output = original_component(*args, **kwargs)
            if (
                captured
                and isinstance(captured[0], torch.Tensor)
                and captured[0].dim() == 4
            ):
                self.hook_pattern(captured[0])
        else:
            output = original_component(*args, **kwargs)

        return self.hook_out(output)


class RavenBlockBridge(BlockBridge):
    """Expose only block aliases supported by Raven's native attention wrapper.

    RavenAttentionBridge captures patterns but does not implement upstream's
    pre-normalization attention-input or per-head Q/K/V input hooks.
    """

    hook_aliases: ClassVar[dict[str, str]] = {
        name: target
        for name, target in BlockBridge.hook_aliases.items()
        if name not in {"hook_attn_in", "hook_q_input", "hook_k_input", "hook_v_input"}
    }


def _split_gate_up(original_mlp):
    """Split Raven's fused gate-up MLP projection into separate gate and up linears.

    Raven's mlp.fc weight has shape [intermediate_size * 2, d_model].
    The first half is the gate projection, the second half is the up projection.
    """
    fc = original_mlp.fc
    weight = fc.weight.data  # shape: [intermediate_size * 2, d_model]
    half = weight.shape[0] // 2
    d_model = weight.shape[1]

    gate_weight = weight[:half, :]
    up_weight = weight[half:, :]

    has_bias = hasattr(fc, "bias") and fc.bias is not None

    gate_linear = torch.nn.Linear(d_model, half, bias=has_bias)
    gate_linear.weight = torch.nn.Parameter(gate_weight)
    up_linear = torch.nn.Linear(d_model, half, bias=has_bias)
    up_linear.weight = torch.nn.Parameter(up_weight)

    if has_bias:
        bias = fc.bias.data
        gate_linear.bias = torch.nn.Parameter(bias[:half])
        up_linear.bias = torch.nn.Parameter(bias[half:])

    return gate_linear, up_linear


def _make_sandwich_block_submodules(cfg: Any, n_norms: int = 4) -> dict:
    """Create the submodule mapping for a Raven SandwichBlock.

    Uses RavenAttentionBridge for attention, which delegates to the patched
    CausalSelfAttention forward pass and fires hook_pattern with the post-RoPE
    attention pattern during run_with_cache.

    Args:
        cfg: TransformerBridgeConfig.
        n_norms: Number of RMSNorms per block.  The original Raven (huginn-0125)
            uses 4 (sandwich normalization: pre+post for both attn and MLP).
            Retrofitted models (smcleish/Recurrent-*) use 2 (standard pre-norm
            only: norm_1 before attn, norm_2 before MLP — no norm_3/norm_4).
    """
    submodules: dict = {
        "ln1": RMSNormalizationBridge(name="norm_1", config=cfg),
        "ln2": RMSNormalizationBridge(name="norm_2", config=cfg),
    }
    if n_norms >= 4:
        submodules["ln3"] = RMSNormalizationBridge(name="norm_3", config=cfg)
        submodules["ln4"] = RMSNormalizationBridge(name="norm_4", config=cfg)
    submodules["attn"] = RavenAttentionBridge(
        name="attn",
        config=cfg,
        submodules={
            "qkv": LinearBridge(name="Wqkv"),
            "o": LinearBridge(name="proj"),
        },
    )
    submodules["mlp"] = JointGateUpMLPBridge(
        name="mlp",
        config=cfg,
        split_gate_up_matrix=_split_gate_up,
        submodules={
            "out": LinearBridge(name="proj"),
        },
    )
    return submodules


class RavenArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Raven (Huginn) recurrent transformer models.

    Raven uses a 3-stage sandwich architecture:
      - Prelude blocks: initial non-recurrent processing
      - Core blocks: recurrent layers (single pass in this adapter)
      - Coda blocks: final refinement layers

    Each SandwichBlock has 4 RMSNorms, fused QKV attention, and fused gate-up MLP.

    Optional Parameters (not present in state_dict):
    ------------------------------------------------
    Raven does NOT have biases on attention/MLP projections (except optional qk_bias):
    - blocks.*.attn.b_Q/b_K/b_V/b_O - No bias
    - blocks.*.mlp.b_in/b_gate/b_out - No bias
    - All norms are RMSNorm (no bias)
    """

    def __init__(self, cfg: Any, n_norms: int = 4) -> None:
        """Initialize the Raven architecture adapter.

        Args:
            cfg: TransformerBridgeConfig.
            n_norms: Number of RMSNorms per SandwichBlock.  Use ``4`` for the
                original Raven/Huginn model (sandwich normalization) and ``2``
                for retrofitted models (smcleish/Recurrent-*) which use
                standard pre-norm only (norm_1, norm_2 — no norm_3/norm_4).
                When using ``boot_transformers``, leave at the default — the
                correct value is auto-detected in :meth:`prepare_model`.
        """
        super().__init__(cfg)
        self._n_norms = n_norms

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.act_fn = "silu"
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        self.cfg.eps_attr = "eps"

        n_heads = getattr(cfg, "n_heads", getattr(cfg, "num_attention_heads", 8))
        n_kv_heads = (
            getattr(cfg, "n_key_value_heads", None)
            or getattr(cfg, "num_key_value_heads", None)
            or n_heads
        )
        d_model = getattr(
            cfg, "d_model", getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 512))
        )
        head_dim = d_model // n_heads
        n_ctx = getattr(cfg, "n_ctx", getattr(cfg, "block_size", 2048))

        self.default_cfg: dict[str, Any] = {
            "d_model": d_model,
            "d_head": head_dim,
            "n_heads": n_heads,
            "n_key_value_heads": n_kv_heads,
            "n_ctx": n_ctx,
        }

        block_subs = _make_sandwich_block_submodules(self.cfg, n_norms=n_norms)
        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "prelude_blocks": RavenBlockBridge(
                name="transformer.prelude",
                submodules=block_subs,
            ),
            "state_adapter": LinearBridge(name="transformer.adapter"),
            "core_blocks": RavenBlockBridge(
                name="transformer.core_block",
                submodules=block_subs,
            ),
            "coda_blocks": RavenBlockBridge(
                name="transformer.coda",
                submodules=block_subs,
            ),
            "ln_final": RMSNormalizationBridge(
                name="transformer.ln_f", config=self.cfg
            ),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def prepare_model(self, hf_model: Any) -> None:
        """Auto-detect ``n_norms`` from the loaded model and rebuild component mapping if needed.

        Called by ``boot_transformers`` after the HuggingFace model is loaded
        but before the bridge is constructed.  Retrofitted models
        (smcleish/Recurrent-*) only have ``norm_1``/``norm_2`` per block, while
        the original Huginn model has four sandwich norms.  This check allows
        ``boot_transformers`` to work for both variants without requiring the
        caller to pass ``n_norms`` explicitly.
        """
        try:
            first_block = hf_model.transformer.core_block[0]
            detected = 4 if hasattr(first_block, "norm_3") else 2
        except (AttributeError, IndexError):
            return
        if detected == self._n_norms:
            return
        if self.component_mapping is None:
            return
        self._n_norms = detected
        block_subs = _make_sandwich_block_submodules(self.cfg, n_norms=detected)
        self.component_mapping["prelude_blocks"] = RavenBlockBridge(
            name="transformer.prelude", submodules=block_subs
        )
        self.component_mapping["core_blocks"] = RavenBlockBridge(
            name="transformer.core_block", submodules=block_subs
        )
        self.component_mapping["coda_blocks"] = RavenBlockBridge(
            name="transformer.coda", submodules=block_subs
        )
