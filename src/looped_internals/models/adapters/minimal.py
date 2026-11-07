"""TransformerLens adapter for the minimal recurrent transformer."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)

from looped_internals.capture.attention import capture_sdpa_patterns


class MinimalAttentionBridge(GeneralizedComponent):
    """Attention bridge for MinimalRecurrentLM that exposes ``hook_pattern``.

    Mirrors :class:`~looped_internals.models.adapters.raven.RavenAttentionBridge`: fires
    ``hook_pattern`` with the post-softmax attention pattern when the
    hook has listeners (i.e. during ``run_with_cache``).  Plain forward passes
    use the fast ``F.scaled_dot_product_attention`` path unchanged.
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
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            args = (self.hook_in(args[0]), *args[1:])

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


def _make_block_submodules(cfg: Any, gated_mlp: bool = True) -> dict[str, Any]:
    """Create the submodule mapping for a MinimalRecurrentLM TransformerBlock."""
    if gated_mlp:
        mlp_bridge: Any = GatedMLPBridge(
            name="mlp",
            config=cfg,
            submodules={
                "gate": LinearBridge(name="gate_proj"),
                "in": LinearBridge(name="up_proj"),
                "out": LinearBridge(name="down_proj"),
            },
        )
    else:
        mlp_bridge = GeneralizedComponent(
            name="mlp",
            config=cfg,
            submodules={
                "in": LinearBridge(name="fc_in"),
                "out": LinearBridge(name="fc_out"),
            },
        )
    submodules: dict[str, Any] = {
        "ln1": RMSNormalizationBridge(name="norm_pre_attn", config=cfg),
        "ln2": RMSNormalizationBridge(name="norm_post_attn", config=cfg),
        "ln3": RMSNormalizationBridge(name="norm_pre_mlp", config=cfg),
        "ln4": RMSNormalizationBridge(name="norm_post_mlp", config=cfg),
        "attn": MinimalAttentionBridge(
            name="attn",
            config=cfg,
            submodules={
                "q": LinearBridge(name="q_proj"),
                "k": LinearBridge(name="k_proj"),
                "v": LinearBridge(name="v_proj"),
                "o": LinearBridge(name="o_proj"),
            },
        ),
        "mlp": mlp_bridge,
    }
    return submodules


class MinimalBlockBridge(BlockBridge):
    """Preserve tensor outputs when the recurrent loop calls ``block(hidden)``.

    Upstream BlockBridge treats this signature as a standalone component call
    and wraps the output in a tuple. Here it is also the model's forward path.
    """

    # MinimalAttentionBridge captures patterns without upstream's
    # pre-normalization attention-input or per-head Q/K/V input hooks.
    hook_aliases: ClassVar[dict[str, str]] = {
        name: target
        for name, target in BlockBridge.hook_aliases.items()
        if name not in {"hook_attn_in", "hook_q_input", "hook_k_input", "hook_v_input"}
    }

    @staticmethod
    def _is_standalone_hidden_state_call(args: tuple, kwargs: dict) -> bool:
        return False


class MinimalRecurrentAdapter(ArchitectureAdapter):
    """Architecture adapter for :class:`MinimalRecurrentLM`.

    Maps the minimal model's module structure to the TransformerLens bridge
    interface, providing hooks for activation caching and intervention.
    """

    def __init__(self, cfg: Any, gated_mlp: bool = True) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = gated_mlp
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        self.cfg.eps_attr = "eps"

        n_heads = getattr(cfg, "n_heads", 4)
        d_model = getattr(cfg, "d_model", 256)
        head_dim = d_model // n_heads
        n_ctx = getattr(cfg, "n_ctx", 2048)

        self.default_cfg: dict[str, Any] = {
            "d_model": d_model,
            "d_head": head_dim,
            "n_heads": n_heads,
            "n_key_value_heads": n_heads,
            "n_ctx": n_ctx,
        }

        self.component_mapping: dict[str, Any] = {
            "embed": EmbeddingBridge(name="embed_tokens"),
            "blocks": MinimalBlockBridge(
                name="transformer_layers",
                submodules=_make_block_submodules(self.cfg, gated_mlp=gated_mlp),
            ),
            "ln_final": RMSNormalizationBridge(name="cycle_norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
