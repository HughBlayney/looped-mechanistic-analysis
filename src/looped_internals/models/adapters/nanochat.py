"""TransformerLens bridges for Nanochat modules and functional normalization."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch
import torch.nn as nn
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    LinearBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)

from looped_internals.capture.attention import capture_sdpa_patterns


class NanochatFinalRMSNorm(nn.Module):
    """Parameter-free final RMSNorm used by nanochat before the LM head."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.normalized_shape = (d_model,)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(hidden_states, self.normalized_shape)


class NanochatLogitsWrapper(nn.Module):
    """Wrap nanochat GPT so bridge forwards return logits only.

    nanochat ``GPT.forward`` returns ``(logits, recurrent_state)`` for inference,
    while the rest of this repo expects a logits tensor or an object with a
    ``.logits`` attribute. The wrapper keeps the original model available under
    ``.nanochat`` for component mapping and normalizes the public forward result.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.nanochat = model
        self.final_norm = NanochatFinalRMSNorm(cast(Any, model).config.n_embd)

    @property
    def config(self):
        return self.nanochat.config

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        kwargs.pop("return_type", None)
        # Attention patterns are exposed by NanochatAttentionBridge's hooks;
        # Nanochat itself has no Hugging Face output_attentions argument.
        kwargs.pop("output_attentions", None)
        if input_ids.device.type == "cuda":
            # nanochat checkpoints can mix bf16 embeddings/rotary buffers with
            # fp32 linear weights. The original plotting notebook used CUDA
            # bf16 autocast, which lets these matmuls run without dtype errors.
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.nanochat(input_ids, **kwargs)
        else:
            output = self.nanochat(input_ids, **kwargs)
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "logits"):
            return output.logits
        return output


class NanochatAttentionBridge(GeneralizedComponent):
    """Attention bridge that exposes nanochat post-softmax attention patterns."""

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


class NanochatBlockBridge(GeneralizedComponent):
    """Wrap list elements while leaving Nanochat's functional norms in place."""

    is_list_item = True
    hook_out_is_single_residual_stream = True
    hook_aliases: ClassVar[dict[str, str]] = {
        "hook_resid_pre": "hook_in",
        "hook_resid_post": "hook_out",
        "hook_attn_out": "attn.hook_out",
        "hook_mlp_out": "mlp.hook_out",
    }


def _make_block_submodules(cfg: Any) -> dict[str, Any]:
    return {
        "attn": NanochatAttentionBridge(
            name="attn",
            config=cfg,
            submodules={
                "q": LinearBridge(name="c_q"),
                "k": LinearBridge(name="c_k"),
                "v": LinearBridge(name="c_v"),
                "o": LinearBridge(name="c_proj"),
            },
        ),
        "mlp": GeneralizedComponent(
            name="mlp",
            config=cfg,
            submodules={
                "in": LinearBridge(name="c_fc"),
                "out": LinearBridge(name="c_proj"),
            },
        ),
    }


class NanochatArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for nanochat's TrellisGPT2-style recurrent GPT."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        self.cfg.eps_attr = "eps"

        n_heads = getattr(cfg, "n_heads", getattr(cfg, "n_head", 6))
        n_kv_heads = (
            getattr(cfg, "n_key_value_heads", None)
            or getattr(cfg, "n_kv_head", None)
            or n_heads
        )
        d_model = getattr(cfg, "d_model", getattr(cfg, "n_embd", 768))
        d_head = getattr(cfg, "d_head", d_model // n_heads)
        n_ctx = getattr(cfg, "n_ctx", getattr(cfg, "sequence_len", 1024))

        self.default_cfg: dict[str, Any] = {
            "d_model": d_model,
            "d_head": d_head,
            "n_heads": n_heads,
            "n_key_value_heads": n_kv_heads,
            "n_ctx": n_ctx,
        }

        block_submodules = _make_block_submodules(self.cfg)
        self.component_mapping = {
            "embed": EmbeddingBridge(name="nanochat.transformer.wte"),
            # Nanochat normalizes functionally inside each block. A generic
            # wrapper preserves that forward and exposes hook_in/hook_out
            # without BlockBridge's assumptions about ln1/ln2 modules.
            "prelude_blocks": NanochatBlockBridge(
                name="nanochat.transformer.prelude",
                submodules=block_submodules,
            ),
            "core_blocks": NanochatBlockBridge(
                name="nanochat.transformer.recur",
                submodules=block_submodules,
            ),
            "coda_blocks": NanochatBlockBridge(
                name="nanochat.transformer.coda",
                submodules=block_submodules,
            ),
            "ln_final": GeneralizedComponent(name="final_norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="nanochat.lm_head", config=self.cfg),
        }
