"""TransformerLens adapter for Parcae looped transformers.

Parcae (Prairie et al., 2026) has the same three-stage layout as Raven:
prelude blocks, a recurrent core, and coda blocks. Each block is a standard
pre-norm block with separate ``c_q``/``c_k``/``c_v``/``c_proj`` projections,
QK-norm after RoPE, and a non-gated ``fc``/``proj`` MLP. Between the prelude
and the core, a diagonal state-space injection mixes the input embedding into
the recurrent state; a linear projection ``C`` maps the final state into the
coda. Odd-numbered layers also add gated value embeddings to V.

Attention runs through ``F.scaled_dot_product_attention`` when the loader
disables Parcae's FlashAttention 3 path, so the SDPA-capturing bridges written
for Raven apply unchanged and expose ``attn.hook_pattern``. Block hooks follow
the ``prelude_blocks``/``core_blocks``/``coda_blocks`` naming shared by the
Raven and nanochat adapters, so every experiment reads Parcae like Raven.
"""

from __future__ import annotations

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)

from looped_internals.models.adapters.raven import (
    RavenAttentionBridge,
    RavenBlockBridge,
)


def _make_block_submodules(cfg: Any) -> dict[str, Any]:
    """Map one ``TransformerPreNormBlock`` onto TransformerLens block hooks."""
    return {
        "ln1": RMSNormalizationBridge(name="norm_1", config=cfg),
        "ln2": RMSNormalizationBridge(name="norm_2", config=cfg),
        "attn": RavenAttentionBridge(
            name="attn",
            config=cfg,
            submodules={
                "q": LinearBridge(name="c_q"),
                "k": LinearBridge(name="c_k"),
                "v": LinearBridge(name="c_v"),
                "o": LinearBridge(name="c_proj"),
            },
        ),
        "mlp": MLPBridge(
            name="mlp",
            config=cfg,
            submodules={
                "in": LinearBridge(name="fc"),
                "out": LinearBridge(name="proj"),
            },
        ),
    }


class ParcaeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for ``ParcaeLogitsWrapper``-wrapped Parcae models.

    Component names are relative to the wrapper, whose ``parcae`` attribute is
    the original ``parcae_lm`` model. Parcae has no biases anywhere and every
    norm is an RMSNorm. The core may use a different width from the prelude;
    the released checkpoints do not, and ``d_model`` describes the prelude.
    """

    def __init__(self, cfg: Any, *, prelude_norm: bool = True) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = False
        self.cfg.act_fn = "relu"
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        self.cfg.eps_attr = "eps"

        self.default_cfg: dict[str, Any] = {
            "d_model": cfg.d_model,
            "d_head": cfg.d_head,
            "n_heads": cfg.n_heads,
            "n_key_value_heads": cfg.n_key_value_heads,
            "n_ctx": cfg.n_ctx,
        }

        block_submodules = _make_block_submodules(self.cfg)
        self.component_mapping: dict[str, Any] = {
            "embed": EmbeddingBridge(name="parcae.transformer.wte"),
            "prelude_blocks": RavenBlockBridge(
                name="parcae.transformer.prelude", submodules=block_submodules
            ),
            # Diagonal injection of the (normalized) prelude output into the
            # recurrent state, applied at the start of every recurrence.
            "state_adapter": GeneralizedComponent(
                name="parcae.transformer.adapter", config=self.cfg
            ),
            "core_blocks": RavenBlockBridge(
                name="parcae.transformer.core_block", submodules=block_submodules
            ),
            "core_projection": LinearBridge(name="parcae.transformer.C"),
            "coda_blocks": RavenBlockBridge(
                name="parcae.transformer.coda", submodules=block_submodules
            ),
            "ln_final": RMSNormalizationBridge(
                name="parcae.transformer.ln_f", config=self.cfg
            ),
            "unembed": UnembeddingBridge(name="parcae.lm_head", config=self.cfg),
        }
        if prelude_norm:
            self.component_mapping["ln_prelude"] = RMSNormalizationBridge(
                name="parcae.transformer.ln_prelude", config=self.cfg
            )
