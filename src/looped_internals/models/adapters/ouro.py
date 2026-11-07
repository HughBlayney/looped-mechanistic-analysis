"""Ouro (LoopLM) architecture adapter.

Supports the Ouro looped language model from ByteDance.

The Ouro model loops ALL layers for a configurable number of steps (total_ut_steps),
with no input injection between iterations. This differs from Raven/Huginn which has
separate prelude/core/coda stages with input injection via a state adapter.

Each OuroDecoderLayer uses sandwich normalization with 4 RMSNorms:
  - input_layernorm: pre-attention norm
  - input_layernorm_2: post-attention norm
  - post_attention_layernorm: pre-MLP norm
  - post_attention_layernorm_2: post-MLP norm

Attention uses separate Q/K/V projections (not fused), and MLP uses separate
gate_proj/up_proj/down_proj (not fused).
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)

from looped_internals.models._compat.ouro import prepare_ouro_config


class OuroArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Ouro (LoopLM) looped transformer models.

    Ouro loops all layers for total_ut_steps iterations with no input injection.
    Each OuroDecoderLayer has sandwich norms (4 RMSNorms), separate Q/K/V attention,
    and separate gate/up/down MLP projections.

    Optional Parameters (not present in state_dict):
    ------------------------------------------------
    Ouro does NOT have biases on any projections:
    - blocks.*.attn.b_Q/b_K/b_V/b_O - No bias
    - blocks.*.mlp.b_in/b_gate/b_out - No bias
    - All norms are RMSNorm (no bias)
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Ouro architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        self.cfg.eps_attr = "variance_epsilon"

        n_heads = getattr(cfg, "n_heads", getattr(cfg, "num_attention_heads", 16))
        n_kv_heads = (
            getattr(cfg, "n_key_value_heads", None)
            or getattr(cfg, "num_key_value_heads", None)
            or n_heads
        )
        d_model = getattr(
            cfg, "d_model", getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 2048))
        )
        head_dim = getattr(cfg, "d_head", getattr(cfg, "head_dim", d_model // n_heads))
        n_ctx = getattr(cfg, "n_ctx", getattr(cfg, "max_position_embeddings", 65536))

        self.default_cfg: dict[str, Any] = {
            "d_model": d_model,
            "d_head": head_dim,
            "n_heads": n_heads,
            "n_key_value_heads": n_kv_heads,
            "n_ctx": n_ctx,
        }

        if n_kv_heads is not None:
            self.cfg.n_key_value_heads = n_kv_heads

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(
                        name="input_layernorm", config=self.cfg
                    ),
                    "ln2": RMSNormalizationBridge(
                        name="input_layernorm_2", config=self.cfg
                    ),
                    "ln3": RMSNormalizationBridge(
                        name="post_attention_layernorm", config=self.cfg
                    ),
                    "ln4": RMSNormalizationBridge(
                        name="post_attention_layernorm_2", config=self.cfg
                    ),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    "mlp": GatedMLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="gate_proj"),
                            "in": LinearBridge(name="up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Adapt Ouro's remote code before boot_transformers constructs it."""
        if not model_kwargs.get("trust_remote_code", False):
            raise ValueError("Loading Ouro requires trust_remote_code=True")
        prepare_ouro_config(model_kwargs["config"])

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for Ouro component testing.

        Args:
            hf_model: The HuggingFace Ouro model instance.
            bridge_model: The TransformerBridge model (if available).
        """
        rotary_emb = hf_model.model.rotary_emb

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
