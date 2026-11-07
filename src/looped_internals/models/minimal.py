"""Configurable recurrent transformer for controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MinimalRecurrentConfig:
    """Configuration for :class:`MinimalRecurrentLM`.

    All norm flags default to ``True`` (full sandwich normalization).
    Set individual flags to ``False`` to ablate specific norms.
    """

    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    num_cycles: int = 4
    d_vocab: int = 32000
    expansion_factor: int = 4
    input_injection: str | None = "projection"
    bias: bool = False
    gated_mlp: bool = True
    activation: str = (
        "silu"  # "silu", "relu", or "gelu"; only used when gated_mlp=False
    )
    # Per-block norm flags
    norm_pre_attn: bool = True
    norm_post_attn: bool = True
    norm_pre_mlp: bool = True
    norm_post_mlp: bool = True
    # Post-residual norm flags (norm applied after residual addition)
    norm_post_attn_residual: bool = True
    norm_post_mlp_residual: bool = True
    # Norm applied after all blocks complete each cycle
    norm_post_cycle: bool = True
    # Variance of the random normal used to initialise the latent z
    initialization_variance: float = 1.0

    # A debugging variable to be removed
    residual_scaling: float = 1.0


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with separate Q/K/V projections."""

    def __init__(self, d_model: int, n_heads: int, bias: bool = False) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class GatedMLP(nn.Module):
    """SwiGLU-style gated MLP with separate gate/up/down projections."""

    def __init__(
        self, d_model: int, expansion_factor: int = 4, bias: bool = False
    ) -> None:
        super().__init__()
        intermediate = expansion_factor * d_model
        self.gate_proj = nn.Linear(d_model, intermediate, bias=bias)
        self.up_proj = nn.Linear(d_model, intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


_ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}


class FeedForward(nn.Module):
    """Simple 2-layer MLP (non-gated) with configurable activation."""

    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        activation: str = "relu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        intermediate = expansion_factor * d_model
        act_cls = _ACTIVATIONS.get(activation, nn.ReLU)
        self.fc_in = nn.Linear(d_model, intermediate, bias=bias)
        self.act = act_cls()
        self.fc_out = nn.Linear(intermediate, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.act(self.fc_in(x)))


def _maybe_norm(enabled: bool, d_model: int) -> nn.Module:
    return nn.RMSNorm(d_model) if enabled else nn.Identity()


class TransformerBlock(nn.Module):
    """Single transformer block with configurable sandwich normalization."""

    def __init__(self, config: MinimalRecurrentConfig) -> None:
        super().__init__()
        d = config.d_model
        self.norm_pre_attn = _maybe_norm(config.norm_pre_attn, d)
        self.norm_post_attn = _maybe_norm(config.norm_post_attn, d)
        self.norm_post_attn_residual = _maybe_norm(config.norm_post_attn_residual, d)
        self.norm_pre_mlp = _maybe_norm(config.norm_pre_mlp, d)
        self.norm_post_mlp = _maybe_norm(config.norm_post_mlp, d)
        self.norm_post_mlp_residual = _maybe_norm(config.norm_post_mlp_residual, d)
        self.attn = CausalSelfAttention(d, config.n_heads, bias=config.bias)
        if config.gated_mlp:
            self.mlp: nn.Module = GatedMLP(d, config.expansion_factor, bias=config.bias)
        else:
            self.mlp = FeedForward(
                d, config.expansion_factor, config.activation, bias=config.bias
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-block with sandwich norms
        residual = x.clone()
        x = self.norm_pre_attn(x)
        x = self.attn(x)
        x = self.norm_post_attn(x)
        x = residual + x
        x = self.norm_post_attn_residual(x)
        # MLP sub-block with sandwich norms
        residual = x.clone()
        x = self.norm_pre_mlp(x)
        x = self.mlp(x)
        x = self.norm_post_mlp(x)
        x = residual + x
        x = self.norm_post_mlp_residual(x)
        return x


class MinimalRecurrentOutput:
    """Simple output container matching the HuggingFace interface."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class MinimalRecurrentLM(nn.Module):
    """Minimal recurrent transformer language model.

    The forward pass embeds tokens, then loops all transformer blocks for
    ``num_cycles`` iterations. ``input_injection`` controls how embeddings are
    re-injected each cycle: ``"projection"`` concatenates embeddings with the
    hidden state and projects (Raven-style); ``"sum"`` adds embeddings directly
    to the hidden state; ``None`` feeds hidden states back without injection
    (Ouro-style).

    Args:
        config: Model configuration.
    """

    def __init__(self, config: MinimalRecurrentConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.d_vocab, config.d_model)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        if config.input_injection == "projection":
            self.input_proj = nn.Linear(
                config.d_model * 2, config.d_model, bias=config.bias
            )
        self.cycle_norm = _maybe_norm(config.norm_post_cycle, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.d_vocab, bias=False)

    def recurrent_step(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply one recurrence to latent state ``z`` with fixed embeddings ``x``."""
        z = self.config.residual_scaling * z
        if self.config.input_injection == "projection":
            h = self.input_proj(torch.cat([x, z], dim=-1))
        elif self.config.input_injection == "sum":
            h = x + z
        else:
            h = z
        for block in self.transformer_layers:
            h = block(h)
        return self.cycle_norm(h)

    def forward_no_logits(
        self,
        x: torch.Tensor,
        num_cycles: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass that returns the final hidden state instead of logits."""
        num_cycles = num_cycles if num_cycles is not None else self.config.num_cycles

        if self.config.input_injection in ("projection", "sum"):
            z = torch.randn_like(x) * self.config.initialization_variance
        else:
            z = x.clone()

        for _ in range(num_cycles):
            z = self.recurrent_step(z, x)

        return z

    def forward(
        self,
        input_ids: torch.Tensor,
        num_cycles: int | None = None,
        **kwargs,
    ) -> MinimalRecurrentOutput:
        x = self.embed_tokens(input_ids)
        z = self.forward_no_logits(x, num_cycles=num_cycles, **kwargs)
        logits = self.lm_head(z)
        return MinimalRecurrentOutput(logits=logits)
