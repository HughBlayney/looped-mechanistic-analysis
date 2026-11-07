"""Observe SDPA attention patterns without replacing its outputs."""

from __future__ import annotations

import contextlib
import math as _math_sdpa

import torch


@contextlib.contextmanager
def capture_sdpa_patterns():
    """Context manager that intercepts ``F.scaled_dot_product_attention`` to capture
    attention patterns (post-softmax, at least float32) as a side effect.

    Within the context, ``torch.nn.functional.scaled_dot_product_attention`` is
    temporarily wrapped to append detached, pre-dropout patterns to a local
    list. Outputs and gradients still come from the original PyTorch kernel.
    ``torch.nn.attention.flex_attention.flex_attention`` is
    also replaced with a stub that raises ``NotImplementedError``.

    Yields a list that is populated with one tensor per SDPA call executed
    inside the block.  For a single attention module, assert ``len(captured) == 1``
    and read ``captured[0]`` after the context exits.

    Example::

        with capture_sdpa_patterns() as captured:
            output = attention_module(x)
        attn_patterns = captured[0]  # shape (B, H, T, T), float32
    """
    import torch.nn.attention.flex_attention as _flex_mod
    import torch.nn.functional as _F

    captured: list[torch.Tensor] = []
    _original_sdpa = _F.scaled_dot_product_attention
    _original_flex = getattr(_flex_mod, "flex_attention", None)

    def _capture_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ) -> torch.Tensor:
        # Preserve the actual output, gradients, dropout RNG and kernel choice.
        output = _original_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        with torch.no_grad():
            # Only recompute the patterns SDPA does not expose. Keep float64
            # precision; promote half/bfloat16 inputs for a stable softmax.
            dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
            q, k = query.to(dtype), key.to(dtype)
            if enable_gqa:
                k = k.repeat_interleave(q.size(-3) // k.size(-3), dim=-3)
            scale_factor = 1 / _math_sdpa.sqrt(q.size(-1)) if scale is None else scale
            scores = (q @ k.transpose(-2, -1)) * scale_factor
            if is_causal:
                allowed = torch.ones(
                    q.size(-2), k.size(-2), dtype=torch.bool, device=q.device
                ).tril()
                scores.masked_fill_(~allowed, float("-inf"))
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    scores.masked_fill_(~attn_mask, float("-inf"))
                else:
                    scores = scores + attn_mask
            pattern = scores.softmax(dim=-1)
            # SDPA returns zero for a completely masked row.
            pattern.masked_fill_(torch.isneginf(scores).all(dim=-1, keepdim=True), 0)
            captured.append(pattern)
        return output

    def _flex_not_implemented(*args, **kwargs):
        raise NotImplementedError(
            "capture_sdpa_patterns() does not support flex_attention; "
            "attention patterns are not accessible through the flex_attention API."
        )

    _F.scaled_dot_product_attention = _capture_sdpa  # ty: ignore[invalid-assignment]
    if _original_flex is not None:
        _flex_mod.flex_attention = _flex_not_implemented  # ty: ignore[invalid-assignment]
    try:
        yield captured
    finally:
        _F.scaled_dot_product_attention = _original_sdpa
        if _original_flex is not None:
            _flex_mod.flex_attention = _original_flex
