"""Attention capture must not change PyTorch's output, gradients, or RNG."""

import pytest
import torch
import torch.nn.functional as F

from looped_internals.capture.attention import capture_sdpa_patterns


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gqa", [False, True])
def test_capture_preserves_kernel_output_and_gradients(dtype, gqa):
    torch.manual_seed(42)
    q = torch.randn(1, 4, 3, 8, dtype=dtype, requires_grad=True)
    k = torch.randn(1, 2 if gqa else 4, 3, 8, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    options = dict(is_causal=True, enable_gqa=gqa, dropout_p=0.2)
    torch.manual_seed(7)
    expected = F.scaled_dot_product_attention(q, k, v, **options)
    expected_grads = torch.autograd.grad(expected.sum(), (q, k, v))
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(7)
    with capture_sdpa_patterns() as patterns:
        actual = F.scaled_dot_product_attention(q, k, v, **options)
    actual_grads = torch.autograd.grad(actual.sum(), (q, k, v))
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for left, right in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert len(patterns) == 1
    assert not patterns[0].requires_grad
    assert patterns[0].dtype == dtype
    torch.testing.assert_close(patterns[0].sum(-1), torch.ones(1, 4, 3, dtype=dtype))
    assert patterns[0].triu(1).count_nonzero() == 0


@pytest.mark.parametrize("boolean_mask", [False, True])
def test_fully_masked_rows_are_zero(boolean_mask):
    q = torch.randn(1, 2, 3, 4)
    mask = torch.ones(3, 3, dtype=torch.bool)
    mask[-1] = False
    if not boolean_mask:
        mask = torch.zeros(3, 3).masked_fill(~mask, float("-inf"))
    with capture_sdpa_patterns() as patterns:
        output = F.scaled_dot_product_attention(q, q, q, attn_mask=mask)
    assert output[:, :, -1].count_nonzero() == 0
    assert patterns[0][:, :, -1].count_nonzero() == 0
    assert patterns[0].isfinite().all()


def test_capture_restores_kernel_after_exception():
    original = F.scaled_dot_product_attention
    with pytest.raises(RuntimeError, match="deliberate"):
        with capture_sdpa_patterns():
            raise RuntimeError("deliberate")
    assert F.scaled_dot_product_attention is original
