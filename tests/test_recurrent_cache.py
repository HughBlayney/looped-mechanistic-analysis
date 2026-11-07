"""Regression coverage for the accumulation wrapper around upstream caching."""

from copy import deepcopy

import pytest
import torch
from transformer_lens.model_bridge import TransformerBridge

from looped_internals._initialization import initialize as apply_all_patches
from looped_internals.models import ModelSpec, load_model
from looped_internals.models.minimal import MinimalRecurrentConfig


@pytest.fixture
def bridge():
    model, _ = load_model(
        ModelSpec(
            MinimalRecurrentConfig(
                d_model=16,
                n_heads=2,
                n_layers=2,
                num_cycles=3,
                d_vocab=32,
                initialization_variance=0.0,
            )
        ),
        device="cpu",
    )
    return model


TOKENS = torch.tensor([[1, 2, 3, 4]])
HOOK = "blocks.0.hook_out"


@pytest.mark.parametrize("cache_object", [True, False])
@pytest.mark.parametrize("names_filter", [HOOK, [HOOK], lambda name: name == HOOK])
def test_accumulation_preserves_upstream_cache_options(
    bridge, cache_object, names_filter
):
    options = dict(
        names_filter=names_filter,
        pos_slice=slice(1, 3),
        remove_batch_dim=True,
        return_cache_object=cache_object,
        device="cpu",
    )
    with torch.no_grad():
        expected, plain = bridge.run_with_cache(TOKENS, **options)
        actual, accumulated = bridge.run_with_cache(TOKENS, accumulate=True, **options)
    torch.testing.assert_close(actual, expected)
    data = accumulated.cache_dict if cache_object else accumulated
    assert data[HOOK].shape == (2, 16)
    torch.testing.assert_close(data[HOOK], plain[HOOK])
    torch.testing.assert_close(data[f"{HOOK}.iter_2"], plain[HOOK])
    assert set(data) == {HOOK, *(f"{HOOK}.iter_{i}" for i in range(3))}


def test_snapshots_survive_reused_activation_storage(bridge, monkeypatch):
    buffer = torch.zeros(1, 4, 16)

    def forward(hidden):
        buffer.add_(1)
        return buffer

    monkeypatch.setattr(bridge.blocks[0].original_component, "forward", forward)
    with torch.no_grad():
        _, cache = bridge.run_with_cache(TOKENS, accumulate=True, names_filter=HOOK)
    buffer.fill_(99)
    for index, snapshot in enumerate(cache.get_iterations(HOOK), start=1):
        torch.testing.assert_close(snapshot, torch.full_like(snapshot, index))


def test_user_hooks_survive_and_wrapper_is_idempotent(bridge):
    wrapper = TransformerBridge.run_with_cache
    apply_all_patches()
    assert TransformerBridge.run_with_cache is wrapper
    seen = []

    def observe(value, hook):
        seen.append(value.detach().clone())

    with bridge.hooks(fwd_hooks=[(HOOK, observe)]), torch.no_grad():
        _, cache = bridge.run_with_cache(TOKENS, accumulate=True, names_filter=HOOK)
        assert len(bridge.hook_dict[HOOK].fwd_hooks) == 1
        bridge(TOKENS)
        assert len(seen) == 6
        torch.testing.assert_close(cache.get_iterations(HOOK), torch.stack(seen[:3]))
    assert not bridge.hook_dict[HOOK].fwd_hooks


def test_cleanup_after_forward_exception(bridge):
    def fail(value, hook):
        raise RuntimeError("deliberate failure")

    with bridge.hooks(fwd_hooks=[(HOOK, fail)]):
        with pytest.raises(RuntimeError, match="deliberate failure"):
            bridge.run_with_cache(TOKENS, accumulate=True)
        assert len(bridge.hook_dict[HOOK].fwd_hooks) == 1
    assert bridge.context_level == 0
    assert all(not hp.fwd_hooks for hp in bridge.hook_dict.values())


def test_stop_at_layer_uses_upstream_selection(bridge):
    with torch.no_grad():
        expected, plain = bridge.run_with_cache(TOKENS, stop_at_layer=1)
        actual, cache = bridge.run_with_cache(TOKENS, stop_at_layer=1, accumulate=True)
    torch.testing.assert_close(actual, expected)
    assert not any(key.startswith("blocks.1.") for key in cache)
    for key in plain:
        if isinstance(plain[key], torch.Tensor):
            torch.testing.assert_close(cache[key], plain[key])


def test_gradients_match_upstream_and_accumulate(bridge):
    control = deepcopy(bridge)
    options = dict(return_type="loss", incl_bwd=True, names_filter=HOOK)
    expected, plain = control.run_with_cache(TOKENS, **options)
    actual, cache = bridge.run_with_cache(TOKENS, accumulate=True, **options)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(cache[f"{HOOK}_grad"], plain[f"{HOOK}_grad"])
    assert cache.get_iterations(f"{HOOK}_grad").shape == (3, 1, 4, 16)
    for left, right in zip(bridge.parameters(), control.parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad)


def test_legacy_aliases_have_iteration_entries(bridge):
    # Exercise alias caching without requesting weight preprocessing, which is
    # independent of recurrence and unsupported for this toy architecture.
    bridge.compatibility_mode = True
    alias = "blocks.0.hook_resid_post"
    with torch.no_grad():
        _, cache = bridge.run_with_cache(TOKENS, names_filter=alias, accumulate=True)
    torch.testing.assert_close(cache.get_iterations(alias), cache.get_iterations(HOOK))


def test_retained_hooks_continue_accumulating_into_returned_cache(bridge):
    try:
        with torch.no_grad():
            _, cache = bridge.run_with_cache(
                TOKENS, names_filter=HOOK, accumulate=True, reset_hooks_end=False
            )
            bridge(TOKENS)
        assert cache.get_iterations(HOOK).shape[0] == 6
    finally:
        bridge.reset_hooks()
