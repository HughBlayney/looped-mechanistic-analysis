"""Iteration-aware caching for TransformerLens bridges."""

from __future__ import annotations

import inspect
from functools import partial, wraps

import torch


def patch_activation_cache() -> None:
    """Add get_iterations() method to ActivationCache."""
    from transformer_lens.ActivationCache import ActivationCache

    def get_iterations(self, hook_name: str) -> torch.Tensor:
        """Get all per-iteration cached activations for a hook, stacked along a new leading dim.

        When ``run_with_cache`` is called with ``accumulate=True``, hooks that
        fire multiple times (e.g. recurrent core blocks) store each firing as
        ``{hook_name}.iter_0``, ``{hook_name}.iter_1``, etc.  This method
        collects those entries and returns them stacked into a single tensor of
        shape ``[n_iterations, *activation_shape]``.

        If no ``.iter_*`` entries exist but the plain ``hook_name`` is present,
        the single cached value is returned with an extra leading dimension of
        size 1.

        Args:
            hook_name: The base hook name (without ``.iter_N`` suffix).

        Returns:
            Stacked tensor of shape ``[n_iterations, *activation_shape]``.

        Raises:
            KeyError: If neither iteration entries nor a plain entry exist.
        """
        iterations = []
        i = 0
        while f"{hook_name}.iter_{i}" in self.cache_dict:
            iterations.append(self.cache_dict[f"{hook_name}.iter_{i}"])
            i += 1
        if iterations:
            return torch.stack(iterations, dim=0)
        if hook_name in self.cache_dict:
            return self.cache_dict[hook_name].unsqueeze(0)
        raise KeyError(f"No cache entries found for '{hook_name}'")

    ActivationCache.get_iterations = get_iterations  # ty: ignore[unresolved-attribute]


def patch_bridge() -> None:
    """Add per-iteration snapshots while delegating cache behavior to upstream.

    Ordinary calls use upstream unchanged. Accumulating calls add read-only
    hooks alongside upstream's own hooks, retaining its input handling, aliases,
    gradients, stop_at_layer, and cleanup. Snapshots own their storage because
    recurrent models may reuse or modify activation buffers in place.
    """
    from transformer_lens.model_bridge.bridge import (
        TransformerBridge,
        _pos_axis_for_hook,
        build_alias_to_canonical_map,
    )
    from transformer_lens.utilities import Slice

    original = TransformerBridge.run_with_cache
    if getattr(original, "_looped_accumulation", False):
        return
    signature = inspect.signature(original)

    @wraps(original)
    def run_with_cache(self, *args, accumulate: bool = False, **kwargs):
        if not accumulate:
            return original(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        options = dict(bound.arguments)
        options.update(options.pop("kwargs", {}))
        names_filter = options.get("names_filter")
        aliases = build_alias_to_canonical_map(self.hook_dict)
        if isinstance(names_filter, (str, list)):
            requested = (
                [names_filter] if isinstance(names_filter, str) else names_filter
            )
            selected = set(requested)
            for name in requested:
                target = aliases.get(name, name)
                selected.update(target if isinstance(target, list) else [target])
            matches = selected.__contains__
        elif names_filter is None:

            def matches(name):
                return True
        elif callable(names_filter):
            matches = names_filter
        else:
            raise ValueError(
                "names_filter must be a string, list of strings, or callable"
            )

        history: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        pos_slice = options.get("pos_slice")
        resolved_slice = Slice.unwrap(pos_slice)
        cache_device = options.get("device")
        remove_batch_dim = options.get("remove_batch_dim", False)

        def save(tensor, *, hook, name, point, backward=False):
            if isinstance(tensor, tuple):
                tensor = tensor[0] if tensor else None
            if not isinstance(tensor, torch.Tensor):
                return
            value = tensor.detach()
            if pos_slice is not None and value.ndim >= 2:
                pos_dim = _pos_axis_for_hook(name, point)
                axis = pos_dim if value.ndim >= 1 - pos_dim else 1
                value = resolved_slice.apply(value, dim=axis)
            if remove_batch_dim and value.shape[0] == 1:
                value = value[0]
            value = value.to(device=cache_device, copy=True)
            key = name + ("_grad" if backward else "")
            index = counts.get(key, 0)
            history[f"{key}.iter_{index}"] = value
            counts[key] = index + 1

        fwd_hooks, bwd_hooks = [], []
        for name, point in self.hook_dict.items():
            if not matches(name):
                continue
            try:
                self.check_hooks_to_add(name)
            except ValueError:
                # These hooks cannot fire. Upstream reports explicit requests.
                continue
            fwd_hooks.append((name, partial(save, name=name, point=point)))
            if options.get("incl_bwd", False):
                bwd_hooks.append(
                    (name, partial(save, name=name, point=point, backward=True))
                )

        with self.hooks(
            fwd_hooks=fwd_hooks,
            bwd_hooks=bwd_hooks,
            reset_hooks_end=options.get("reset_hooks_end", True),
            clear_contexts=options.get("clear_contexts", False),
        ):
            output, cache = original(self, *args, **kwargs)
        data = cache.cache_dict if hasattr(cache, "cache_dict") else cache
        # Only retain snapshots of hooks upstream actually cached (e.g. exclude
        # the stopping block's hook_in when stop_at_layer is set).
        for key, count in counts.items():
            if key in data:
                for index in range(count):
                    entry = f"{key}.iter_{index}"
                    data[entry] = history[entry]
        if self.compatibility_mode:
            for alias, targets in aliases.items():
                targets = targets if isinstance(targets, list) else [targets]
                for suffix in ("", "_grad"):
                    for target in targets:
                        key = target + suffix
                        if alias + suffix in data and key in data:
                            for index in range(counts.get(key, 0)):
                                data[f"{alias}{suffix}.iter_{index}"] = data[
                                    f"{key}.iter_{index}"
                                ]
                            break
        history = data  # Preserve retained hooks' access to the returned cache.
        return output, cache

    run_with_cache._looped_accumulation = True  # ty: ignore[unresolved-attribute]
    TransformerBridge.run_with_cache = run_with_cache  # ty: ignore[invalid-assignment]
