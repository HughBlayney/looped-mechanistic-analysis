"""Reduce cached transformer activations to metrics for each input example.

This module connects ``TransformerBridge.run_with_cache`` to the measurements
in :mod:`looped_internals.analysis` and the result containers in
:mod:`looped_internals.results`. Use :func:`metrics_names_filter` when collecting
a cache, then pass that cache, the input token IDs, and the model's
:class:`~looped_internals.structure.ModelLayerStructure` to
:func:`compute_model_run_data`. The result is one :class:`ModelRunData` per
batch example, with a :class:`PerVirtualLayerMetrics` record for each cached
virtual layer in forward-pass order.

A layer identifies a distinct set of transformer-block weights. A virtual
layer is one application of those weights. Recurrent applications share a
``layer_index`` and have different ``virtual_layer_index`` values.

Each virtual layer contributes the first-token residual norm and, when attention
patterns are available, per-head sink scores, mixing scores, and column-sum
concentrations. Optional outputs include residual spectral entropy, comparisons
between virtual layers, comparisons with each layer's final observed state, and raw
residual, attention, and Q/K/V tensors for downstream analysis and visualization.
Private helpers resolve cache keys and head dimensions, enumerate virtual layers,
and compare activation pairs without stacking the full activation history.

Notes
-----
In the shapes below, ``B`` is batch size, ``T`` is sequence length, ``D`` is
residual width, ``H`` is the number of query heads, ``H_kv`` is the number of
key/value heads, and ``D_head`` is head width. Layer outputs have shape
``(B, T, D)``; post-softmax attention patterns have shape ``(B, H, T, T)``.
Keep the batch dimension when caching. Missing attention patterns leave the
corresponding virtual layer's attention metrics unset; attention comparisons require
patterns for every extracted virtual layer.

For recurrent models, use ``accumulate=True`` to retain each virtual layer under
an ``.iter_N`` cache key. Extraction prefers these keys over the unsuffixed
entry. Without accumulated entries, only the unsuffixed cached state of each
layer is available.

Results reside on the CPU. Caching with ``device="cpu"`` also keeps the retained
activation history off the accelerator. After reduction, release the cache to
free activations that the results do not reference. With
``store_raw_values=True``, raw CPU tensors can share cache storage and keep it
alive. To retain Q/K/V as well as residuals and attention patterns, cache their
projection hooks using ``names_filter=None`` or a custom filter;
:func:`metrics_names_filter` excludes them.

Examples
--------
Given a ``model`` and ``structure`` returned by
:func:`~looped_internals.models.load_model`, and batched ``input_ids`` on the
model's device, collect and inspect metrics as follows:

>>> import torch
>>> from looped_internals.capture import (
...     compute_model_run_data, metrics_names_filter,
... )
>>> with torch.no_grad():
...     _, cache = model.run_with_cache(
...         input_ids, accumulate=True,
...         names_filter=metrics_names_filter, device="cpu",
...     )
>>> runs = compute_model_run_data(
...     cache, input_ids, structure,
...     compute_pairwise=False, compute_differences=True,
... )
>>> del cache
>>> first_example_norms = runs[0].get_metric("residual_norms")
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict, cast

import torch
from torch import Tensor
from tqdm.auto import tqdm

from looped_internals.analysis import (
    get_colsum_concentrations,
    get_mixing_scores,
    get_residual_entropy,
    get_sink_scores,
)
from looped_internals.analysis.comparisons import (
    compare_pairwise_residuals,
    compare_residuals_to_final_state,
    compare_successive_recurrence_residuals,
    compute_pairwise_attn_pattern_diff_norm,
    compute_successive_recurrence_attn_pattern_diff_norm,
    compute_to_final_state_attn_pattern_diff_norm,
)
from looped_internals.results import ModelRunData, PerVirtualLayerMetrics
from looped_internals.structure import ModelLayerStructure, VirtualLayer

if TYPE_CHECKING:
    from transformer_lens.ActivationCache import ActivationCache
    from transformer_lens.model_bridge import TransformerBridge

logger = logging.getLogger(__name__)


class _ComparisonMetrics(TypedDict, total=False):
    """Optional comparison fields, grouped by the states being compared."""

    pairwise_residual_cosine_sim: Tensor
    pairwise_residual_diff_norm: Tensor
    pairwise_attn_pattern_diff_norm: Tensor | None
    to_final_state_residual_cosine_sim: Tensor
    to_final_state_residual_diff_norm: Tensor
    to_final_state_attn_pattern_diff_norm: Tensor | None
    successive_recurrence_residual_cosine_sim: list[Tensor]
    successive_recurrence_residual_diff_norm: list[Tensor]
    successive_recurrence_attn_pattern_diff_norm: list[Tensor] | None


# ---------------------------------------------------------------------------
# Cache filter
# ---------------------------------------------------------------------------


def metrics_names_filter(name: str) -> bool:
    """Select layer outputs and attention patterns for metrics extraction.

    Parameters
    ----------
    name : str
        Canonical hook name supplied by ``run_with_cache`` before any
        ``.iter_N`` suffix is added to the stored cache key.

    Returns
    -------
    bool
        Whether the name matches a layer output, ``{prefix}.{N}.hook_out``,
        or an attention pattern, ``{prefix}.{N}.attn.hook_pattern``, with
        a numeric layer index ``N``. Other hooks are excluded.

    Notes
    -----
    Pass this callable as ``names_filter`` to ``run_with_cache``. The selected
    hooks suffice for all summary metrics and for storing raw residuals and
    attention patterns. Q/K/V projection hooks are excluded; use
    ``names_filter=None`` or a custom filter if those raw values are needed
    by :func:`compute_model_run_data`. Already suffixed cache keys do not match.

    Examples
    --------
    >>> metrics_names_filter("blocks.0.hook_out")
    True
    >>> metrics_names_filter("core_blocks.2.attn.hook_pattern")
    True
    >>> metrics_names_filter("blocks.0.attn.q.hook_out")
    False
    """
    parts = name.split(".")
    n = len(parts)
    # Block residual stream: {prefix}.{N}.hook_out
    if n >= 3 and parts[-1] == "hook_out" and parts[-2].isdigit():
        return True
    # Attention pattern: {prefix}.{N}.attn.hook_pattern
    for i in range(n - 1):
        if parts[i].isdigit() and i + 1 < n and parts[i + 1] == "attn":
            if parts[i + 2 :] == ["hook_pattern"]:
                return True
    return False


# ---------------------------------------------------------------------------
# Attention pattern extraction
# ---------------------------------------------------------------------------


def _get_attn_pattern(cache: ActivationCache, layer_key: str) -> Tensor | None:
    """Look up a layer's unsuffixed attention pattern without copying it.

    Parameters
    ----------
    cache : ActivationCache
        Cache exposing a ``cache_dict`` mapping hook names to tensors.
    layer_key : str
        Layer prefix, such as ``"blocks.0"``, without a hook name.

    Returns
    -------
    torch.Tensor or None
        The value at ``{layer_key}.attn.hook_pattern``, expected to contain
        post-softmax probabilities of shape ``(B, H, T, T)``. Returns ``None``
        if the entry is absent or explicitly ``None``. No iteration lookup,
        shape validation, or device conversion is performed.
    """
    key = f"{layer_key}.attn.hook_pattern"
    cd = cache.cache_dict
    val = cd.get(key)
    return val if val is not None else None


def _get_attn_pattern_iter(
    cache: ActivationCache, layer_key: str, iter_idx: int
) -> Tensor | None:
    """Look up an attention pattern for one accumulated virtual layer.

    Parameters
    ----------
    cache : ActivationCache
        Cache exposing a ``cache_dict`` mapping hook names to tensors.
    layer_key : str
        Layer prefix, such as ``"core_blocks.0"``, without a hook name.
    iter_idx : int
        Zero-based recurrence index used in the ``.iter_N`` cache suffix.

    Returns
    -------
    torch.Tensor or None
        The value at ``{layer_key}.attn.hook_pattern.iter_{iter_idx}``,
        expected to have shape ``(B, H, T, T)``. Returns ``None`` if the entry
        is absent or explicitly ``None``, without falling back to the
        unsuffixed key. The cached tensor is returned without copying,
        validating its shape, or changing its device.
    """
    key = f"{layer_key}.attn.hook_pattern.iter_{iter_idx}"
    cd = cache.cache_dict
    val = cd.get(key)
    return val if val is not None else None


# ---------------------------------------------------------------------------
# Raw Q/K/V extraction
# ---------------------------------------------------------------------------


def _get_head_info(
    cache: ActivationCache,
) -> tuple[int | None, int | None, int | None]:
    """Read the adapter dimensions needed to reshape Q/K/V projections.

    Parameters
    ----------
    cache : ActivationCache
        Cache whose ``model.adapter.default_cfg`` provides attention geometry.

    Returns
    -------
    n_heads : int or None
        Number of query heads, from ``n_heads`` in the adapter configuration.
    n_kv_heads : int or None
        Number of key/value heads, from ``n_key_value_heads``. Defaults to
        ``n_heads`` when that setting is absent or false-valued.
    d_head : int or None
        Width of each attention head, from ``d_head`` in the configuration.

    Notes
    -----
    Returns ``(None, None, None)`` if a required attribute is unavailable or
    either ``n_heads`` or ``d_head`` is absent or false-valued. Dimensions are
    not inferred from cached tensor shapes.
    """
    try:
        # ActivationCache types its model as HookedTransformer, even for bridges.
        bridge = cast("TransformerBridge", cache.model)
        default_cfg = bridge.adapter.default_cfg
        n_heads: int | None = default_cfg.get("n_heads")
        n_kv_heads: int | None = default_cfg.get("n_key_value_heads") or n_heads
        d_head: int | None = default_cfg.get("d_head")
        if n_heads and d_head:
            return n_heads, n_kv_heads or n_heads, d_head
    except AttributeError:
        pass
    return None, None, None


def _get_raw_qkv(
    cache: ActivationCache,
    virtual_layer: VirtualLayer,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Extract separate or fused Q/K/V projection outputs for one virtual layer.

    Parameters
    ----------
    cache : ActivationCache
        Cache with projection tensors in ``cache_dict`` and an adapter
        configuration accessible through ``model.adapter.default_cfg`` when
        flattened projections need reshaping.
    virtual_layer : VirtualLayer
        Layer prefix and iteration identifying the cached application.
        An iteration of ``None`` selects unsuffixed entries.

    Returns
    -------
    tuple of torch.Tensor or None
        ``(q, k, v)`` with expected shapes ``(B, T, H, D_head)`` for queries
        and ``(B, T, H_kv, D_head)`` for keys and values. Returns ``None``
        when no supported complete projection set is available or the head
        dimensions needed for reshaping are unknown. Tensors retain their
        cached dtype and device and share storage with the cached projections.

    Raises
    ------
    RuntimeError
        If a flattened projection cannot be viewed with the configured head
        dimensions, for example because its shape or strides are incompatible.

    Notes
    -----
    A complete, non-``None`` set of ``attn.q.hook_out``, ``attn.k.hook_out``,
    and ``attn.v.hook_out`` takes precedence. If the query is three-dimensional,
    all three projections are reshaped using :func:`_get_head_info`; otherwise
    they are returned unchanged, without shape validation. If this complete
    set is unavailable, a three-dimensional ``attn.qkv.hook_out`` is split
    along its last axis in Q, K, V order and reshaped. Each hook name is
    resolved through the corresponding :class:`VirtualLayer` key property.

    These are raw projection outputs; this helper does not apply positional
    rotations, normalization, or key/value head repetition. Missing head
    dimensions for a complete flattened separate set return ``None`` without
    trying the fused hook.
    """
    cd = cache.cache_dict

    # 1. Separate Q/K/V projections (Minimal, Ouro)
    q = cd.get(virtual_layer.q_key)
    k = cd.get(virtual_layer.k_key)
    v = cd.get(virtual_layer.v_key)
    if q is not None and k is not None and v is not None:
        if q.dim() == 3:
            n_heads, n_kv_heads, d_head = _get_head_info(cache)
            if n_heads is None or n_kv_heads is None or d_head is None:
                return None
            B, T, _ = q.shape
            q = q.view(B, T, n_heads, d_head)
            k = k.view(B, T, n_kv_heads, d_head)
            v = v.view(B, T, n_kv_heads, d_head)
        return q, k, v

    # 2. Fused QKV projection (Raven)
    qkv = cd.get(virtual_layer.qkv_key)
    if qkv is not None and qkv.dim() == 3:
        n_heads, n_kv_heads, d_head = _get_head_info(cache)
        if n_heads is None or n_kv_heads is None or d_head is None:
            return None
        B, T, _ = qkv.shape
        q_dim = n_heads * d_head
        k_dim = n_kv_heads * d_head
        q = qkv[..., :q_dim].view(B, T, n_heads, d_head)
        k = qkv[..., q_dim : q_dim + k_dim].view(B, T, n_kv_heads, d_head)
        v = qkv[..., q_dim + k_dim :].view(B, T, n_kv_heads, d_head)
        return q, k, v

    return None


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def _cached_virtual_layers(
    cache: ActivationCache, structure: ModelLayerStructure
) -> list[VirtualLayer]:
    """Enumerate cached virtual layers in the model's forward-pass order.

    Parameters
    ----------
    cache : ActivationCache
        Cache whose ``cache_dict`` contains layer ``hook_out`` entries,
        optionally with ``.iter_N`` suffixes, where ``N`` is a non-negative
        integer written in ASCII decimal digits without leading zeros.
    structure : ModelLayerStructure
        Ordered layer groups defining hook prefixes, layer counts, and which
        groups execute recurrently.

    Returns
    -------
    list of VirtualLayer
        Virtual layer descriptors. Groups follow ``structure.groups`` order. Recurrent
        groups are ordered by iteration, then layer. Non-recurrent groups
        follow layer order, with at most one cached application per layer.
        ``layer_index`` counts layers across all groups, including uncached
        layers. ``virtual_layer_index`` counts only returned virtual layers,
        starting at zero.

    Raises
    ------
    ValueError
        If a layer's ``hook_out`` cache key has a suffix other than the
        ``.iter_N`` format above, or a layer in a non-recurrent group has
        multiple cached iterations.

    Notes
    -----
    For each layer, numerically sorted iteration keys take precedence over
    the unsuffixed key. If no iteration keys exist, the unsuffixed entry is
    used when present. Layers without either form are omitted. Enumeration
    inspects keys only and does not read, copy, or stack activation tensors.
    """
    entries = []
    layer_offset = 0
    for group in structure.groups:
        group_entries: list[tuple[int | None, int]] = []
        for layer in range(group.n_layers):
            key = f"{group.hook_prefix}.{layer}.hook_out"
            iterations = []
            for name in cache.cache_dict:
                if name == key or not name.startswith(key):
                    continue
                match = re.fullmatch(r"\.iter_(0|[1-9][0-9]*)", name[len(key) :])
                if match is None:
                    raise ValueError(
                        f"Invalid iteration cache key {name!r}: expected "
                        f"'{key}.iter_N', where N is a non-negative integer "
                        "without leading zeros"
                    )
                iterations.append(int(match[1]))
            if not group.is_recurrent and len(iterations) > 1:
                raise ValueError(
                    f"Non-recurrent group '{group.hook_prefix}' has multiple "
                    f"cached iterations for layer {layer}: {sorted(iterations)}"
                )
            if iterations:
                group_entries.extend((iteration, layer) for iteration in iterations)
            elif key in cache.cache_dict:
                group_entries.append((None, layer))
        if group.is_recurrent:
            group_entries.sort(key=lambda item: (item[0] or 0, item[1]))
        for iteration, layer in group_entries:
            entries.append(
                VirtualLayer(
                    layer_offset + layer,
                    len(entries),
                    f"{group.hook_prefix}.{layer}",
                    iteration,
                )
            )
        layer_offset += group.n_layers
    return entries


def _comparisons(
    residuals: Sequence[Tensor],
    attentions: Sequence[Tensor | None],
    layers: Sequence[int],
    compute_pairwise: bool,
    compute_differences: bool,
) -> _ComparisonMetrics:
    """Collect CPU comparison metrics for one example, grouped by pair selection.

    Residuals are ``(T, D)`` tensors in virtual-layer execution order. Each has
    a corresponding ``(H, T, T)`` attention pattern or ``None``, and a layer
    index. Vector comparisons always retain both cosine similarity and the
    Euclidean difference norm. Attention comparisons retain Frobenius norms.

    ``compute_pairwise`` selects every pair, including self-pairs: residual
    outputs have shape ``(S, S, T)`` and attention outputs ``(S, S, H)``.
    ``compute_differences`` independently selects two families:

    * ``to_final_state`` compares every virtual layer with each layer's last
      observed state: ``(S, L, T)`` / ``(S, L, H)``. Unobserved layer slots are
      ``NaN``; a last observed state need not be a converged fixed point.
    * ``successive_recurrence`` compares consecutive applications of the same
      layer: one tensor per layer, ``(max(R_l-1, 0), T)`` / ``(max(R_l-1, 0), H)``.

    Disabled families are omitted; missing attention anywhere makes requested
    attention fields ``None``. Fewer than two residuals produce no comparisons.
    Shared analysis helpers process individual pairs without stacking history.
    """
    if len(residuals) < 2:
        return {}
    output: _ComparisonMetrics = {}
    has_attention = all(pattern is not None for pattern in attentions)
    attention_patterns = cast(Sequence[Tensor], attentions) if has_attention else ()

    if compute_pairwise:
        pairwise_residual = compare_pairwise_residuals(residuals, output_device="cpu")
        output.update(
            pairwise_residual_cosine_sim=pairwise_residual.cosine_sim,
            pairwise_residual_diff_norm=pairwise_residual.diff_norm,
            pairwise_attn_pattern_diff_norm=(
                compute_pairwise_attn_pattern_diff_norm(
                    attention_patterns, output_device="cpu"
                )
                if has_attention
                else None
            ),
        )

    if compute_differences:
        to_final_state_residual = compare_residuals_to_final_state(
            layers, residuals, output_device="cpu"
        )
        output.update(
            to_final_state_residual_cosine_sim=to_final_state_residual.cosine_sim,
            to_final_state_residual_diff_norm=to_final_state_residual.diff_norm,
            to_final_state_attn_pattern_diff_norm=(
                compute_to_final_state_attn_pattern_diff_norm(
                    layers, attention_patterns, output_device="cpu"
                )
                if has_attention
                else None
            ),
        )

        successive_recurrence_residual = compare_successive_recurrence_residuals(
            layers, residuals, output_device="cpu"
        )
        output.update(
            successive_recurrence_residual_cosine_sim=[
                comparison.cosine_sim for comparison in successive_recurrence_residual
            ],
            successive_recurrence_residual_diff_norm=[
                comparison.diff_norm for comparison in successive_recurrence_residual
            ],
            successive_recurrence_attn_pattern_diff_norm=(
                compute_successive_recurrence_attn_pattern_diff_norm(
                    layers, attention_patterns, output_device="cpu"
                )
                if has_attention
                else None
            ),
        )
    return output


@torch.no_grad()
def compute_model_run_data(
    cache: ActivationCache,
    input_ids: Tensor,
    structure: ModelLayerStructure,
    compute_pairwise: bool = True,
    compute_differences: bool = False,
    store_raw_values: bool = False,
    compute_residual_entropy: bool = False,
    residual_entropy_components: int | None = None,
    verbose: bool = False,
) -> list[ModelRunData]:
    """Reduce a batch activation cache to one CPU metrics record per example.

    Parameters
    ----------
    cache : ActivationCache
        Cache returned by ``TransformerBridge.run_with_cache``, with layer
        outputs of shape ``(B, T, D)`` in ``cache_dict``. Optional attention
        patterns have shape ``(B, H, T, T)``. When generating this cache using
        ``run_with_cache``, retain the batch dimension with
        ``remove_batch_dim=False`` and use ``accumulate=True`` so that later
        applications do not overwrite earlier ones, preserving recurrent
        virtual layers. Iteration keys take precedence over unsuffixed keys;
        layers without cached outputs are skipped. Attention and projection
        hooks must use the same iteration suffix as their layer output.
    input_ids : torch.Tensor of shape (B, T)
        Token IDs corresponding to the cached examples and positions, in
        batch order. Each result stores its example's IDs on the CPU with
        shape ``(1, T)``. Padding positions are included in the measurements;
        this function does not accept or apply an attention mask.
    structure : ModelLayerStructure
        Model layout, normally returned alongside the model by ``load_model``.
        Defines cache prefixes, global layer indices, and virtual layer
        order. Within recurrent groups, virtual layers are ordered by iteration,
        then layer; groups follow the layout's order.
    compute_pairwise : bool, default=True
        Compute ``pairwise_residual_cosine_sim`` and
        ``pairwise_residual_diff_norm``, both shaped ``(S, S, T)``, and,
        when every virtual layer has an attention pattern,
        ``pairwise_attn_pattern_diff_norm`` of shape ``(S, S, H)``. Here ``S`` is the
        number of extracted virtual layers. These symmetric summaries
        include self-comparisons and require quadratic storage in ``S``.
    compute_differences : bool, default=False
        Compute per-token residual L2 distances and cosine similarities, and
        per-head attention Frobenius distances, for two sets of pairs:
        every virtual layer against each layer's final observed state; and
        consecutive applications of the same layer. Independent of ``compute_pairwise``.
    store_raw_values : bool, default=False
        Retain raw CPU residuals ``(1, T, D)`` and available attention patterns
        ``(1, H, T, T)`` in each :class:`PerVirtualLayerMetrics`. Also retain query
        projections ``(1, H, T, D_head)`` and key/value projections
        ``(1, H_kv, T, D_head)`` when :func:`_get_raw_qkv` can extract them.
        Cache projection hooks with ``names_filter=None`` or a custom filter
        to make Q/K/V available; :func:`metrics_names_filter` excludes them.
    compute_residual_entropy : bool, default=False
        Compute the entropy of the normalized squared singular values of
        each residual matrix, in nats, via :func:`get_residual_entropy`.
        Otherwise, each virtual layer's ``residual_entropy`` remains ``None``.
    residual_entropy_components : int or None, default=None
        Component count forwarded to :func:`get_residual_entropy` when
        enabled. ``None`` uses the full float32 Gram spectrum; an integer
        selects the randomized ``torch.pca_lowrank`` path, capped at
        ``min(T, D)``. That path uses PyTorch's default centering, so it
        measures the centered residual
        spectrum. Ignored when ``compute_residual_entropy=False``.
    verbose : bool, default=False
        Display a progress bar for the metric reductions at each virtual layer.
        The bar does not cover subsequent per-example comparisons.

    Returns
    -------
    list of ModelRunData
        One result per input example, preserving batch order. Each result's
        ``per_virtual_layer_metrics`` contains ``S`` records in forward-pass
        order, with a ``layer_index`` and a unique zero-based
        ``virtual_layer_index``. All stored tensors are on the CPU. Fields in
        each record retain a singleton batch dimension: ``residual_norms`` is
        ``(1,)`` and measures the L2
        norm at token position zero. ``residual_norms_last`` and
        ``residual_norms_mean`` have the same shape and measure the last token's
        norm and the mean of all token norms, respectively.
        ``sink_scores``, ``mixing_scores``, and
        ``colsum_concentrations`` are ``(1, H)`` when attention is available.
        Optional ``residual_entropy`` is ``(1,)``. Comparison tensors have no
        batch dimension; see :class:`ModelRunData` for their field layouts.

    Raises
    ------
    ValueError
        If any extracted residual is not three-dimensional or its batch and
        token dimensions do not match ``input_ids.shape``, or if a layer output
        cache key has a suffix that does not follow the ``.iter_N`` format
        described in :func:`_cached_virtual_layers`. Also raised if a layer in
        a non-recurrent group has multiple cached iterations.

    See Also
    --------
    metrics_names_filter : Select hooks required for summary metrics.
    ModelRunData : Result fields, metric grouping, and serialization.
    get_residual_entropy : Full-spectrum and randomized residual entropy reduction.

    Notes
    -----
    If a virtual layer's attention pattern is missing from the cache, its
    attention metrics remain ``None`` in the resulting
    :class:`PerVirtualLayerMetrics`. All attention comparison fields in the
    example's :class:`ModelRunData` also remain ``None``, even if other virtual
    layers have cached attention patterns. Residual metrics remain available.
    If fewer than two virtual layers are found, all
    comparison fields remain ``None`` regardless of the flags. An empty cache
    produces results with empty ``per_virtual_layer_metrics`` lists.

    Final-state comparisons reference the output of each layer's last cached
    application, not a proven fixed point. Their layer axis ends at the highest
    layer index with a cached output; slots for layers with no cached outputs
    contain ``NaN``.
    Per-layer successive comparisons for layers with fewer than two virtual layers
    are empty tensors.

    Extraction runs without gradient tracking, leaves the cache unchanged,
    and reduces each virtual layer independently. Comparisons use float32 arithmetic
    on one pair at a time; the full raw virtual layer history is never stacked.
    This bounds temporary activation storage, but does not remove the memory
    cost of the cache or the requested comparison outputs.

    Raw fields are not cloned: slices and views of CPU cache tensors share
    their storage, while tensors cached on an accelerator are copied to the
    CPU. Retaining raw CPU fields can therefore keep an entire batch's backing
    storage alive after the cache is deleted, and later in-place edits can
    affect the shared tensors. See the module example for the recommended
    cache, reduce, and release workflow.
    """
    entries = _cached_virtual_layers(cache, structure)
    residuals = [cache.cache_dict[e.residual_key] for e in entries]
    attentions = [cache.cache_dict.get(e.attn_key) for e in entries]
    if any(r.ndim != 3 or r.shape[:2] != input_ids.shape for r in residuals):
        raise ValueError(
            "Expected residuals shaped (batch, tokens, hidden), matching input_ids"
        )
    # Small batched reductions, computed once per virtual layer.
    reduced = []
    for residual, attention in tqdm(
        zip(residuals, attentions, strict=True),
        total=len(entries),
        disable=not verbose,
        desc="Reduce cache",
    ):
        token_norms = torch.linalg.vector_norm(residual.float(), dim=-1)
        values = {
            # Copy slices so scalar summaries do not retain the token-norm matrix.
            "residual_norms": token_norms[:, 0].clone().cpu(),
            "residual_norms_last": token_norms[:, -1].clone().cpu(),
            "residual_norms_mean": token_norms.mean(dim=-1).cpu(),
        }
        for name, function in [
            ("sink_scores", get_sink_scores),
            ("mixing_scores", get_mixing_scores),
            ("colsum_concentrations", get_colsum_concentrations),
        ]:
            values[name] = function(attention).cpu() if attention is not None else None
        values["residual_entropy"] = (
            get_residual_entropy(
                residual, num_components=residual_entropy_components
            ).cpu()
            if compute_residual_entropy
            else None
        )
        reduced.append(values)

    results = []
    for batch in range(input_ids.shape[0]):
        metrics = []
        for i, virtual_layer in enumerate(entries):
            values = {
                name: value[batch : batch + 1] if value is not None else None
                for name, value in reduced[i].items()
            }
            if store_raw_values:
                values["raw_residual"] = residuals[i][batch : batch + 1].cpu()
                attention = attentions[i]
                if attention is not None:
                    values["raw_attn_pattern"] = attention[batch : batch + 1].cpu()
                qkv = _get_raw_qkv(cache, virtual_layer)
                if qkv is not None:
                    for name, tensor in zip(
                        ("raw_attn_q", "raw_attn_k", "raw_attn_v"), qkv, strict=True
                    ):
                        values[name] = (
                            tensor[batch : batch + 1].permute(0, 2, 1, 3).cpu()
                        )
            metrics.append(
                PerVirtualLayerMetrics(
                    layer_index=virtual_layer.layer_index,
                    virtual_layer_index=virtual_layer.virtual_layer_index,
                    **values,  # ty: ignore[invalid-argument-type]
                )
            )
        comparisons = _comparisons(
            [r[batch] for r in residuals],
            [a[batch] if a is not None else None for a in attentions],
            [e.layer_index for e in entries],
            compute_pairwise,
            compute_differences,
        )
        results.append(
            ModelRunData(
                input_ids=input_ids[batch : batch + 1].cpu(),
                per_virtual_layer_metrics=metrics,
                **comparisons,
            )
        )
    return results
