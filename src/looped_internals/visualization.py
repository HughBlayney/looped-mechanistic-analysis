# Copyright (c) 2025 Hugh Blayney. MIT license: licenses/transformer-tools.txt
"""Visualisation utilities for transformer / SSM attention and residual streams.

All functions accept plain tensors (typically retrieved from a TransformerLens
``ActivationCache``) and return either Matplotlib figures or HTML strings.

Typical usage with a loaded ``model`` and ``structure``::

    from looped_internals.capture import compute_model_run_data, metrics_names_filter

    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_ids, accumulate=True, names_filter=metrics_names_filter
        )
    run = compute_model_run_data(
        cache, input_ids, structure, compute_pairwise=False, store_raw_values=True
    )[0]
    attn_per_virtual_layer = [
        metrics.raw_attn_pattern.squeeze(0)
        for metrics in run.per_virtual_layer_metrics
    ]

    fig, axes = create_floorplan(attn_per_virtual_layer)
    fig.savefig("floorplan.png")
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm
from tqdm import tqdm

__all__ = [
    "create_floorplan",
    "create_interactive_attention_visualisation",
    "create_looped_floorplan",
    "plot_attribute",
]


def _resolve_color_scale(
    tensors: list[torch.Tensor],
    vmin: float | None,
    vmax: float | None,
    cmap: str | None,
) -> tuple["Normalize | TwoSlopeNorm", str]:
    """Compute a Matplotlib colour normaliser and resolve the colormap name.

    Selects a sequential map (``"YlGnBu"``) for non-negative data or a
    diverging map (``"RdBu_r"``) when negative values are present.  Pass
    ``cmap`` explicitly to override.
    """
    resolved_vmin = min(t.min().item() for t in tensors) if vmin is None else vmin
    resolved_vmax = max(t.max().item() for t in tensors) if vmax is None else vmax
    if resolved_vmax == 0.0:
        resolved_vmax = 1.0  # guard against all-zero tensors
    has_negatives = resolved_vmin < 0
    if cmap is None:
        cmap = "RdBu_r" if has_negatives else "YlGnBu"
    norm: Normalize | TwoSlopeNorm = (
        TwoSlopeNorm(vcenter=0, vmin=resolved_vmin, vmax=resolved_vmax)
        if has_negatives
        else Normalize(vmin=resolved_vmin, vmax=resolved_vmax)
    )
    return norm, cmap


def plot_attribute(
    axes: plt.Axes,
    values: torch.Tensor,
    attribute_name: str,
    rescale_to_percentage_depth: bool = False,
    label: str | None = None,
) -> None:
    """Plot a per-virtual-layer scalar attribute on a Matplotlib ``Axes``.

    Args:
        axes: Matplotlib ``Axes`` to draw on.
        values: ``(n_virtual_layers,)`` tensor of values to plot (moved to CPU
            automatically).
        attribute_name: Used as the plot title and y-axis label.
        rescale_to_percentage_depth: If ``True``, the x-axis shows fractional
            depth in ``[0, 1]`` instead of integer virtual layer indices.
        label: Legend label.  Passed directly to ``Axes.plot``.
    """
    n = len(values)
    if rescale_to_percentage_depth:
        x_values: list = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    else:
        x_values = list(range(n))
    axes.plot(x_values, values.detach().cpu().numpy(), label=label)
    axes.set_title(attribute_name)
    axes.set_xlabel(
        "Relative virtual layer depth"
        if rescale_to_percentage_depth
        else "Virtual layer"
    )
    axes.set_ylabel(attribute_name)


def create_floorplan(
    attention_patterns_list: list[torch.Tensor],
    tokens: list[str] | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    max_tick_labels: int = 20,
    virtual_layer_labels: list[str] | None = None,
) -> tuple[plt.Figure, "plt.Axes"]:
    """Grid visualisation ("floorplan") of attention patterns across virtual layers and heads.

    Each cell is a heatmap of one head's attention pattern at one virtual layer.
    Virtual layers are stacked vertically with later virtual layers at the **top**;
    heads are
    arranged horizontally.

    Colourmap is chosen automatically:

    * **Sequential** (``"YlGnBu"``) when all values are non-negative — suitable
      for raw attention probabilities.
    * **Diverging** (``"RdBu_r"``) when negative values are present — suitable
      for unnormalised scores, difference maps, etc.  The colour scale is
      symmetrised around zero so that white/neutral always sits at 0.

    Pass ``cmap`` explicitly to override auto-selection.

    Args:
        attention_patterns_list: One entry per virtual layer, each shape
            ``(n_heads, seq_len, seq_len)`` (batch dimension already removed,
            e.g. ``cache["pattern", l].squeeze(0)``).
        tokens: Optional token strings.  When the sequence length is ≤
            ``max_tick_labels``, tick labels are added to the bottom-left cell.
        cmap: Matplotlib colormap name.  ``None`` (default) auto-selects based
            on whether negative values are present.
        vmin: Lower bound of the colour scale.  ``None`` uses the global data
            minimum.
        vmax: Upper bound of the colour scale.  ``None`` uses the global data
            maximum.
        max_tick_labels: Maximum sequence length for which token tick labels are
            shown.  Longer sequences skip labelling to avoid illegible plots.
        virtual_layer_labels: Optional row labels, one per virtual layer.
            Defaults to ``"V0"``, ``"V1"``, …

    Returns:
        ``(fig, axes)`` — Matplotlib figure and 2-D ``ndarray`` of ``Axes``
        with shape ``(n_virtual_layers, n_heads)``.  Row 0 of ``axes`` corresponds to
        the **last** virtual layer (top of the plot, because of the inversion).
    """
    if not attention_patterns_list:
        raise ValueError("attention_patterns_list must not be empty")

    n_virtual_layers = len(attention_patterns_list)
    n_heads = attention_patterns_list[0].shape[0]
    seq_len = attention_patterns_list[0].shape[-1]

    # ── Colour scale ────────────────────────────────────────────────────────
    norm, cmap = _resolve_color_scale(attention_patterns_list, vmin, vmax, cmap)

    # ── Figure layout ────────────────────────────────────────────────────────
    # Extra left margin for layer labels; extra bottom margin for token labels.
    fig, axes = plt.subplots(
        n_virtual_layers,
        n_heads,
        figsize=(max(1.5 * n_heads, 3), max(1.2 * n_virtual_layers, 2)),
    )

    # Normalise to always 2-D for uniform indexing
    if n_virtual_layers == 1:
        axes = axes[None, :]
    if n_heads == 1:
        axes = axes[:, None]

    # Invert vertical order: later virtual layers appear at the top
    axes = axes[::-1, :]

    # ── Head labels (top row) ────────────────────────────────────────────────
    for h in range(n_heads):
        axes[-1, h].set_title(f"H{h}", fontsize=7, pad=2)

    # ── Hide all spines / ticks globally ────────────────────────────────────
    for axrow in axes:
        for ax in axrow:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # ── Draw heatmaps ────────────────────────────────────────────────────────
    for row, attn in tqdm(
        enumerate(attention_patterns_list),
        desc="Drawing virtual layers",
        total=n_virtual_layers,
    ):
        attn_np = attn.detach().cpu().float().numpy()
        for h in range(n_heads):
            axes[row, h].imshow(
                attn_np[h],
                norm=norm,
                cmap=cmap,
                aspect="auto",
                interpolation="nearest",
            )

    # ── Layer labels (left edge of each row) ─────────────────────────────────
    for row in range(n_virtual_layers):
        label = virtual_layer_labels[row] if virtual_layer_labels else f"V{row}"
        axes[row, 0].set_ylabel(label, rotation=0, labelpad=24, va="center", fontsize=7)

    # ── Optional token tick labels (bottom-left cell only) ───────────────────
    if tokens is not None and seq_len <= max_tick_labels:
        label_ax = axes[0, 0]  # bottom-left after inversion
        label_ax.set_xticks(range(seq_len))
        label_ax.set_yticks(range(seq_len))
        label_ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        label_ax.set_yticklabels(tokens, fontsize=6)

    # ── Shared colorbar (right side, with numerical ticks) ───────────────────
    # tight_layout with rect leaves a right-margin strip for the colorbar,
    # avoiding the solver overhead of constrained_layout.
    plt.tight_layout(pad=0.4, rect=(0, 0, 0.88, 1))
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=axes.ravel().tolist(), location="right", shrink=0.8)

    return fig, axes


def create_looped_floorplan(
    attention_patterns_list: list[torch.Tensor],
    num_prelude_layers: int,
    num_core_layers: int,
    num_coda_layers: int,
    num_loops: int,
    gap: int = 1,
    tokens: list[str] | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    max_tick_labels: int = 20,
    log_scale: bool = False,
    virtual_layer_labels: list[str] | None = None,
) -> tuple[plt.Figure, "plt.Axes"]:
    """Floorplan for looped transformer architectures.

    Extends :func:`create_floorplan` by visually separating the **prelude**,
    **core** (each loop shown as its own block), and **coda** sections
    with uniform blank-row gaps and colour-coded spine outlines:

    * **Blue** — prelude layers
    * **Green** — core layers (each loop in its own block)
    * **Red** — coda layers

    Later virtual layers are placed at the **top** of the figure; heads are arranged
    horizontally.

    Args:
        attention_patterns_list: One entry per virtual layer (all sections concatenated
            in order: prelude → core (all loops) → coda), each shape
            ``(n_heads, seq_len, seq_len)`` with the batch dimension removed.
        num_prelude_layers: Number of leading layers belonging to the prelude.
        num_core_layers: Number of layers in **one** loop of the core block.
        num_coda_layers: Number of trailing layers belonging to the coda.
        num_loops: How many times the core block is repeated.  Each loop
            is drawn as a separate green block separated by a blank-row gap.
        gap: Number of blank grid rows inserted between adjacent sections
            (prelude/core/coda) and between consecutive core loops.
            Defaults to ``1``.
        tokens: Optional token strings.  When the sequence length is ≤
            ``max_tick_labels``, tick labels are added to the bottom-left cell.
        cmap: Matplotlib colormap name.  ``None`` (default) auto-selects.
        vmin: Lower bound of the colour scale.  ``None`` uses the global minimum.
        vmax: Upper bound of the colour scale.  ``None`` uses the global maximum.
        max_tick_labels: Maximum sequence length for which token tick labels are
            shown.
        log_scale: If ``True``, apply ``log(x + 1e-10)`` to all attention
            pattern entries before plotting (useful for sparse attention).
        virtual_layer_labels: Optional list of strings (one per virtual layer, not
            counting gap rows) to annotate the left edge of each row.
            Defaults to ``"V0"``, ``"V1"``, …

    Returns:
        ``(fig, axes)`` — Matplotlib figure and 2-D ``ndarray`` of ``Axes``
        with shape ``(total_rows, n_heads)`` where ``total_rows`` includes
        blank gap rows.  Row 0 of ``axes`` corresponds to the **last** virtual layer
        (top of the plot).
    """
    if not attention_patterns_list:
        raise ValueError("attention_patterns_list must not be empty")

    total_core = num_core_layers * num_loops

    # ── Split into sections ───────────────────────────────────────────────────
    draw_list = (
        [torch.log(a.float().clamp(min=1e-10)) for a in attention_patterns_list]
        if log_scale
        else attention_patterns_list
    )
    prelude = draw_list[:num_prelude_layers]
    core = draw_list[num_prelude_layers : num_prelude_layers + total_core]
    coda = draw_list[num_prelude_layers + total_core :]

    n_heads = draw_list[0].shape[0]
    seq_len = draw_list[0].shape[-1]

    # ── Colour scale ─────────────────────────────────────────────────────────
    norm, cmap = _resolve_color_scale(draw_list, vmin, vmax, cmap)

    # ── Total rows including uniform gap rows between blocks ──────────────────
    num_actual_gaps = 0
    if num_prelude_layers > 0 and total_core > 0:
        num_actual_gaps += 1  # prelude | core
    if total_core > 0 and num_loops > 1:
        num_actual_gaps += num_loops - 1  # between core loops
    if num_coda_layers > 0 and (num_prelude_layers > 0 or total_core > 0):
        num_actual_gaps += 1  # core | coda  (or prelude | coda if no core)

    total_rows = (
        num_prelude_layers + total_core + num_coda_layers + gap * num_actual_gaps
    )
    if total_rows == 0:
        raise ValueError("No virtual layers to plot.")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        total_rows,
        n_heads,
        figsize=(max(1.5 * n_heads, 3), max(1.2 * total_rows, 2)),
    )
    if total_rows == 1:
        axes = axes[None, :]
    if n_heads == 1:
        axes = axes[:, None]

    # Invert vertical order: later virtual layers appear at the top.
    axes = axes[::-1, :]

    # Hide all spines/ticks globally; draw_layer re-enables them for real rows.
    for axrow in axes:
        for ax in axrow:
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

    # ── Head labels (top row of the figure) ──────────────────────────────────
    for h in range(n_heads):
        axes[-1, h].set_title(f"H{h}", fontsize=7, pad=2)

    # ── Drawing helpers (following reference data.py closely) ─────────────────
    row = 0
    virtual_layer_idx = 0

    def draw_layer(attn: torch.Tensor, color: str) -> None:
        nonlocal row, virtual_layer_idx
        attn_np = attn.detach().cpu().float().numpy()
        for h in range(n_heads):
            ax = axes[row, h]
            ax.imshow(
                attn_np[h], norm=norm, cmap=cmap, aspect="auto", interpolation="nearest"
            )
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(color)
                sp.set_linewidth(1.5)
        label = (
            virtual_layer_labels[virtual_layer_idx]
            if virtual_layer_labels
            else f"V{virtual_layer_idx}"
        )
        axes[row, 0].set_ylabel(label, rotation=0, labelpad=24, va="center", fontsize=7)
        row += 1
        virtual_layer_idx += 1

    def skip() -> None:
        nonlocal row
        row += gap  # leave these rows blank (spines already hidden)

    # ── Prelude ───────────────────────────────────────────────────────────────
    if num_prelude_layers > 0:
        for attn in prelude:
            draw_layer(attn, "tab:blue")
        if total_core > 0:
            skip()

    # ── Core (each loop in its own block) ─────────────────────────────────────
    if total_core > 0:
        for loop_idx in range(num_loops):
            for attn in core[
                loop_idx * num_core_layers : (loop_idx + 1) * num_core_layers
            ]:
                draw_layer(attn, "tab:green")
            if loop_idx < num_loops - 1:
                skip()
        if num_coda_layers > 0:
            skip()
    elif num_coda_layers > 0 and num_prelude_layers > 0:
        # No core but both prelude and coda exist: insert gap between them.
        skip()

    # ── Coda ─────────────────────────────────────────────────────────────────
    if num_coda_layers > 0:
        for attn in coda:
            draw_layer(attn, "tab:red")

    # ── Optional token tick labels (bottom-left cell only) ────────────────────
    if tokens is not None and seq_len <= max_tick_labels:
        label_ax = axes[0, 0]  # bottom-left after inversion
        label_ax.set_xticks(range(seq_len))
        label_ax.set_yticks(range(seq_len))
        label_ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        label_ax.set_yticklabels(tokens, fontsize=6)

    # ── Shared colorbar ───────────────────────────────────────────────────────
    plt.tight_layout(pad=0.4, rect=(0, 0, 0.88, 1))
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=axes.ravel().tolist(), location="right", shrink=0.8)

    return fig, axes


def create_interactive_attention_visualisation(
    attention_patterns_by_virtual_layer: list[torch.Tensor],
    virtual_layer_indices: list[int],
    head_indices: list[int],
    tokens: list[str],
    max_width: int | None = 700,
) -> str:
    """Interactive HTML attention visualisation using ``circuitsvis``.

    Selects a subset of (virtual layer, head) pairs and renders them as an interactive
    attention-pattern viewer.

    Args:
        attention_patterns_by_virtual_layer: One attention tensor per virtual layer,
            each ``(n_heads, seq_len, seq_len)`` (batch dimension removed).
            Entry ``i`` is accessed as ``attention_patterns_by_virtual_layer[i]``.
        virtual_layer_indices: Indices into ``attention_patterns_by_virtual_layer``
            selecting the virtual layer for each head.
        head_indices: Head index within that virtual layer's attention tensor.  Must
            have the same length as ``virtual_layer_indices``.
        tokens: Token strings to use as row/column labels in the viewer.
        max_width: Maximum width of the HTML ``<div>`` in pixels.

    Returns:
        Raw HTML string containing the interactive visualisation.

    Raises:
        ValueError: If ``virtual_layer_indices`` and ``head_indices`` have different
            lengths, or if any index is out of range.
        ImportError: If ``circuitsvis`` is not installed.
    """
    if len(virtual_layer_indices) != len(head_indices):
        raise ValueError(
            f"virtual_layer_indices (len={len(virtual_layer_indices)}) and head_indices "
            f"(len={len(head_indices)}) must have the same length."
        )

    try:
        from circuitsvis.attention import attention_heads
    except ImportError as exc:
        raise ImportError(
            "circuitsvis is required for create_interactive_attention_visualisation. "
            "Install it with: pip install circuitsvis"
        ) from exc

    labels: list[str] = []
    patterns: list[torch.Tensor] = []

    for virtual_layer_idx, head_idx in zip(
        virtual_layer_indices, head_indices, strict=True
    ):
        labels.append(f"V{virtual_layer_idx}H{head_idx}")
        # attention_patterns_by_virtual_layer[virtual_layer_idx] has shape (H, T, T)
        patterns.append(
            attention_patterns_by_virtual_layer[virtual_layer_idx][head_idx]
        )

    # Stack into (n_selected_heads, T, T)
    patterns_tensor = torch.stack(patterns, dim=0)

    plot_html = attention_heads(
        attention=patterns_tensor,
        tokens=tokens,
        attention_head_names=labels,
    ).show_code()

    return f"<div style='max-width: {max_width}px;'>{plot_html}</div>"
