from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor

from ._repr import _FastReprMixin


class DepthMetric(Protocol):
    """Coordinates identifying a layer's weights and one application of them."""

    layer_index: int
    virtual_layer_index: int


@dataclass(repr=False)
class DepthValues(_FastReprMixin):
    """Mean and std of a metric across multiple model runs at a set of depth positions.

    Attributes:
        positions: 1-D float tensor of position values for each entry.  The
            meaning depends on the ``scale_by`` argument passed to
            :func:`aggregate_values` — see :data:`_POSITION_LABELS`.
        position_label: Human-readable axis label, e.g. ``"Virtual layer depth (%)"``.
        mean: Tensor of the same length — mean value across runs at each position.
        std: Tensor of the same length — std across runs at each position.
    """

    positions: Tensor
    position_label: str
    mean: Tensor
    std: Tensor


def _recurrent_layer_indices(pbms: Sequence[DepthMetric]) -> list[int]:
    """Return ordered list of layer indices with multiple virtual layers."""
    order: list[int] = []
    counts: dict[int, int] = {}
    for pbm in pbms:
        if pbm.layer_index not in counts:
            order.append(pbm.layer_index)
            counts[pbm.layer_index] = 0
        counts[pbm.layer_index] += 1
    return [idx for idx in order if counts[idx] > 1]


def _get_metric_impl(
    pbms: Sequence[DepthMetric],
    field: str,
    group_by: Literal[None, "layer", "recurrence"],
    recurrent_layer_indices: list[int],
) -> Tensor | OrderedDict[str, Tensor]:
    """Extract *field* from every virtual layer, with optional grouping."""
    tensors: list[Tensor] = []
    for pbm in pbms:
        val: Tensor | None = getattr(pbm, field)
        if val is None:
            raise ValueError(
                f"{type(pbm).__name__}.{field} is None at layer {pbm.layer_index}, "
                f"virtual layer {pbm.virtual_layer_index}."
            )
        if val.shape[0] != 1:
            raise ValueError(
                f"{type(pbm).__name__}.{field} has batch size {val.shape[0]} "
                f"(expected 1) at layer {pbm.layer_index}, "
                f"virtual layer {pbm.virtual_layer_index}."
            )
        tensors.append(val.squeeze(0))

    if group_by is None:
        return torch.stack(tensors, dim=0)

    groups = _build_groups(pbms, group_by, recurrent_layer_indices)
    return OrderedDict(
        {
            key: torch.stack([tensors[i] for i in idxs], dim=0)
            for key, idxs in groups.items()
        }
    )


def _build_groups(
    pbms: Sequence[DepthMetric],
    group_by: Literal["layer", "recurrence"],
    recurrent_layer_indices: list[int],
) -> OrderedDict[str, list[int]]:
    """Return an ordered mapping from group name to virtual layer indices.

    Args:
        pbms: Virtual layers in execution order.
        group_by: ``"layer"`` groups by layer index; ``"recurrence"`` groups by
            recurrent iteration with optional prelude / coda entries.
        recurrent_layer_indices: Layers with multiple virtual layers (from
            :attr:`ModelRunData.recurrent_layer_indices`).

    Returns:
        ``OrderedDict`` mapping group key to a list of indices into *pbms*.
    """
    if group_by not in ("layer", "recurrence"):
        raise ValueError("group_by must be 'layer' or 'recurrence'")
    groups: OrderedDict[str, list[int]] = OrderedDict()

    if group_by == "layer":
        raw: dict[int, list[int]] = {}
        for i, p in enumerate(pbms):
            raw.setdefault(p.layer_index, []).append(i)
        for li in sorted(raw):
            groups[f"L{li}"] = raw[li]

    else:  # "recurrence"
        recurrent_set = set(recurrent_layer_indices)
        if recurrent_set:
            lo, hi = min(recurrent_set), max(recurrent_set)
            prelude_layers = {p.layer_index for p in pbms if p.layer_index < lo}
            coda_layers = {p.layer_index for p in pbms if p.layer_index > hi}
        else:
            prelude_layers = coda_layers = set()

        if prelude_layers:
            groups["Prelude"] = [
                i for i, p in enumerate(pbms) if p.layer_index in prelude_layers
            ]
        layer_count: dict[int, int] = {}
        for i, p in enumerate(pbms):
            if p.layer_index not in recurrent_set:
                continue
            k = layer_count.get(p.layer_index, 0)
            layer_count[p.layer_index] = k + 1
            groups.setdefault(f"R{k}", []).append(i)
        if coda_layers:
            groups["Coda"] = [
                i for i, p in enumerate(pbms) if p.layer_index in coda_layers
            ]

    return groups
