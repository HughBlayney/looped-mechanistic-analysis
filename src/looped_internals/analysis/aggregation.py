from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
from torch import Tensor

from looped_internals.results.depth import DepthValues, _build_groups
from looped_internals.results.runs import ModelRunData

_POSITION_LABELS = {
    None: "Virtual layer",
    "virtual_layer": "Virtual layer depth (%)",
    "layer": "Layer depth (%)",
}


def aggregate_values(
    model_data_list: list[ModelRunData],
    attribute_name: str,
    group_by: Literal[None, "layer", "recurrence"] = None,
    scale_by: Literal[None, "virtual_layer", "layer"] = None,
) -> OrderedDict[str, DepthValues]:
    """Aggregate a metric across multiple runs, grouped and scaled as requested.

    Calls :meth:`ModelRunData.get_metric` on each model, averages over the
    last dimension (e.g. heads), then groups the per-virtual-layer values and
    computes mean and std across all models.

    All ``ModelRunData`` objects are assumed to be homogeneous (same
    architecture, same total virtual layers S).

    Args:
        model_data_list: One or more runs through the same architecture.
        attribute_name: :class:`PerVirtualLayerMetrics` field name, e.g.
            ``"sink_scores"``.
        group_by: How to group virtual layers in the output.

            * ``None`` — no grouping; all S virtual layers in one ``"all"`` entry.
            * ``"layer"`` — group by layer index; keys ``"L0"``, ``"L1"``, …
            * ``"recurrence"`` — group by recurrent iteration; keys ``"R0"``,
              ``"R1"``, … with optional ``"Prelude"`` / ``"Coda"`` entries for
              non-recurrent layers detected at the start/end of the model.

        scale_by: How to compute the ``positions`` axis.

            * ``None`` — raw virtual layer indices (integers as float).
            * ``"virtual_layer"`` — virtual layer index as a percentage of the
              maximum virtual layer index (global denominator; positions
              comparable across groups).
            * ``"layer"`` — layer index as a percentage of the maximum layer
              index (global denominator).

    Returns:
        An :class:`~collections.OrderedDict` mapping group keys to
        :class:`DepthValues` instances.  The ``position_label`` field of each
        entry reflects the chosen ``scale_by`` mode.
    """

    # ------------------------------------------------------------------
    # Step 1: extract all values in virtual-layer order → (N, S)
    # ------------------------------------------------------------------
    def _reduce(t: Tensor) -> Tensor:
        t = t.float()
        return t if t.ndim == 1 else t.mean(dim=-1)

    stacked = torch.stack(
        [_reduce(m.get_metric(attribute_name)) for m in model_data_list],
        dim=0,
    )

    # ------------------------------------------------------------------
    # Step 2: build global position tensor (S,)
    # ------------------------------------------------------------------
    pbms = model_data_list[0].per_virtual_layer_metrics
    virtual_layer = torch.tensor(
        [p.virtual_layer_index for p in pbms], dtype=torch.float
    )
    layer = torch.tensor([p.layer_index for p in pbms], dtype=torch.float)

    if scale_by is None:
        all_pos = virtual_layer
    elif scale_by == "virtual_layer":
        all_pos = virtual_layer / virtual_layer.max().clamp(min=1.0) * 100.0
    elif scale_by == "layer":
        all_pos = layer / layer.max().clamp(min=1.0) * 100.0
    else:
        raise ValueError("scale_by must be None, 'virtual_layer', or 'layer'")

    # ------------------------------------------------------------------
    # Step 3: map each group to its virtual layer indices
    # ------------------------------------------------------------------
    if group_by is None:
        groups: OrderedDict[str, list[int]] = OrderedDict(
            {"all": list(range(len(pbms)))}
        )
    else:
        groups = _build_groups(
            pbms, group_by, model_data_list[0].recurrent_layer_indices
        )

    # ------------------------------------------------------------------
    # Step 4: assemble result
    # ------------------------------------------------------------------
    label = _POSITION_LABELS[scale_by]
    result: OrderedDict[str, DepthValues] = OrderedDict()
    for key, indices in groups.items():
        idx = torch.tensor(indices, dtype=torch.long)
        vals = stacked[:, indices]  # (N, group_size)
        result[key] = DepthValues(
            positions=all_pos[idx],
            position_label=label,
            mean=vals.mean(dim=0),
            std=vals.std(dim=0),
        )
    return result
