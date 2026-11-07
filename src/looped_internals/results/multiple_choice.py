"""Multiple-choice accuracies at each virtual layer depth."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

import torch
from torch import Tensor

from ._repr import _FastReprMixin
from ._serialization import state_dict
from .depth import _get_metric_impl, _recurrent_layer_indices


@dataclass(repr=False)
class DepthAccuracy(_FastReprMixin):
    """Accuracy at one virtual layer.

    ``layer_index`` identifies the weights; ``virtual_layer_index`` identifies
    their application in the forward pass.
    """

    layer_index: int
    virtual_layer_index: int
    logitlens_accuracy: Tensor
    logitlens_norm_accuracy: Tensor
    probe_accuracy: Tensor
    # None when not requested; NaN at observed coda layers (no second coda pass).
    coda_accuracy: Tensor | None = None
    # Parcae logit lens with prelude/core states projected through C, if requested.
    projected_logitlens_accuracy: Tensor | None = None


@dataclass(repr=False)
class DepthEvaluationResult(_FastReprMixin):
    """Multiple-choice accuracy at each virtual layer depth.

    Stores ``logitlens_accuracy``, ``logitlens_norm_accuracy`` and
    ``probe_accuracy`` inside ``DepthAccuracy`` objects, with optional
    ``coda_accuracy`` for prelude/core readouts, so that the grouping
    machinery (``get_metric``, ``_build_groups``) works identically to
    ``ModelRunData``.

    Attributes:
        per_virtual_layer_metrics: One entry per virtual layer in execution order.
            Each entry has ``logitlens_accuracy``, ``logitlens_norm_accuracy``
            and ``probe_accuracy`` set.
        model_name: Model identifier recorded by the experiment.
        dataset_name: Dataset name (e.g. ``"arc_easy"``).
        n_test: Number of test examples evaluated.
        n_train: Number of training examples used for probe training.
    """

    per_virtual_layer_metrics: list[DepthAccuracy]
    model_name: str
    dataset_name: str
    n_test: int
    n_train: int

    @property
    def recurrent_layer_indices(self) -> list[int]:
        """Ordered list of layer indices that are recurrent (applied more than once)."""
        return _recurrent_layer_indices(self.per_virtual_layer_metrics)

    @overload
    def get_metric(self, field: str, group_by: None = ...) -> Tensor: ...

    @overload
    def get_metric(
        self, field: str, group_by: Literal["layer", "recurrence"]
    ) -> OrderedDict[str, Tensor]: ...

    def get_metric(
        self,
        field: str,
        group_by: Literal[None, "layer", "recurrence"] = None,
    ) -> Tensor | OrderedDict[str, Tensor]:
        """Extract *field* from every virtual layer, with optional grouping.

        The primary fields are ``"logitlens_accuracy"``,
        ``"logitlens_norm_accuracy"`` and ``"probe_accuracy"``, each returning a
        ``(1,)`` scalar tensor per virtual layer.  With ``group_by=None`` the
        result is shape ``(n_virtual_layers,)``; with ``group_by="recurrence"``
        the keys are ``"Prelude"``, ``"R0"`` … ``"R{n-1}"``, ``"Coda"`` (looped
        models) or an empty dict (feedforward).

        ``"coda_accuracy"`` is available when requested during evaluation;
        observed coda positions are NaN and should be omitted from coda-lens plots.
        ``"projected_logitlens_accuracy"`` is available when requested for Parcae.
        """
        return _get_metric_impl(
            self.per_virtual_layer_metrics,
            field,
            group_by,
            self.recurrent_layer_indices,
        )

    def save(self, path: str | Path) -> None:
        """Save to a ``.pt`` file via ``torch.save``."""
        torch.save(state_dict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> DepthEvaluationResult:
        """Load from a ``.pt`` file written by :meth:`save`."""
        raw = torch.load(path, weights_only=False)
        metrics = [DepthAccuracy(**m) for m in raw.pop("per_virtual_layer_metrics")]
        return cls(per_virtual_layer_metrics=metrics, **raw)
