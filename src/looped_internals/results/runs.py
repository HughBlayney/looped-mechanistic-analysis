from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

import numpy as np
import torch
from torch import Tensor

from looped_internals.analysis.behaviors import (
    LimitingBehavior,
    _classify_series_without_last,
)
from looped_internals.analysis.comparisons import VectorComparisonMetric

from ._repr import _FastReprMixin
from ._serialization import state_dict
from .depth import _build_groups, _get_metric_impl, _recurrent_layer_indices


@dataclass(repr=False)
class PerVirtualLayerMetrics(_FastReprMixin):
    """Metrics for one application of a layer's weights.

    Attributes:
        layer_index: Zero-based identity of the weights across all layer groups.
            For a Raven model with 2 prelude + 4 core + 2 coda layers, the core
            layers are at indices 2–5 and the coda layers at 6–7.
        virtual_layer_index: Zero-based position in the recorded forward pass.
            Recurrent applications share a layer index and have distinct virtual
            layer indices. Non-recurrent layers have exactly one entry.
        sink_scores: ``(batch, heads)`` — mean attention to key position 0.
        mixing_scores: ``(batch, heads)`` — normalised row-entropy of attention.
        residual_norms: ``(batch,)`` — L2 norm of residual at position 0.
        residual_norms_last: ``(batch,)`` — L2 norm at the final token position.
        residual_norms_mean: ``(batch,)`` — mean of per-token L2 norms.
        colsum_concentrations: ``(batch, heads)`` — attention column concentration.
        raw_residual: ``(batch, T, D)`` — output residual stream for this
            virtual layer.  Populated when ``store_raw_values=True`` is passed to
            :func:`~looped_internals.capture.compute_model_run_data`.
        raw_attn_pattern: ``(batch, H, T, T)`` — attention pattern for this
            virtual layer.  ``None`` if the pattern could not be reconstructed or
            ``store_raw_values=False``.
        raw_attn_q: ``(batch, H, T, D_head)`` — query projection for this
            virtual layer.  ``None`` if Q/K/V could not be extracted or
            ``store_raw_values=False``.
        raw_attn_k: ``(batch, H_kv, T, D_head)`` — key projection.
        raw_attn_v: ``(batch, H_kv, T, D_head)`` — value projection.
        residual_entropy: ``(batch,)`` — effective-rank entropy of the residual
            stream.  Populated when ``compute_residual_entropy=True`` is passed
            to :func:`~looped_internals.capture.compute_model_run_data`.
    """

    layer_index: int
    virtual_layer_index: int
    sink_scores: Tensor | None
    mixing_scores: Tensor | None
    residual_norms: Tensor
    colsum_concentrations: Tensor | None
    raw_residual: Tensor | None = None
    raw_attn_pattern: Tensor | None = None
    raw_attn_q: Tensor | None = None
    raw_attn_k: Tensor | None = None
    raw_attn_v: Tensor | None = None
    residual_entropy: Tensor | None = None
    residual_norms_last: Tensor | None = None
    residual_norms_mean: Tensor | None = None

    @property
    def sink_rates(self) -> Tensor:
        """``(batch,)`` — fraction of heads whose sink score exceeds 0.3."""
        if self.sink_scores is None:
            raise ValueError("sink_scores is None; cannot compute sink_rates.")
        return (self.sink_scores.float() >= 0.3).float().mean(dim=-1)


@dataclass(repr=False)
class ModelRunData(_FastReprMixin):
    """All metrics collected during a single model forward pass (one batch).

    ``per_virtual_layer_metrics`` follows forward-pass order: entry ``i`` is
    virtual layer ``i`` (``virtual_layer_index == i``). ``S`` counts virtual
    layers; ``L`` is the layer axis length, including any unobserved layer slots.

    Attributes:
        input_ids: ``(batch, seq_len)`` — the input token IDs.
        per_virtual_layer_metrics: One entry per virtual layer, in execution order.
        pairwise_residual_cosine_sim: ``(S, S, T)`` — per-token cosine similarity
            for every pair of virtual layers, including self-comparisons.
        pairwise_residual_diff_norm: ``(S, S, T)`` — per-token Euclidean norm
            of differences for those same pairs, on unnormalized residuals.
        pairwise_attn_pattern_diff_norm: ``(S, S, H)`` — per-head Frobenius
            difference norm for every pair of attention patterns.
        to_final_state_residual_cosine_sim: ``(S, L, T)`` — residual cosine
            similarity to each layer's last observed state.
        to_final_state_residual_diff_norm: ``(S, L, T)`` — residual Euclidean
            difference norm to each layer's last observed state.
        to_final_state_attn_pattern_diff_norm: ``(S, L, H)`` — attention-pattern
            Frobenius difference norm to each layer's last observed pattern.
        successive_recurrence_residual_cosine_sim: List of L tensors, each
            ``(max(R_l-1, 0), T)`` — cosine similarity between consecutive
            applications of the same layer.
        successive_recurrence_residual_diff_norm: Same layout — Euclidean
            difference norm between those same residuals.
        successive_recurrence_attn_pattern_diff_norm: List of L tensors, each
            ``(max(R_l-1, 0), H)`` — Frobenius difference norm between patterns
            of consecutive applications of the same layer.

    Disabled metrics are ``None``. Requested attention comparisons are ``None``
    unless all virtual layers have patterns. Unobserved final-reference layer
    slots contain ``NaN``; recurrence lists have empty tensors for layers applied
    fewer than twice. Newly computed residual comparisons retain both metrics.
    """

    input_ids: Tensor
    per_virtual_layer_metrics: list[PerVirtualLayerMetrics]
    pairwise_residual_cosine_sim: Tensor | None = None
    pairwise_residual_diff_norm: Tensor | None = None
    pairwise_attn_pattern_diff_norm: Tensor | None = None
    to_final_state_residual_cosine_sim: Tensor | None = None
    to_final_state_residual_diff_norm: Tensor | None = None
    to_final_state_attn_pattern_diff_norm: Tensor | None = None
    successive_recurrence_residual_cosine_sim: list[Tensor] | None = None
    successive_recurrence_residual_diff_norm: list[Tensor] | None = None
    successive_recurrence_attn_pattern_diff_norm: list[Tensor] | None = None

    def save(self, path: str | Path) -> None:
        """Save to a ``.pt`` file via ``torch.save``."""
        torch.save(state_dict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> ModelRunData:
        """Load from a ``.pt`` file written by :meth:`save`."""
        raw = torch.load(path, weights_only=False)
        metrics = [
            PerVirtualLayerMetrics(**m) for m in raw.pop("per_virtual_layer_metrics")
        ]
        return cls(per_virtual_layer_metrics=metrics, **raw)

    @property
    def recurrent_layer_indices(self) -> list[int]:
        """Ordered list of unique layer indices that are recurrent (applied more than once).

        Ordering reflects first appearance in the execution sequence.
        """
        return _recurrent_layer_indices(self.per_virtual_layer_metrics)

    @property
    def prelude_layer_indices(self) -> list[int]:
        """Ordered list of layer indices in the prelude (layers before the first recurrent layer)."""
        prelude_layer_indices = []
        recurrent_layer_indices = set(self.recurrent_layer_indices)
        for pbm in self.per_virtual_layer_metrics:
            if pbm.layer_index in recurrent_layer_indices:
                break
            else:
                prelude_layer_indices.append(pbm.layer_index)
        return prelude_layer_indices

    @property
    def coda_layer_indices(self) -> list[int]:
        """Ordered list of layer indices in the coda (layers after the last recurrent layer)."""
        coda_layer_indices = []
        recurrent_layer_indices = set(self.recurrent_layer_indices)
        for pbm in reversed(self.per_virtual_layer_metrics):
            if pbm.layer_index in recurrent_layer_indices:
                break
            else:
                coda_layer_indices.append(pbm.layer_index)
        return list(reversed(coda_layer_indices))

    @property
    def num_loops(self) -> int:
        """Number of loops (recurrent iterations) in this model run."""
        all_layer_indicies = [pbm.layer_index for pbm in self.per_virtual_layer_metrics]
        counts = Counter(all_layer_indicies)
        max_count = max(counts.values())
        assert all(count == 1 or count == max_count for count in counts.values()), (
            "Unexpected virtual layer counts per layer: " + str(counts)
        )
        return max_count

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

        Validates that every :class:`PerVirtualLayerMetrics` has *field* set (not
        ``None``) and that the batch size is exactly 1, then squeezes the
        batch dimension.

        Args:
            field: :class:`PerVirtualLayerMetrics` attribute name, e.g.
                ``"sink_scores"``.
            group_by: How to group the virtual layers.

                * ``None`` — returns a ``Tensor`` of shape ``(S, *dims)``
                  containing all virtual layers in execution order.
                * ``"layer"`` — returns an ``OrderedDict[str, Tensor]`` keyed
                  ``"L0"``, ``"L1"``, … (sorted by layer index).  Each tensor
                  has shape ``(n_virtual_layers_for_layer, *dims)``.
                * ``"recurrence"`` — returns an ``OrderedDict[str, Tensor]``
                  keyed ``"Prelude"``, ``"R0"``, ``"R1"``, …, ``"Coda"``
                  (prelude / coda keys only present when the model has them).
                  Each tensor has shape ``(n_virtual_layers_in_group, *dims)``.

        Returns:
            A single ``Tensor`` when ``group_by=None``, otherwise an
            ``OrderedDict`` mapping group name to stacked ``Tensor``.
        """
        return _get_metric_impl(
            self.per_virtual_layer_metrics,
            field,
            group_by,
            self.recurrent_layer_indices,
        )

    @overload
    def get_pca_residual_trajectories(
        self,
        num_dimensions: int = ...,
        separate_across_tokens: bool = ...,
        group_by: None = ...,
    ) -> np.ndarray: ...

    @overload
    def get_pca_residual_trajectories(
        self,
        num_dimensions: int = ...,
        separate_across_tokens: bool = ...,
        group_by: Literal["layer", "recurrence"] = ...,
    ) -> OrderedDict[str, np.ndarray]: ...

    def get_pca_residual_trajectories(
        self,
        num_dimensions: int = 2,
        separate_across_tokens: bool = False,
        group_by: Literal[None, "layer", "recurrence"] = None,
    ) -> np.ndarray | OrderedDict[str, np.ndarray]:
        """PCA-reduce the residual-stream trajectories stored in this run.

        Requires every :class:`PerVirtualLayerMetrics` to have ``raw_residual`` set
        (i.e. the data must have been collected with ``store_raw_values=True``).

        PCA is always fitted on the **full** set of virtual layers so that axes are
        consistent across groups when ``group_by`` is used.

        Args:
            num_dimensions: Number of PCA output components.  Defaults to 2.
            separate_across_tokens: If ``False`` (default), a single PCA is
                fitted jointly on all ``(S × T)`` residual vectors, so token
                positions share the same embedding space.  If ``True``, a
                separate PCA is fitted independently for each token position
                using its ``S`` residual vectors — axes across token positions
                are *not* comparable to one another.
            group_by: How to group the output.

                * ``None`` — returns a single ``ndarray`` of shape
                  ``(S, T, num_dimensions)``.
                * ``"layer"`` — returns an ``OrderedDict[str, ndarray]`` keyed
                  ``"L0"``, ``"L1"``, … Each value has shape
                  ``(n_virtual_layers_for_layer, T, num_dimensions)``.
                * ``"recurrence"`` — returns an ``OrderedDict[str, ndarray]``
                  keyed ``"Prelude"``, ``"R0"``, ``"R1"``, …, ``"Coda"``
                  (prelude / coda keys only present when the model has them).
                  Each value has shape ``(n_virtual_layers_in_group, T,
                  num_dimensions)``.

        Returns:
            A single ``ndarray`` when ``group_by=None``, otherwise an
            ``OrderedDict`` mapping group name to ``ndarray``.

        Raises:
            ValueError: If any ``raw_residual`` is ``None`` or batch size ≠ 1
                (propagated from :meth:`get_metric`).
        """
        from sklearn.decomposition import PCA  # local import — optional dep

        # (S, T, D) — batch dim already squeezed by get_metric
        residuals_np = self.get_metric("raw_residual").float().numpy()
        S, T, D = residuals_np.shape

        if separate_across_tokens:
            result = np.empty((S, T, num_dimensions), dtype=np.float32)
            for t in range(T):
                pca = PCA(n_components=num_dimensions)
                result[:, t, :] = pca.fit_transform(residuals_np[:, t, :])
        else:
            pca = PCA(n_components=num_dimensions)
            result = pca.fit_transform(residuals_np.reshape(S * T, D)).reshape(
                S, T, num_dimensions
            )

        if group_by is None:
            return result

        groups = _build_groups(
            self.per_virtual_layer_metrics, group_by, self.recurrent_layer_indices
        )
        return OrderedDict({key: result[idxs] for key, idxs in groups.items()})

    def classify_limiting_behavior_per_token(
        self,
        closeness_threshold: float = 1e-2,
        recurrence_lookback_distance: int = 64,
        metric: VectorComparisonMetric = "cosine_sim",
        fixed_point_fraction: float = 0.9,
        slider_gradient_threshold: float | None = None,
    ) -> dict[int, list[LimitingBehavior]]:
        """Classify the limiting behavior of each token at each recurrent layer.

        For each recurrent layer, examines the last *recurrence_lookback_distance*
        virtual layers of that layer (excluding the trivially-zero/one final virtual
        layer) and
        classifies each token's residual trajectory as (in order):

        * :class:`FixedPoint` — at least ``fixed_point_fraction`` of the lookback
          window is within ``closeness_threshold`` of the final state.
        * :class:`Orbit` — the detrended, Hann-windowed FFT of the lookback series
          has a dominant non-DC component with Hann-corrected amplitude >=
          ``closeness_threshold / 2`` (so that peak-to-peak equals
          ``closeness_threshold``), and the dominant period is at most ``n/2``
          (at least 2 full cycles visible in the window).
        * :class:`Slider` — the linear-fit gradient of the similarity series
          exceeds ``slider_gradient_threshold``.
        * :class:`Unknown` — no pattern matched.

        Args:
            closeness_threshold: Difference norms ``<= closeness_threshold``
                or cosine similarities ``>= 1 - closeness_threshold`` are close.
            recurrence_lookback_distance: Number of recurrences before the final
                virtual layer to examine. If fewer are available, all are used.
            metric: ``"cosine_sim"`` (default) or ``"diff_norm"``, selecting the
                corresponding ``to_final_state_residual_*`` tensor. Thresholds
                are expressed in units of the selected metric.
            fixed_point_fraction: Fraction of the lookback window that must be
                "close" to classify as FixedPoint (0 < value <= 1).  Default 0.9.
            slider_gradient_threshold: Minimum linear-fit gradient to classify
                as Slider.  ``None`` uses ``closeness_threshold / n``.

        Returns:
            Dict mapping each recurrent ``layer_index`` to a list of
            :class:`LimitingBehavior` instances, one per input token.

        Raises:
            ValueError: If the required difference tensor is ``None``.
        """
        if metric == "cosine_sim":
            tensor = self.to_final_state_residual_cosine_sim
        elif metric == "diff_norm":
            tensor = self.to_final_state_residual_diff_norm
        else:
            raise ValueError("metric must be 'cosine_sim' or 'diff_norm'")
        if tensor is None:
            raise ValueError(
                f"to_final_state_residual_{metric} is None; recompute with "
                "compute_differences=True."
            )

        # tensor shape: (S, L, T)
        result: dict[int, list[LimitingBehavior]] = {}
        for layer_idx in self.recurrent_layer_indices:
            virtual_layer_indices = [
                pbm.virtual_layer_index
                for pbm in self.per_virtual_layer_metrics
                if pbm.layer_index == layer_idx
            ]
            # Extract this layer's virtual layer series: (n_virtual_layers, T)
            series = tensor[virtual_layer_indices, layer_idx, :]

            # Apply lookback (+1 so the final virtual layer is always included for trimming)
            series = series[-(recurrence_lookback_distance + 1) :]

            # The final virtual layer is trivially 0 (norm) or 1 (sim); exclude from analysis
            series_without_last = series[:-1]  # (effective_lookback, T)

            result[layer_idx] = _classify_series_without_last(
                series_without_last,
                closeness_threshold=closeness_threshold,
                fixed_point_fraction=fixed_point_fraction,
                slider_gradient_threshold=slider_gradient_threshold,
                metric=metric,
            )

        return result
