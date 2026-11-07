from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch import Tensor

from looped_internals.analysis.behaviors import (
    FixedPoint,
    Orbit,
    Slider,
    _classify_series_without_last,
)
from looped_internals.analysis.comparisons import VectorComparisonMetric

if TYPE_CHECKING:
    from .runs import ModelRunData

from ._repr import _FastReprMixin
from ._serialization import state_dict


@dataclass(repr=False)
class ExampleResidualComparisonData(_FastReprMixin):
    """Paired per-layer residual comparison series for one example, ready to classify.

    Stores the minimal data needed to re-run
    :func:`~looped_internals.results.ModelRunData.classify_limiting_behavior_per_token`
    with arbitrary parameters — without re-running model inference.

    Attributes:
        example_idx: GSM8K (or other dataset) example index.
        token_strs: Decoded token string for each position in the sequence.
            Length ``T`` — self-contained, no tokenizer required at load time.
        question_start: First token position belonging to the question (i.e.
            ``system_token_count``).  Only tokens from this position onward
            appear in the DataFrame returned by :meth:`classify`.
        to_final_state_residual_cosine_sim_by_layer: Maps each recurrent layer
            to ``(R_l, T)`` cosine similarities with that layer's final state.
        to_final_state_residual_diff_norm_by_layer: The same series of pairs,
            measured with Euclidean difference norms.
    """

    example_idx: int
    token_strs: list[str]
    question_start: int
    to_final_state_residual_cosine_sim_by_layer: dict[int, Tensor]
    to_final_state_residual_diff_norm_by_layer: dict[int, Tensor]

    @classmethod
    def from_run(
        cls,
        run: ModelRunData,
        example_idx: int,
        token_strs: list[str],
        question_start: int = 0,
    ) -> ExampleResidualComparisonData:
        """Retain both token metrics for each recurrent layer's final reference."""
        if (
            run.to_final_state_residual_cosine_sim is None
            or run.to_final_state_residual_diff_norm is None
        ):
            raise ValueError(
                "Collect both residual comparison metrics with compute_differences=True "
                "before extracting trajectories"
            )
        if len(token_strs) != run.input_ids.shape[-1]:
            raise ValueError("token_strs must contain one string per input token")
        to_final_state_residual_cosine_sim_by_layer = {}
        to_final_state_residual_diff_norm_by_layer = {}
        for layer in run.recurrent_layer_indices:
            indices = [
                b.virtual_layer_index
                for b in run.per_virtual_layer_metrics
                if b.layer_index == layer
            ]
            to_final_state_residual_cosine_sim_by_layer[layer] = (
                run.to_final_state_residual_cosine_sim[indices, layer].cpu()
            )
            to_final_state_residual_diff_norm_by_layer[layer] = (
                run.to_final_state_residual_diff_norm[indices, layer].cpu()
            )
        return cls(
            example_idx=example_idx,
            token_strs=token_strs,
            question_start=question_start,
            to_final_state_residual_cosine_sim_by_layer=to_final_state_residual_cosine_sim_by_layer,
            to_final_state_residual_diff_norm_by_layer=to_final_state_residual_diff_norm_by_layer,
        )

    def classify(
        self,
        closeness_threshold: float = 1e-2,
        recurrence_lookback_distance: int = 64,
        fixed_point_fraction: float = 0.9,
        slider_gradient_threshold: float | None = None,
        metric: VectorComparisonMetric = "cosine_sim",
    ) -> pd.DataFrame:
        """Classify per-token limiting behavior and return a tidy DataFrame.

        Classifies every stored layer and returns one row per
        (layer, question token) combination.

        Args:
            closeness_threshold: Amplitude threshold for orbit detection and
                closeness criterion in units of the selected metric.
            recurrence_lookback_distance: Number of virtual layers before the final
                (trivial) virtual layer to examine.
            fixed_point_fraction: Fraction of lookback frames that must be
                "close" to classify as FixedPoint.
            slider_gradient_threshold: Minimum linear-fit gradient to classify
                as Slider.  ``None`` uses ``closeness_threshold / n``.
            metric: ``"cosine_sim"`` (default) or ``"diff_norm"``.

        Returns:
            DataFrame with one row per (layer, question token) combination
            (positions ``question_start`` to ``T-1``) and columns:
            ``example_idx``, ``layer_idx``, ``sequence_pos``, ``token_str``,
            ``behavior_type``, ``orbit_frequency``, ``orbit_amplitude``,
            ``slider_gradient``.
        """
        if metric == "cosine_sim":
            series_by_layer = self.to_final_state_residual_cosine_sim_by_layer
        elif metric == "diff_norm":
            series_by_layer = self.to_final_state_residual_diff_norm_by_layer
        else:
            raise ValueError("metric must be 'cosine_sim' or 'diff_norm'")
        rows = []
        for layer_idx, full_series in series_by_layer.items():
            # Apply lookback (+1 so the final virtual layer is always included for trimming)
            series = full_series[-(recurrence_lookback_distance + 1) :]
            series_without_last = series[:-1]  # (effective_lookback, T)

            behaviors = _classify_series_without_last(
                series_without_last,
                closeness_threshold=closeness_threshold,
                fixed_point_fraction=fixed_point_fraction,
                slider_gradient_threshold=slider_gradient_threshold,
                metric=metric,
            )

            for seq_pos in range(self.question_start, len(behaviors)):
                b = behaviors[seq_pos]
                if isinstance(b, Orbit):
                    btype, freq, amp, grad = (
                        "Orbit",
                        b.frequency,
                        b.amplitude,
                        float("nan"),
                    )
                elif isinstance(b, Slider):
                    btype, freq, amp, grad = (
                        "Slider",
                        float("nan"),
                        float("nan"),
                        b.gradient,
                    )
                elif isinstance(b, FixedPoint):
                    btype, freq, amp, grad = (
                        "FixedPoint",
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )
                else:
                    btype, freq, amp, grad = (
                        "Unknown",
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )
                rows.append(
                    {
                        "example_idx": self.example_idx,
                        "layer_idx": layer_idx,
                        "sequence_pos": seq_pos,
                        "token_str": self.token_strs[seq_pos],
                        "behavior_type": btype,
                        "orbit_frequency": freq,
                        "orbit_amplitude": amp,
                        "slider_gradient": grad,
                    }
                )
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        """Save to a ``.pt`` file via ``torch.save``."""
        torch.save(state_dict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> ExampleResidualComparisonData:
        """Load from a ``.pt`` file written by :meth:`save`."""
        raw = torch.load(path, weights_only=False)
        return cls(**raw)
