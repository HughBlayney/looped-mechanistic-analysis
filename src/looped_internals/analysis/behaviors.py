from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .comparisons import VectorComparisonMetric


@dataclass
class LimitingBehavior:
    """Base class for per-token limiting-behavior classifications."""


@dataclass
class FixedPoint(LimitingBehavior):
    """Residual converges to the final state well before the last recurrence."""


@dataclass
class Orbit(LimitingBehavior):
    """Residual cycles periodically through the final state.

    Attributes:
        frequency: Dominant frequency in cycles per recurrence loop,
            estimated via FFT of the lookback window.
        amplitude: Hann-corrected peak amplitude in the selected metric
            units, computed as ``4 * |FFT[dominant_idx]| / n``.
    """

    frequency: float
    amplitude: float


@dataclass
class Slider(LimitingBehavior):
    """Residual is converging with a significant positive gradient in cosine similarity.

    Attributes:
        gradient: Slope of the linear fit to the lookback similarity series
            (similarity units per recurrence virtual layer).
    """

    gradient: float


@dataclass
class Unknown(LimitingBehavior):
    """No recognized pattern detected in the lookback window."""


def _classify_series_without_last(
    series_without_last: Tensor,
    closeness_threshold: float,
    fixed_point_fraction: float,
    slider_gradient_threshold: float | None = None,
    metric: VectorComparisonMetric = "cosine_sim",
) -> list[LimitingBehavior]:
    """Classify each token's limiting behavior from a pre-sliced lookback series.

    Classification is applied in order; the first matching class wins:

    1. :class:`FixedPoint` — ``fixed_point_fraction`` of frames are "close".
    2. :class:`Orbit` — dominant non-DC FFT amplitude >= ``closeness_threshold / 2``
       (peak-to-peak equals ``closeness_threshold``) **and** the dominant period is at
       most ``n/2`` (i.e. at least 2 full cycles are visible; ``freq >= 2/n``).
    3. :class:`Slider` — linear-fit gradient > ``slider_gradient_threshold``.
    4. :class:`Unknown` — no pattern matched.

    Args:
        series_without_last: ``(effective_lookback, T)`` similarity or norm
            series with the trivial final virtual layer already removed.
        closeness_threshold: Amplitude threshold for orbit detection and
            closeness criterion for FixedPoint.
        fixed_point_fraction: Fraction of lookback frames that must be "close"
            to classify as FixedPoint.
        slider_gradient_threshold: Minimum linear-fit gradient (similarity per
            virtual layer) to classify as Slider.  ``None`` (default) uses
            ``closeness_threshold / n``, corresponding to a change of one
            closeness threshold unit over the full lookback window.
        metric: ``"cosine_sim"`` (default) treats values
            ``>= 1 - closeness_threshold`` as close; ``"diff_norm"`` treats
            values ``<= closeness_threshold`` as close.

    Returns:
        One :class:`LimitingBehavior` per token position.
    """
    n, T_len = series_without_last.shape
    if n == 0:
        raise ValueError("At least one non-final observation is required")

    if slider_gradient_threshold is None:
        slider_gradient_threshold = closeness_threshold / n

    if metric == "cosine_sim":
        close = series_without_last >= (1.0 - closeness_threshold)
    elif metric == "diff_norm":
        close = series_without_last <= closeness_threshold
    else:
        raise ValueError("metric must be 'cosine_sim' or 'diff_norm'")

    n_close = close.sum(dim=0)  # (T,)
    fp_threshold = round(fixed_point_fraction * n)

    behaviors: list[LimitingBehavior] = []
    for t in range(T_len):
        signal = series_without_last[:, t].float()

        # --- Detrend: subtract linear least-squares trend ---
        t_idx = torch.arange(n, dtype=torch.float32)
        A = torch.stack([t_idx, torch.ones(n)], dim=1)  # (n, 2)
        coeffs = torch.linalg.lstsq(A, signal.unsqueeze(1)).solution
        gradient = float(coeffs[0].item())
        trend = (A @ coeffs).squeeze(1)
        detrended = signal - trend

        # --- Hann window to reduce spectral leakage ---
        windowed = detrended * torch.hann_window(n, periodic=False)

        # --- Step A: detect orbit via FFT ---
        magnitudes = torch.abs(torch.fft.rfft(windowed))[1:]  # skip DC
        if magnitudes.numel() > 0:
            dominant_idx = int(magnitudes.argmax().item())
            dominant_amp = magnitudes[dominant_idx].item()
            amplitude = 4.0 * dominant_amp / n  # Hann-corrected
            freq = (dominant_idx + 1) / n  # cycles per loop
        else:
            amplitude = 0.0
            freq = 0.0

        nc = int(n_close[t].item())
        if nc >= fp_threshold:
            behavior = FixedPoint()
        elif amplitude >= closeness_threshold / 2.0 and freq >= 2.0 / n:
            # Require at least 2 full cycles in the window (period ≤ n/2, i.e. freq ≥ 2/n).
            # A single-cycle "orbit" is indistinguishable from a slow ramp after detrending.
            # Amplitude threshold is half the closeness_threshold so that the total
            # peak-to-peak excursion (2 × amplitude) equals closeness_threshold.
            behavior: LimitingBehavior = Orbit(frequency=freq, amplitude=amplitude)
        else:
            # --- Potential Slider ---
            # For cosine similarity: converging = positive gradient.
            # For difference norm: converging = negative gradient; flip sign.
            convergence_gradient = gradient if metric == "cosine_sim" else -gradient
            if convergence_gradient > slider_gradient_threshold:
                behavior = Slider(gradient=gradient)
            # --- Step D: Unknown ---
            else:
                behavior = Unknown()

        behaviors.append(behavior)

    return behaviors
