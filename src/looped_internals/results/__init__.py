"""Typed, serializable analysis results."""

from looped_internals.analysis.aggregation import aggregate_values
from looped_internals.analysis.behaviors import (
    FixedPoint,
    LimitingBehavior,
    Orbit,
    Slider,
    Unknown,
)

from .depth import DepthValues
from .multiple_choice import DepthAccuracy, DepthEvaluationResult
from .runs import ModelRunData, PerVirtualLayerMetrics
from .trajectories import ExampleResidualComparisonData

__all__ = [
    "DepthAccuracy",
    "DepthEvaluationResult",
    "DepthValues",
    "ExampleResidualComparisonData",
    "FixedPoint",
    "LimitingBehavior",
    "ModelRunData",
    "Orbit",
    "PerVirtualLayerMetrics",
    "Slider",
    "Unknown",
    "aggregate_values",
]
