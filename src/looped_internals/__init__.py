"""Looped transformer models, activation capture, and compact analysis results.

The public workflow is load_model → run_with_cache → compute_model_run_data.
Model loading initializes the TransformerLens integration on demand.
"""

from .capture import compute_model_run_data, metrics_names_filter
from .results import ModelRunData, PerVirtualLayerMetrics
from .structure import LayerGroup, ModelLayerStructure, VirtualLayer


def load_model(*args, **kwargs):
    """Load a model as ``(TransformerBridge, ModelLayerStructure)``."""
    from .models import load_model as load

    return load(*args, **kwargs)


__all__ = [
    "load_model",
    "compute_model_run_data",
    "metrics_names_filter",
    "ModelRunData",
    "PerVirtualLayerMetrics",
    "LayerGroup",
    "ModelLayerStructure",
    "VirtualLayer",
]
