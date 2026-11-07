"""Model loading and configuration."""

from .loading import MODEL_SPECS, LoadedModel, ModelSpec, load_model
from .minimal import MinimalRecurrentConfig, MinimalRecurrentLM
from .parcae import ParcaeCheckpoint

__all__ = [
    "MODEL_SPECS",
    "LoadedModel",
    "MinimalRecurrentConfig",
    "MinimalRecurrentLM",
    "ModelSpec",
    "ParcaeCheckpoint",
    "load_model",
]

from looped_internals._initialization import initialize

initialize()
