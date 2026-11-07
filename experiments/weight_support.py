"""Model loading and run settings for weight-only experiments."""

import gc
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import torch

from experiments.support import MODEL_CHECKPOINTS
from looped_internals.models import MODEL_SPECS, load_model


@dataclass(frozen=True)
class WeightExperimentConfig:
    mode: str
    models: tuple[str, ...]
    device: str
    dtype: str
    output_dir: Path
    experiment: str = "neuron_detection"


def configure_weight_experiment(
    *, models, mode="quick", device="auto", dtype="auto", output_dir=None
):
    """Resolve weight-analysis settings without dataset or rollout parameters."""
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be 'quick' or 'full'")
    if not models or len(set(models)) != len(models):
        raise ValueError("Select at least one model, without duplicates")
    if unknown := {
        name for name in models if MODEL_CHECKPOINTS.get(name, name) not in MODEL_SPECS
    }:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        dtype = "bfloat16" if device.startswith("cuda") else "float32"
    if dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError("dtype must be float32, bfloat16, float16, or auto")
    root = Path(__file__).resolve().parents[1]
    return WeightExperimentConfig(
        mode=mode,
        models=tuple(models),
        device=device,
        dtype=dtype,
        output_dir=root / (output_dir or "experiments/results"),
    )


def collect_weight_summary(
    name, config, summarize, *, cache=None, analysis_version=1, analysis_options=None
):
    from experiments.cache import ExperimentCache

    cache = cache if cache is not None else ExperimentCache(mode="off")
    key = cache.key(
        config,
        name,
        analysis_version=analysis_version,
        analysis_options=analysis_options,
    )
    return cache.run(
        "neuron_detection",
        key,
        lambda: _collect_weights(name, config, summarize, cache),
    )


def _collect_weights(name, config, summarize, cache):
    """Load one checkpoint, reduce its weights, and release it without inference."""
    checkpoint = MODEL_CHECKPOINTS.get(name, name)
    bridge, structure = load_model(
        cache.model(name)[0] if cache.enabled else checkpoint,
        device=config.device,
        dtype=getattr(torch, config.dtype),
    )
    try:
        with torch.inference_mode():
            summary = summarize(bridge, structure)
        source = MODEL_SPECS[checkpoint].source
        info = {
            "checkpoint": asdict(source) if is_dataclass(source) else source,
            "revision": getattr(
                getattr(bridge.original_model, "config", None), "_commit_hash", None
            ),
            "device": config.device,
            "dtype": config.dtype,
            "layer_groups": [asdict(group) for group in structure.groups],
            "d_model": bridge.W_U.shape[0],
            "d_vocab": bridge.W_U.shape[1],
        }
        return summary, info
    finally:
        del bridge
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
