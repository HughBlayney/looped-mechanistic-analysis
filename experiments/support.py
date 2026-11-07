"""Shared setup, data collection, and exports for the experiment notebooks.

The notebooks own the scientific reductions and plots. This module owns model
loading, task selection, seeds, and output paths. No work runs at import.
"""

from __future__ import annotations

import gc
import hashlib
import json
import random
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from looped_internals.evaluation.lm_eval import get_task_manager
from looped_internals.models.tokenization import BOSPolicy, tokenizer_metadata
from looped_internals.results import ModelRunData

# These are the same custom likelihood tasks used by create_paper_data*.py.
# New datasets need a packaged, zero-shot, single-request likelihood task.
DATASETS = {"gsm8k": "gsm8k_likelihood", "hellaswag": "hellaswag_likelihood"}
PAPER_PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots" / "mechanistic_analysis"
# LaTeX dimensions from the original paper notebook, in TeX points (72.27/inch).
ICML_SINGLE_COLUMN_WIDTH = 234.8775
ICML_TWO_COLUMN_WIDTH = 487.8225


# Experiment selections and their display labels.
MODELS = {
    "ouro_1_4b": "Ouro 1.4B",
    "ouro_2_6b": "Ouro 2.6B",
    "retrofitted_llama": "Retrofitted Llama",
    "retrofitted_olmo": "Retrofitted OLMo",
    "retrofitted_tinyllama": "Retrofitted TinyLlama",
    "raven": "Huginn-0125",
    "parcae_140m": "Parcae 140M",
    "parcae_370m": "Parcae 370M",
    "parcae_770m": "Parcae 770M",
    "parcae_1_3b": "Parcae 1.3B",
}
QUICK_MODELS = ("ouro_1_4b", "retrofitted_tinyllama")
# Parcae controls share their recurrent model's architecture, data, and token
# budget, but apply the core block exactly once throughout training.
BASELINES = {
    "llama_1b": "Llama 3.2 1B",
    "olmo2": "OLMo2",
    "tinyllama": "TinyLlama",
    "parcae_140m_rec1": "140M feedforward",
    "parcae_370m_rec1": "Parcae 370M feedforward",
    "parcae_770m_rec1": "Parcae 770M feedforward",
}
BASELINE_FOR = {
    "ouro_1_4b": "llama_1b",
    "ouro_2_6b": "llama_1b",
    "retrofitted_llama": "llama_1b",
    "retrofitted_olmo": "olmo2",
    "retrofitted_tinyllama": "tinyllama",
    "raven": "llama_1b",
    "parcae_140m": "parcae_140m_rec1",
    "parcae_370m": "parcae_370m_rec1",
    "parcae_770m": "parcae_770m_rec1",
}
# Injection-stability sweep: one 140M model per (sampler, spectral radius) arm.
# These are selectable by name but excluded from MODELS, whose members are the
# default selection for every full-mode run.
STABILITY = {
    f"stab_{schedule}_{rho.replace('.', '_')}": f"Parcae {schedule} rho {rho}"
    for schedule in ("fixed", "poisson")
    for rho in ("0.25", "0.5", "0.75", "1.0", "1.25", "learned")
}
MODEL_LABELS = MODELS | BASELINES | STABILITY

# Preserve existing experiment/result names for the checkpoints trained at 32 steps.
MODEL_CHECKPOINTS = {
    "retrofitted_llama": "retrofitted_llama_32",
    "retrofitted_olmo": "retrofitted_olmo_32",
    "retrofitted_tinyllama": "retrofitted_tinyllama_32",
}

COLORMAPS = {
    "residual_streams": "magma",
    "attention_patterns": "viridis",
    "layers": "winter",
    "recurrences": "summer",
}


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    task: str
    mode: str
    models: tuple[str, ...]
    limit: int | None
    num_steps: int
    seed: int
    device: str
    dtype: str
    output_dir: Path
    experiment: str = "convergence"
    include_baselines: bool = False
    bos_policy: BOSPolicy = "native"
    compute_residual_entropy: bool = False


def configure_experiment(
    *,
    dataset: str = "gsm8k",
    mode: str = "quick",
    models: str | list[str] | None = None,
    limit: int | None = None,
    num_steps: int | None = None,
    seed: int = 42,
    device: str = "auto",
    dtype: str = "auto",
    output_dir: str | None = None,
    experiment: str = "convergence",
    include_baselines: bool = False,
    bos_policy: BOSPolicy = "native",
    compute_residual_entropy: bool | None = None,
) -> ExperimentConfig:
    """Resolve experiment settings after Papermill injects parameter overrides.

    Parameters
    ----------
    dataset : {'gsm8k', 'hellaswag'}, optional
        Dataset whose packaged likelihood task is evaluated.
    mode : {'quick', 'full'}, optional
        Preset controlling the default models, sample limit, and recurrence count.
    models : str or list of str, optional
        Registered recurrent model names, as a comma-separated string or list.
        Defaults to ``QUICK_MODELS`` in quick mode and all ``MODELS`` in full mode.
        Stages of inference, BOS comparison, entropy lens, multiple choice, and the
        smoke test accept explicit feedforward baseline names.
    limit : int, optional
        Number of examples to evaluate. ``None`` selects two examples in quick
        mode and 256 seeded examples in full mode; zero always selects the whole split.
    num_steps : int, optional
        Recurrence count, at least two. Defaults to 128 for full convergence runs
        and eight otherwise. A baseline-only selection uses one forward pass.
    seed : int, optional
        Random seed in ``[0, 2**32)``, defaulting to 42.
    device : str, optional
        Model device. ``'auto'`` selects CUDA when available and CPU otherwise.
    dtype : {'auto', 'float32', 'bfloat16', 'float16'}, optional
        Model precision. ``'auto'`` selects bfloat16 on CUDA and float32 otherwise.
    output_dir : str, optional
        Output root, defaulting to ``experiments/results``. Relative paths are
        resolved from the nearest ancestor containing ``pyproject.toml``.
    experiment : str, optional
        Analysis to configure, defaulting to convergence.
    include_baselines : bool, optional
        Whether to collect matched feedforward baselines for stages of inference.
        Defaults to False.
    bos_policy : {'native', 'bos', 'no_bos'}, optional
        Native checkpoint tokenization by default. The BOS appendix explicitly
        selects the two interventions; other notebooks retain native defaults.
    compute_residual_entropy : bool, optional
        Whether to compute the residual spectrum at each virtual layer. Defaults
        to True for stages of inference and BOS comparison,
        False otherwise.

    Returns
    -------
    ExperimentConfig
        Validated settings with defaults and output paths resolved.

    Raises
    ------
    ValueError
        If a selection or numeric setting is invalid, or baselines are requested
        for convergence.
    RuntimeError
        If no ancestor of the working directory contains ``pyproject.toml``.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(DATASETS)}")
    if experiment not in {
        "convergence",
        "stages_of_inference",
        "isolating_stability",
        "cyclic_similarity",
        "trajectories",
        "initialisation",
        "multiple_choice",
        "limiting_behaviour",
        "bos_comparison",
        "entropy_lens",
        "smoke_test",
    }:
        raise ValueError(f"Unknown experiment {experiment!r}")
    if bos_policy not in {"native", "bos", "no_bos"}:
        raise ValueError("bos_policy must be native, bos, or no_bos")
    if include_baselines and experiment not in {
        "stages_of_inference",
        "multiple_choice",
        "bos_comparison",
    }:
        raise ValueError("Feedforward baselines are only used for stages of inference")
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be 'quick' or 'full'")
    if isinstance(models, str):
        models = [name.strip() for name in models.split(",")]
    selected = (
        tuple(models)
        if models is not None
        else (QUICK_MODELS if mode == "quick" else tuple(MODELS))
    )
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select at least one model, without duplicates")
    available = (
        MODEL_LABELS
        if experiment
        in {
            "stages_of_inference",
            "isolating_stability",
            "bos_comparison",
            "multiple_choice",
            "entropy_lens",
            "smoke_test",
        }
        else MODELS
    )
    if unknown := set(selected) - available.keys():
        raise ValueError(
            f"Unknown models {sorted(unknown)}; choose from {list(available)}"
        )
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a nonnegative integer or None")
    resolved_limit = (2 if mode == "quick" else 256) if limit is None else limit
    if resolved_limit == 0:
        resolved_limit = None
    default_steps = 128 if mode == "full" and experiment == "convergence" else 8
    baseline_only = all(name in BASELINES for name in selected)
    if baseline_only:
        default_steps = 1
    steps = default_steps if num_steps is None else num_steps
    if baseline_only and (type(steps) is not int or steps != 1):
        raise ValueError("num_steps must be 1 for a feedforward-only selection")
    if not baseline_only and (type(steps) is not int or steps < 2):
        raise ValueError("num_steps must be an integer >= 2 for recurrent analysis")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        dtype = "bfloat16" if device.startswith("cuda") else "float32"
    if dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError("dtype must be float32, bfloat16, float16, or auto")

    root = next(
        (
            p
            for p in (Path.cwd(), *Path.cwd().parents)
            if (p / "pyproject.toml").is_file()
        ),
        None,
    )
    if root is None:
        raise RuntimeError("Open the notebook from this repository or experiments/")
    return ExperimentConfig(
        dataset=dataset,
        task=DATASETS[dataset],
        mode=mode,
        models=selected,
        limit=resolved_limit,
        num_steps=steps,
        seed=seed,
        device=device,
        dtype=dtype,
        output_dir=root / (output_dir or "experiments/results"),
        experiment=experiment,
        include_baselines=include_baselines,
        bos_policy=bos_policy,
        compute_residual_entropy=(
            experiment in {"stages_of_inference", "bos_comparison"}
            if compute_residual_entropy is None
            else compute_residual_entropy
        ),
    )


def sample_indices(size: int, limit: int | None, seed: int) -> list[int]:
    """Select examples using the seeded shuffle from the legacy scripts.

    Parameters
    ----------
    size : int
        Number of examples in the evaluation split; must be positive.
    limit : int or None
        Number of shuffled indices to keep, or None for the whole split.
        Experiment configuration supplies a positive integer or None.
    seed : int
        Seed for a local random generator; global random state is unchanged.

    Returns
    -------
    list of int
        Selected example indices in shuffled order.

    Raises
    ------
    ValueError
        If the split is empty or the requested limit exceeds its size.
    """
    if size < 1:
        raise ValueError("The evaluation split is empty")
    if limit is not None and limit > size:
        raise ValueError(f"Requested {limit} examples, but the split contains {size}")
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return indices[:limit]


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch, including available CUDA devices.

    Parameters
    ----------
    seed : int
        Seed in ``[0, 2**32)`` to apply to the global random generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model(name: str, config: ExperimentConfig, *, cache=None):
    """Load a registered model with the metrics required by the experiment.

    Parameters
    ----------
    name : str
        Model key in ``MODELS`` or ``BASELINES``.
    config : ExperimentConfig
        Device, precision, recurrence count, and analysis settings. Recurrence
        settings apply only to recurrent models.

    Returns
    -------
    BridgeLM
        Evaluation wrapper using one example per batch and the requested metrics.

    Raises
    ------
    KeyError
        If the model name is absent from both registries.
    """
    from looped_internals import models
    from looped_internals.evaluation.lm_eval import BridgeLM

    if name not in MODEL_LABELS:
        raise KeyError(name)
    bridge, structure = models.load_model(
        (
            cache.model(name)[0]
            if cache is not None and cache.enabled
            else MODEL_CHECKPOINTS.get(name, name)
        ),
        num_steps=config.num_steps if name in MODELS else None,
        device=config.device,
        dtype=getattr(torch, config.dtype),
    )
    return BridgeLM(
        bridge,
        structure,
        batch_size=1,
        compute_pairwise=config.experiment in {"cyclic_similarity", "bos_comparison"},
        compute_differences=config.experiment == "convergence",
        compute_residual_entropy=config.compute_residual_entropy,
        bos_policy=config.bos_policy,
    )


def _pack_profiles(run):
    """Pack existing small layer measurements, without retaining activations."""
    fields = (
        "sink_scores",
        "mixing_scores",
        "colsum_concentrations",
        "residual_norms",
        "residual_norms_last",
        "residual_norms_mean",
    )
    blocks = run.per_virtual_layer_metrics
    return {
        "input_ids": run.input_ids,
        "layers": [(b.layer_index, b.virtual_layer_index) for b in blocks],
        "values": {
            field: torch.stack([getattr(b, field) for b in blocks]) for field in fields
        },
    }


def _unpack_profiles(record):
    from looped_internals.results import PerVirtualLayerMetrics

    return ModelRunData(
        input_ids=record["input_ids"],
        per_virtual_layer_metrics=[
            PerVirtualLayerMetrics(
                layer_index=layer,
                virtual_layer_index=virtual,
                **{field: values[i] for field, values in record["values"].items()},
            )
            for i, (layer, virtual) in enumerate(record["layers"])
        ],
    )


STAGE_FIELDS = (
    "sink_rates",
    "mixing_scores",
    "colsum_concentrations",
    "residual_entropy",
    "residual_norms",
    "residual_norms_last",
    "residual_norms_mean",
)


def summarize_stages(run):
    """Average heads within one example, retaining every virtual layer.

    Stages of inference and mixing correlation share this cached reduction.
    """
    curves = {}
    for field in STAGE_FIELDS:
        if (
            field == "residual_entropy"
            and run.per_virtual_layer_metrics[0].residual_entropy is None
        ):
            continue
        values = run.get_metric(field).float()
        if values.ndim > 1:
            values = values.mean(dim=-1)
        curves[field] = values.cpu().tolist()
    return {
        "layers": [block.layer_index for block in run.per_virtual_layer_metrics],
        "virtual_layers": [
            block.virtual_layer_index for block in run.per_virtual_layer_metrics
        ],
        "recurrent_layers": run.recurrent_layer_indices,
        "curves": curves,
    }


def collect_metrics(
    config: ExperimentConfig,
    summarize: Callable[[ModelRunData], dict] | None,
    *,
    fwd_hooks: list[tuple[str, Callable]] | None = None,
    cache=None,
    analysis_version=1,
    intervention=None,
    cache_reducer: Callable | None = None,
    names_filter_factory: Callable | None = None,
    save_profiles: bool = False,
) -> tuple[dict[str, list[dict]], dict]:
    """Stream a generic or cache-specialized reduction and resume prefixes.

    Inputs follow the packaged zero-shot, single-request likelihood tasks.
    RNG state is restored after loading the model, preserving stochastic states.
    Forward hooks need an explicit intervention description when caching.
    A ``cache_reducer`` replaces generic ``ModelRunData`` construction and must
    return one ``{"input_ids": tensor, "summary": dict}`` record per example.
    Its optional name-filter factory receives the loaded layer structure.
    ``save_profiles`` additionally saves complete, entropy-free layer profiles
    for stages from the same inference. It is useful when both are requested.
    Cache identities describe the scientific outputs, not the reduction algorithm.
    """
    from dataclasses import replace
    from importlib.metadata import version

    from tqdm.auto import tqdm

    from experiments.cache import ExperimentCache
    from looped_internals._random import restore_rng
    from looped_internals.capture import compute_model_run_data, metrics_names_filter

    cache = cache if cache is not None else ExperimentCache(mode="off")
    if cache.enabled and fwd_hooks and intervention is None:
        raise ValueError("Cached forward hooks require an intervention description")
    if (summarize is None) == (cache_reducer is None):
        raise ValueError("Provide exactly one of summarize or cache_reducer")
    if save_profiles and (config.experiment != "convergence" or fwd_hooks):
        raise ValueError("Shared profiles require unmodified convergence inputs")
    task = get_task_manager().load([config.task])["tasks"][config.task]
    split_size = len(task.eval_docs)
    indices = sorted(sample_indices(split_size, config.limit, config.seed))
    texts = [
        (task.doc_to_text(task.eval_docs[i]), task.doc_to_target(task.eval_docs[i]))
        for i in indices
    ]
    selected_models = list(config.models)
    if config.include_baselines:
        selected_models.extend(
            dict.fromkeys(
                BASELINE_FOR[name]
                for name in config.models
                if name in BASELINE_FOR and BASELINE_FOR[name] not in selected_models
            )
        )
    run_info = {
        "versions": {
            package: version(package)
            for package in ("torch", "transformer-lens", "transformers")
        },
        **asdict(config),
        "output_dir": str(config.output_dir),
        "model_specs": {
            name: {
                "label": MODEL_LABELS[name],
                "model_name": MODEL_CHECKPOINTS.get(name, name),
            }
            for name in selected_models
        },
        "indices": indices,
        "sample_count": len(indices),
        "split_size": split_size,
        "batch_size": 1,
        "compute_pairwise": config.experiment
        in {"cyclic_similarity", "bos_comparison"},
        "compute_differences": config.experiment == "convergence",
        "forward_hooks": [name for name, _ in fwd_hooks or []],
        "reduction": "notebook" if cache_reducer is not None else "generic",
        "save_profiles": save_profiles and cache.enabled,
        "tokenization": {},
    }
    if cache.enabled:
        run_info["cache"] = cache.settings
    results = {}
    for name in selected_models:
        pair_config = replace(config, num_steps=1) if name in BASELINES else config
        inputs = dict(
            task=config.task,
            indices=indices,
            texts=texts,
            intervention=intervention,
            input_protocol="likelihood_context_and_reference_v1",
        )
        key = cache.key(
            pair_config,
            name,
            analysis_version=analysis_version,
            residual_entropy=config.compute_residual_entropy,
            **inputs,
        )
        profile_key = cache.key(pair_config, name, **inputs)
        # In require mode, complete shared measurements are sufficient to draw
        # stages even if its notebook-specific summary has not been saved yet.
        if (
            cache.mode == "require"
            and config.experiment == "stages_of_inference"
            and not config.compute_residual_entropy
            and not fwd_hooks
            and cache_reducer is None
            and cache.read(config.experiment, key) is None
        ):
            shared = cache.read("layer_profiles", profile_key)
            if shared is not None:
                assert summarize is not None
                results[name] = [
                    summarize(_unpack_profiles(p)) for p in shared["profiles"]
                ]
                run_info["tokenization"][name] = shared["tokenization"]
                continue
        with cache.entry(config.experiment, key, device=config.device) as entry:
            if entry.complete:
                saved = entry.load()
                results[name], run_info["tokenization"][name] = (
                    saved["examples"],
                    saved["tokenization"],
                )
                continue
            # Convergence already computed these metrics. Stages retains its own
            # reduction, including the per-head sink threshold and example weights.
            shared = (
                cache.read("layer_profiles", profile_key)
                if (
                    config.experiment == "stages_of_inference"
                    and not config.compute_residual_entropy
                    and not fwd_hooks
                    and cache_reducer is None
                )
                else None
            )
            if shared is not None:
                assert summarize is not None
                examples = [summarize(_unpack_profiles(p)) for p in shared["profiles"]]
                saved = {"examples": examples, "tokenization": shared["tokenization"]}
                entry.save(saved)
                results[name], run_info["tokenization"][name] = (
                    examples,
                    shared["tokenization"],
                )
                continue
            _set_seed(config.seed)
            lm = (
                _load_model(name, pair_config, cache=cache)
                if cache.enabled
                else _load_model(name, pair_config)
            )
            try:
                tokenization = {
                    **tokenizer_metadata(lm.bridge.tokenizer),
                    "bos_policy": config.bos_policy,
                    "model_revision": getattr(
                        lm.bridge.original_model.config, "_commit_hash", None
                    ),
                    "inputs": [],
                }
                examples, profiles = [], []
                records = entry.records()
                # simple_evaluate previously reset the seed after model loading.
                _set_seed(config.seed)
                if records:
                    examples = [record["value"]["summary"] for record in records]
                    profiles = [record["value"]["profile"] for record in records]
                    tokenization["inputs"] = [
                        record["value"]["input"] for record in records
                    ]
                    restore_rng(records[-1]["rng"], config.device)
                del records
                # A completed convergence entry can be reused without profiles.
                # Likewise, do not publish an incomplete profile set when a run
                # originally started without them. Stages can collect its own.
                collect_profiles = (
                    save_profiles
                    and cache.enabled
                    and all(profile is not None for profile in profiles)
                    and cache.read("layer_profiles", profile_key) is None
                )
                names_filter = (
                    names_filter_factory(lm.structure)
                    if names_filter_factory is not None and not collect_profiles
                    else metrics_names_filter
                )
                with lm.bridge.hooks(fwd_hooks=fwd_hooks or []):
                    for position in tqdm(
                        range(len(examples), len(indices)), desc=f"{name}: examples"
                    ):
                        context, continuation = lm._encode_pair(*texts[position])
                        input_ids = torch.tensor(
                            [context + continuation], device=lm.device
                        )
                        with torch.no_grad():
                            output, activations = lm.bridge.run_with_cache(
                                input_ids,
                                accumulate=True,
                                names_filter=names_filter,
                                device="cpu",
                                **lm.forward_kwargs,
                            )
                        del output
                        run = None
                        if cache_reducer is None:
                            assert summarize is not None
                            (run,) = compute_model_run_data(
                                activations,
                                input_ids,
                                lm.structure,
                                compute_pairwise=lm.compute_pairwise,
                                compute_differences=lm.compute_differences,
                                compute_residual_entropy=lm.compute_residual_entropy,
                                residual_entropy_components=lm.residual_entropy_components,
                            )
                            examples.append(summarize(run))
                        else:
                            (reduced,) = cache_reducer(
                                activations, input_ids, lm.structure
                            )
                            examples.append(reduced["summary"])
                            del reduced
                        if collect_profiles:
                            if run is None:
                                (run,) = compute_model_run_data(
                                    activations,
                                    input_ids,
                                    lm.structure,
                                    compute_pairwise=False,
                                    compute_differences=False,
                                )
                            profiles.append(_pack_profiles(run))
                        else:
                            profiles.append(None)
                        del activations, run
                        input_ids = input_ids.cpu()
                        tokenization["inputs"].append(
                            {
                                "dataset_index": indices[position],
                                "length": input_ids.numel(),
                                "first_ids": input_ids.flatten()[:8].tolist(),
                                "sha256": hashlib.sha256(
                                    input_ids.numpy().tobytes()
                                ).hexdigest(),
                            }
                        )
                        entry.append(
                            {
                                "summary": examples[-1],
                                "profile": profiles[-1],
                                "input": tokenization["inputs"][-1],
                            }
                        )
                if collect_profiles:
                    with cache.entry("layer_profiles", profile_key) as profile_entry:
                        if not profile_entry.complete:
                            profile_entry.save(
                                {"profiles": profiles, "tokenization": tokenization}
                            )
                entry.save({"examples": examples, "tokenization": tokenization})
                results[name], run_info["tokenization"][name] = examples, tokenization
            finally:
                lm.all_model_data.clear()
                del lm
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return results, run_info


def save_results(summaries: dict, run_info: dict, output_dir: Path) -> Path:
    """Export reduced results and settings to a new timestamped directory.

    Parameters
    ----------
    summaries : dict
        JSON-compatible reduced results, typically keyed by model name.
    run_info : dict
        JSON-compatible run settings containing ``experiment`` and ``mode``.
        An optional ``dataset`` key adds a subdirectory for dataset analyses.
    output_dir : pathlib.Path
        Root beneath which experiment, dataset, and timestamp directories are made.

    Returns
    -------
    pathlib.Path
        New directory containing ``summaries.json`` and ``run.json``.

    Raises
    ------
    FileExistsError
        If the generated timestamped directory already exists.
    TypeError
        If a value cannot be serialized as JSON.
    ValueError
        If a value contains NaN or infinity.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output_dir / run_info["experiment"]
    if run_info.get("dataset") is not None:
        output /= run_info["dataset"]
    output /= f"{run_info['mode']}-{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    for filename, value in (("summaries.json", summaries), ("run.json", run_info)):
        with (output / filename).open("w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return output


def load_experiment_model(name, config, *, steps=None, cache=None):
    """Resolve the paper's short checkpoint names through the library loader."""
    from looped_internals.models import load_model

    return load_model(
        (
            cache.model(name)[0]
            if cache is not None and cache.enabled
            else MODEL_CHECKPOINTS.get(name, name)
        ),
        device=config.device,
        dtype=getattr(torch, config.dtype),
        num_steps=(config.num_steps if steps is None else steps)
        if name in MODELS or name in STABILITY
        else None,
    )


class ModelSession:
    """Load a checkpoint on the first cache miss and release it on exit."""

    def __init__(self, name, config, cache):
        self.name, self.config, self.cache = name, config, cache
        self.loaded = None

    def __enter__(self):
        return self

    def get(self):
        if self.loaded is None:
            _set_seed(self.config.seed)
            kwargs = {"cache": self.cache} if self.cache.enabled else {}
            self.loaded = load_experiment_model(self.name, self.config, **kwargs)
        return self.loaded

    def __exit__(self, *_):
        self.loaded = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def create_run_dir(config, **parameters):
    """Save resolved settings and dependency versions before a long collection."""
    from importlib.metadata import version

    metadata = {**asdict(config), "output_dir": str(config.output_dir), **parameters}
    metadata["versions"] = {
        name: version(name) for name in ("torch", "transformer-lens", "transformers")
    }
    return save_results({}, metadata, config.output_dir)


def record_tokenization(run_dir, name, bridge):
    """Record checkpoint tokenization for experiments that collect directly."""
    write_tokenization(
        run_dir, name, {"bos_policy": "native", **tokenizer_metadata(bridge.tokenizer)}
    )


def write_tokenization(run_dir, name, metadata):
    """Retain the original tokenizer provenance on cache hits as well as misses."""
    path = run_dir / "tokenization.json"
    records = json.loads(path.read_text()) if path.exists() else {}
    records[name] = metadata
    path.write_text(json.dumps(records, indent=2) + "\n")


def shorten_label(label: str) -> str:
    """Abbreviate plot labels while preserving the retained letters' case.

    Shorten whole words: residual -> resid, attention -> attn,
    cosine -> cos, similarity -> sim.
    Remove "pattern", including adjacent hyphens and extra whitespace.
    Drop Frobenius/Euclidean only when they qualify "difference norm".
    """

    def abbreviate(match: re.Match[str]) -> str:
        word = match[0]
        if word.lower() == "residual":
            return word[:5]
        if word.lower() == "attention":
            return word[:3] + word[-1]
        return word[:3]

    label = re.sub(r"[-\s]*\bpattern\b[-\s]*", " ", label, flags=re.IGNORECASE).strip()
    label = re.sub(
        r"\b(?:Frobenius|Euclidean)\s+(?=difference\s+norm\b)",
        "",
        label,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"\b(?:residual|attention|cosine|similarity)\b",
        abbreviate,
        label,
        flags=re.IGNORECASE,
    )


def get_figsize(
    fullpage: bool,
    subplots: tuple[int, int],
    height_per_subplot: float = 1.0,
    fraction: float = 1.0,
) -> tuple[float, float]:
    """Return paper figure dimensions in inches, without later LaTeX scaling.

    Use the original paper's text width (6.75 inches) when ``fullpage=True``
    and column width (3.25 inches) otherwise. ``fraction`` scales that width;
    the number of subplot columns does not. Height is the number of subplot
    rows multiplied by ``height_per_subplot``, which is already in inches.
    Save with ``save_figure(..., bbox_inches=None)`` to retain these dimensions.
    """
    width = ICML_TWO_COLUMN_WIDTH if fullpage else ICML_SINGLE_COLUMN_WIDTH
    return width * fraction / 72.27, height_per_subplot * subplots[0]


def model_subplots(
    model_count: int, *, height_per_subplot: float = 1.5, columns: int = 4
):
    """Create full-width model panels; return only the used axes."""
    import matplotlib.pyplot as plt

    columns = min(columns, model_count)
    rows = (model_count + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=get_figsize(True, (rows, columns), height_per_subplot),
        layout="constrained",
    )
    for ax in axes.flat[model_count:]:
        ax.remove()
    return fig, axes.ravel()[:model_count]


def add_prelude_coda_connectors(ax, lines):
    """Join boundary endpoints in an ordered mapping of profile names to lines."""
    from matplotlib.collections import LineCollection

    recurrences = [name for name in lines if name.startswith("R")]
    if not recurrences:
        return
    segments = []
    for before, after in (("Prelude", recurrences[0]), (recurrences[-1], "Coda")):
        if before not in lines or after not in lines:
            continue
        left, right = lines[before].get_xydata(), lines[after].get_xydata()
        if not len(left) or not len(right):
            continue
        segment = np.array([left[-1], right[0]])
        if np.isfinite(segment).all():
            segments.append(segment)
    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors="grey",
                linestyles="--",
                linewidths=0.8,
                alpha=0.45,
                zorder=1,
                label="_nolegend_",
                gid="prelude_coda_connectors",
            ),
            autolim=False,
        )


def add_prelude_coda_legend(fig, *, include_coda=True, include_control=True):
    """Put the shared profile legend in one row above the subplot titles."""
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color="orange", linewidth=2, label="Prelude")]
    if include_coda:
        handles.append(Line2D([], [], color="black", linewidth=2, label="Coda"))
    if include_control:
        handles.append(
            Line2D([], [], color="red", linestyle="--", linewidth=2, label="Control")
        )
    return fig.legend(handles=handles, loc="outside upper center", ncols=len(handles))


def add_depth_colorbar(
    fig, axes, kind, *, mappable=None, cax=None, label=None, pad=0.025
):
    """Add a slim layer/recurrence scale, with Early below and Late above."""
    import matplotlib.pyplot as plt

    axes = list(np.asarray(axes).flat)
    if mappable is None:
        mappable = plt.cm.ScalarMappable(
            cmap=COLORMAPS[kind + "s"], norm=plt.Normalize(0, 1)
        )
    height = fig.get_figheight() * (
        max(ax.get_position().y1 for ax in axes)
        - min(ax.get_position().y0 for ax in axes)
    )
    colorbar = fig.colorbar(
        mappable,
        ax=axes,
        cax=cax,
        fraction=0.025,
        pad=pad,
        shrink=0.8,
        # Aim for a 0.04-inch bar, independent of the number of subplot rows.
        aspect=0.8 * height / 0.04,
    )
    colorbar.set_ticks([])
    colorbar.ax.set_xlabel("Early", labelpad=3)
    colorbar.ax.set_title("Late", pad=3)
    colorbar.set_label(label or kind.capitalize(), labelpad=3)
    return colorbar


def save_figure(fig, run_dir, name, *, bbox_inches: str | None = "tight"):
    """Write a publication PDF, display it, then release the figure.

    Set ``bbox_inches=None`` to preserve the exact figure size, even when the
    active style requests tight cropping. The default crops surrounding space.
    """
    import matplotlib.pyplot as plt

    # Passing None to savefig alone would fall back to the active style's bbox.
    with plt.rc_context({"savefig.bbox": bbox_inches}):
        fig.savefig(run_dir / f"{name}.pdf")
    plt.show()
    plt.close(fig)


def export_paper_figures(
    config, run_dir: Path, *, dataset: str | None = None
) -> Path | None:
    """Copy a completed full run's PDFs to the stable paper directory.

    Call this after all collection, plotting, and numerical exports succeed.
    Quick runs do nothing. Dataset-specific notebooks pass their dataset name;
    figures spanning datasets or using synthetic inputs omit it.

    The destination is ``plots/mechanistic_analysis/<experiment>/<dataset>/``
    beneath the repository root, with the dataset component omitted when absent.
    Run subdirectories become filename prefixes, so convergence's
    ``eight-step/<metric>.pdf`` becomes ``eight-step-<metric>.pdf``.
    Matching PDFs are replaced; other files and the run archive are preserved.
    Return the destination for full runs and ``None`` for quick runs.
    """
    if config.mode != "full":
        return None

    destination = PAPER_PLOTS_DIR / config.experiment
    if dataset is not None:
        destination /= dataset
    destination.mkdir(parents=True, exist_ok=True)
    for figure in sorted(run_dir.rglob("*.pdf")):
        filename = "-".join(figure.relative_to(run_dir).parts)
        shutil.copy2(figure, destination / filename)
    return destination
