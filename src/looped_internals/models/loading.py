"""Load registered checkpoints or custom specs as ``(bridge, structure)`` pairs.

``load_model`` provides the same inference options for every model. Checkpoint
selection, layer layout, and defaults live in ``MODEL_SPECS``; the backends
handle Hugging Face checkpoints, Parcae checkpoints, and randomly initialized
minimal models.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, TypeAlias, cast

import torch
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformers import AutoTokenizer

from looped_internals.models._compat.ouro import prepare_ouro_source
from looped_internals.models._compat.raven import prepare_raven_source
from looped_internals.models.minimal import MinimalRecurrentConfig
from looped_internals.models.parcae import ParcaeCheckpoint, load_parcae
from looped_internals.models.tokenization import tokenizer_metadata
from looped_internals.structure import LayerGroup, ModelLayerStructure

LoadedModel: TypeAlias = tuple[TransformerBridge, ModelLayerStructure]
LayerLayout: TypeAlias = Literal["standard", "recurrent", "sandwich"]

# Hugging Face account publishing the recurrence-ablated controls and the
# injection-stability sweep. Set PARCAE_MODELS_ORG to read them from a fork.
# A local export directory can still be loaded by passing an explicit
# ``ParcaeCheckpoint("/path/to/export")`` instead of a registered name.
PARCAE_MODELS_ORG = os.environ.get("PARCAE_MODELS_ORG", "blayney")
PARCAE_SWEEP_REPO = f"{PARCAE_MODELS_ORG}/parcae-stability-sweep"
_PARCAE_SIZES = ("140m", "370m", "770m", "1_3b")
# Recurrence-ablated controls were trained only for these sizes.
_PARCAE_CONTROL_SIZES = ("140m", "370m", "770m")
# Injection spectral radius, fixed at these values or left learned, crossed with
# the two recurrence samplers. Arm names are the sweep's own directory names.
_STABILITY_ARMS = {
    f"stab_{schedule}_{rho.replace('.', '_')}": f"{schedule}_rho{rho}"
    for schedule in ("fixed", "poisson")
    for rho in ("0.25", "0.5", "0.75", "1.0", "1.25", "learned")
}


@dataclass(frozen=True)
class ModelSpec:
    """Checkpoint ID or toy configuration, with its loading defaults.

    String sources use the Hugging Face backend. ``ParcaeCheckpoint`` sources
    load ``parcae_lm`` models; their layout is always prelude, core, coda.
    ``MinimalRecurrentConfig`` sources create a fresh toy model; their layout is
    always entirely recurrent. ``default_dtype=None`` means bfloat16 for
    checkpoints and float32 for toy models. ``default_steps=None`` preserves the
    source configuration's recurrence count.
    """

    source: str | ParcaeCheckpoint | MinimalRecurrentConfig
    revision: str | None = None
    layout: LayerLayout | ModelLayerStructure = "standard"
    recurrence_parameter: str | None = None
    prepare_source: Callable[..., None] | None = None
    default_dtype: torch.dtype | None = None
    default_steps: int | None = None
    trust_remote_code: bool = False
    unsupported_dtypes: tuple[torch.dtype, ...] = ()


@dataclass(frozen=True)
class LoadOptions:
    """Resolved options accepted by every loading backend."""

    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    num_steps: int | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)


class ModelLoader(Protocol):
    """Static interface checked when backends are added to ``_LOADERS``."""

    def __call__(self, spec: ModelSpec, options: LoadOptions) -> LoadedModel: ...


MODEL_SPECS: dict[str, ModelSpec] = {
    "raven": ModelSpec(
        "tomg-group-umd/huginn-0125",
        layout="sandwich",
        recurrence_parameter="mean_recurrence",
        prepare_source=prepare_raven_source,
        trust_remote_code=True,
    ),
    **{
        f"ouro_{size.lower().replace('.', '_')}": ModelSpec(
            f"ByteDance/Ouro-{size}",
            layout="recurrent",
            recurrence_parameter="total_ut_steps",
            prepare_source=prepare_ouro_source,
            default_steps=4,
            trust_remote_code=True,
        )
        for size in ("1.4B", "2.6B")
    },
    **{
        f"retrofitted_{base}_{steps}": ModelSpec(
            f"smcleish/Recurrent-{full_name}-train-recurrence-{steps}",
            layout="sandwich",
            recurrence_parameter="mean_recurrence",
            prepare_source=prepare_raven_source,
            default_steps=32,
            trust_remote_code=True,
        )
        for base, full_name in (
            ("llama", "Llama-3.2"),
            ("olmo", "OLMo-2-0425"),
            ("tinyllama", "TinyLlama-3T"),
        )
        for steps in (4, 8, 16, 32)
    },
    **{
        f"parcae_{size}": ModelSpec(
            ParcaeCheckpoint(f"SandyResearch/parcae-{size.replace('_', '.')}"),
            layout="sandwich",
            default_steps=8,
        )
        for size in _PARCAE_SIZES
    },
    **{
        f"parcae_{size}_rec1": ModelSpec(
            ParcaeCheckpoint(
                f"{PARCAE_MODELS_ORG}/parcae-{size.replace('_', '.')}-rec1"
            ),
            layout="sandwich",
            default_steps=1,
        )
        for size in _PARCAE_CONTROL_SIZES
    },
    **{
        name: ModelSpec(
            ParcaeCheckpoint(PARCAE_SWEEP_REPO, subfolder=arm),
            layout="sandwich",
            default_steps=8,
        )
        for name, arm in _STABILITY_ARMS.items()
    },
    "gpt2": ModelSpec("gpt2", default_dtype=torch.float32),
    **{
        f"pythia_{size}": ModelSpec(
            f"EleutherAI/pythia-{size}-deduped",
            unsupported_dtypes=(torch.float16,),
        )
        for size in (
            "70m",
            "160m",
            "410m",
            "1b",
            "1.4b",
            "2.8b",
            "6.9b",
            "12b",
        )
    },
    **{
        f"llama_{size.lower()}": ModelSpec(repo_id)
        for size, repo_id in (
            ("1B", "meta-llama/Llama-3.2-1B"),
            ("3B", "meta-llama/Llama-3.2-3B"),
            ("8B", "meta-llama/Meta-Llama-3.1-8B"),
        )
    },
    **{f"gemma_{size}": ModelSpec(f"google/gemma-{size}") for size in ("2b", "7b")},
    "tinyllama": ModelSpec("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"),
    "olmo2": ModelSpec("allenai/OLMo-2-0425-1B", trust_remote_code=True),
    "olmo2_instruct": ModelSpec(
        "allenai/OLMo-2-0425-1B-Instruct", trust_remote_code=True
    ),
    "minimal": ModelSpec(MinimalRecurrentConfig()),
}


def _load_hf(spec: ModelSpec, options: LoadOptions) -> LoadedModel:
    assert isinstance(spec.source, str)
    overrides = dict(options.config_overrides)
    if options.num_steps is not None:
        if spec.recurrence_parameter is None:
            raise ValueError("Set ModelSpec.recurrence_parameter for this architecture")
        overrides.setdefault(spec.recurrence_parameter, options.num_steps)
    # Remote config constructors require the checkpoint-local source
    # fixes before AutoConfig is called inside boot_transformers.
    if spec.prepare_source is not None:
        spec.prepare_source(spec.source, revision=spec.revision)
    bridge = TransformerBridge.boot_transformers(
        spec.source,
        hf_config_overrides=overrides or None,
        device=options.device,
        dtype=options.dtype,
        trust_remote_code=spec.trust_remote_code,
        revision=spec.revision,
    )
    # TL loads/reloads tokenizers with add_bos_token=True, even when the
    # checkpoint does not. Restore the checkpoint tokenizer after bridge setup.
    tokenizer = AutoTokenizer.from_pretrained(
        spec.source,
        config=bridge.original_model.config,
        revision=getattr(bridge.original_model.config, "_commit_hash", None),
        trust_remote_code=spec.trust_remote_code,
    )
    if tokenizer is None:
        raise ValueError(f"Checkpoint {spec.source!r} did not provide a tokenizer")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Bypass the public setter: reassignment calls setup_tokenizer, which
    # reloads with BOS enabled again. This is the same checkpoint vocabulary;
    # retain the model's vocabulary dimensions and update tokenization flags.
    bridge._tokenizer = tokenizer
    tokenization = tokenizer_metadata(tokenizer)
    bridge.cfg.default_prepend_bos = tokenization["prepends_bos"]
    bridge.cfg.tokenizer_prepends_bos = tokenization["prepends_bos"]
    bridge.cfg.tokenizer_appends_eos = tokenization["appends_eos"]
    if isinstance(spec.layout, ModelLayerStructure):
        return bridge, spec.layout
    if spec.layout == "sandwich":
        config = cast(Any, bridge.original_model.config)
        groups = [
            LayerGroup("prelude_blocks", config.n_layers_in_prelude, False),
            LayerGroup("core_blocks", config.n_layers_in_recurrent_block, True),
            LayerGroup("coda_blocks", config.n_layers_in_coda, False),
        ]
    else:
        groups = [LayerGroup("blocks", bridge.cfg.n_layers, spec.layout == "recurrent")]
    return bridge, ModelLayerStructure(groups=groups)


def _load_minimal(spec: ModelSpec, options: LoadOptions) -> LoadedModel:
    from transformer_lens.config import TransformerBridgeConfig

    from looped_internals.models.adapters.minimal import MinimalRecurrentAdapter
    from looped_internals.models.minimal import MinimalRecurrentLM

    assert isinstance(spec.source, MinimalRecurrentConfig)
    overrides = dict(options.config_overrides)
    if options.num_steps is not None:
        overrides.setdefault("num_cycles", options.num_steps)
    config = replace(spec.source, **overrides)
    model = MinimalRecurrentLM(config).to(device=options.device, dtype=options.dtype)
    tl_config = TransformerBridgeConfig(
        d_model=config.d_model,
        d_head=config.d_model // config.n_heads,
        n_layers=config.n_layers,
        n_ctx=2048,
        n_heads=config.n_heads,
        d_vocab=config.d_vocab,
        architecture="MinimalRecurrentLM",
    )
    adapter = MinimalRecurrentAdapter(tl_config, gated_mlp=config.gated_mlp)
    bridge = TransformerBridge(model, adapter, tokenizer=None)
    return bridge, ModelLayerStructure(
        groups=[LayerGroup("blocks", config.n_layers, True)]
    )


def _load_parcae(spec: ModelSpec, options: LoadOptions) -> LoadedModel:
    assert isinstance(spec.source, ParcaeCheckpoint)
    return load_parcae(
        spec.source,
        device=options.device,
        dtype=options.dtype,
        num_steps=options.num_steps,
        config_overrides=options.config_overrides,
    )


_LOADERS: dict[str, ModelLoader] = {
    "hf": _load_hf,
    "parcae": _load_parcae,
    "minimal": _load_minimal,
}
_BACKENDS: dict[type, str] = {
    str: "hf",
    ParcaeCheckpoint: "parcae",
    MinimalRecurrentConfig: "minimal",
}


def load_model(
    name: str | ModelSpec,
    *,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
    num_steps: int | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> LoadedModel:
    """Return an evaluation-mode bridge and its layer structure.

    HF models use the checkpoint's published tokenizer defaults, undoing
    TransformerLens's forced BOS insertion. Direct tokenizer calls and the
    bridge's default ``to_tokens`` agree about whether to prepend BOS.

    Args:
        name: Key in ``MODEL_SPECS``, or a custom ``ModelSpec``. For example,
            ``"retrofitted_llama_32"`` selects the checkpoint trained at 32 steps;
            ``"parcae_140m_rec1"`` selects the one-recurrence Parcae control;
            ``ModelSpec(MinimalRecurrentConfig(...))`` creates a custom toy model.
        device: Target device.
        dtype: Parameter dtype; ``None`` uses the model's default.
        num_steps: Exact core recurrence count at inference; ``None`` uses the
            model's default. Non-recurrent models require ``None``.
        config_overrides: Native configuration fields, applied last (including
            over ``num_steps``). Toy overrides are passed to ``dataclasses.replace``.

    Raven's and Parcae's native forward ``num_steps`` argument can still override
    the loaded default for an individual call. Loading a model never changes its
    checkpoint.
    """
    if isinstance(name, str):
        try:
            spec = MODEL_SPECS[name]
        except KeyError:
            raise ValueError(
                f"Unknown model {name!r}; choose from {list(MODEL_SPECS)} "
                "or pass a ModelSpec"
            ) from None
    else:
        spec = name
    backend = _BACKENDS[type(spec.source)]
    steps = spec.default_steps if num_steps is None else num_steps
    if steps is not None:
        if type(steps) is not int or steps < 1:
            raise ValueError("num_steps must be a positive integer or None")
        if backend == "hf" and spec.layout == "standard":
            raise ValueError("num_steps applies only to recurrent models")
    resolved_dtype = (
        dtype
        or spec.default_dtype
        or (torch.float32 if backend == "minimal" else torch.bfloat16)
    )
    if resolved_dtype in spec.unsupported_dtypes:
        raise ValueError(f"{spec.source} does not support dtype={resolved_dtype}")
    options = LoadOptions(
        device=device,
        dtype=resolved_dtype,
        num_steps=steps,
        config_overrides=dict(config_overrides or {}),
    )
    bridge, structure = _LOADERS[backend](spec, options)
    # TransformerBridge does not register the original model as a child module.
    bridge.original_model.eval()
    bridge.eval()
    return bridge, structure
