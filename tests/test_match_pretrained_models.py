"""Full-checkpoint output matching for repo model loaders.

These tests compare models loaded via ``looped_internals.models`` against the
plain HuggingFace implementation, or the native ``parcae_lm`` model, for the
same pretrained checkpoint. Recurrent cases include the recurrence counts of
the paper's main analyses. They are memory-intensive because several cases load
1B+ parameter checkpoints. Set ``LOOPED_MATCH_DEVICE=cuda`` to run on a GPU;
the default is CPU.
"""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial

import pytest
import torch
from parcae_lm.attention_backends import flash_attention
from parcae_lm.models.parcae.config import ParcaeConfig

import looped_internals  # noqa: F401 - registers local TransformerBridge patches
from looped_internals.capture import metrics_names_filter
from looped_internals.models import MODEL_SPECS, LoadedModel, load_model
from looped_internals.models.parcae import _resolve
from looped_internals.structure import ModelLayerStructure

DEVICE = os.environ.get("LOOPED_MATCH_DEVICE", "cpu")
DTYPE = torch.float32
ATTN_IMPLEMENTATION = "eager"
INPUT_IDS = torch.tensor([[1, 150, 892, 338, 263, 1332, 29889]], dtype=torch.long)
FORWARD_SEED = 42

pytestmark = pytest.mark.hf_pretrained


@dataclass(frozen=True)
class PretrainedMatchCase:
    name: str
    repo_id: str
    load_bridge: Callable[..., LoadedModel]
    dtype: torch.dtype = DTYPE
    attn_implementation: str | None = ATTN_IMPLEMENTATION
    trust_remote_code: bool = False
    hf_config_overrides: dict[str, object] = field(default_factory=dict)
    forward_kwargs: dict[str, object] = field(default_factory=dict)
    recurrences: int = 1
    bridge_return_type: str | None = "logits"
    output_key: str | None = None
    forward_seed: int = FORWARD_SEED
    atol: float = 1e-2
    rtol: float = 1e-2


CASES = [
    PretrainedMatchCase(
        name="gpt2",
        repo_id="gpt2",
        load_bridge=partial(load_model, "gpt2"),
        atol=1e-4,
        rtol=1e-4,
    ),
    PretrainedMatchCase(
        name="llama_3_2_1b",
        repo_id="meta-llama/Llama-3.2-1B",
        load_bridge=partial(load_model, "llama_1b"),
    ),
    PretrainedMatchCase(
        name="tinyllama",
        repo_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        load_bridge=partial(load_model, "tinyllama"),
    ),
    PretrainedMatchCase(
        name="olmo2_1b",
        repo_id="allenai/OLMo-2-0425-1B",
        load_bridge=partial(load_model, "olmo2"),
        trust_remote_code=True,
    ),
    PretrainedMatchCase(
        name="ouro_1_4b_4_steps",
        repo_id="ByteDance/Ouro-1.4B",
        load_bridge=partial(load_model, "ouro_1_4b", num_steps=4),
        trust_remote_code=True,
        hf_config_overrides={"total_ut_steps": 4},
        recurrences=4,
    ),
    PretrainedMatchCase(
        name="ouro_2_6b_4_steps",
        repo_id="ByteDance/Ouro-2.6B",
        load_bridge=partial(load_model, "ouro_2_6b", num_steps=4),
        trust_remote_code=True,
        hf_config_overrides={"total_ut_steps": 4},
        recurrences=4,
    ),
    PretrainedMatchCase(
        name="huginn_4_steps",
        repo_id="tomg-group-umd/huginn-0125",
        load_bridge=partial(load_model, "raven", num_steps=4),
        trust_remote_code=True,
        hf_config_overrides={"mean_recurrence": 4},
        forward_kwargs={"num_steps": 4},
        recurrences=4,
    ),
    PretrainedMatchCase(
        name="huginn_32_steps",
        repo_id="tomg-group-umd/huginn-0125",
        load_bridge=partial(load_model, "raven", num_steps=32),
        trust_remote_code=True,
        hf_config_overrides={"mean_recurrence": 32},
        forward_kwargs={"num_steps": 32},
        recurrences=32,
    ),
    PretrainedMatchCase(
        name="retrofitted_llama_4_steps",
        repo_id="smcleish/Recurrent-Llama-3.2-train-recurrence-32",
        load_bridge=partial(load_model, "retrofitted_llama_32", num_steps=4),
        trust_remote_code=True,
        hf_config_overrides={"mean_recurrence": 4},
        forward_kwargs={"num_steps": 4},
        recurrences=4,
    ),
    PretrainedMatchCase(
        name="retrofitted_llama_32_steps",
        repo_id="smcleish/Recurrent-Llama-3.2-train-recurrence-32",
        load_bridge=partial(load_model, "retrofitted_llama_32", num_steps=32),
        trust_remote_code=True,
        hf_config_overrides={"mean_recurrence": 32},
        forward_kwargs={"num_steps": 32},
        recurrences=32,
    ),
    PretrainedMatchCase(
        name="retrofitted_olmo_4_steps",
        repo_id="smcleish/Recurrent-OLMo-2-0425-train-recurrence-32",
        load_bridge=partial(load_model, "retrofitted_olmo_32", num_steps=4),
        trust_remote_code=True,
        hf_config_overrides={"mean_recurrence": 4},
        forward_kwargs={"num_steps": 4},
        recurrences=4,
    ),
    PretrainedMatchCase(
        name="retrofitted_tinyllama_4_steps",
        repo_id="smcleish/Recurrent-TinyLlama-3T-train-recurrence-32",
        load_bridge=partial(load_model, "retrofitted_tinyllama_32", num_steps=4),
        trust_remote_code=True,
        hf_config_overrides={"mean_recurrence": 4},
        forward_kwargs={"num_steps": 4},
        recurrences=4,
    ),
]


def _load_or_skip(repo_id: str, load_fn):
    try:
        return load_fn()
    except OSError as exc:
        pytest.skip(f"{repo_id} is not available: {exc}")


def _load_hf_model(case: PretrainedMatchCase):
    from transformers import AutoConfig, AutoModelForCausalLM

    if case.trust_remote_code:
        from looped_internals.models._compat.ouro import prepare_ouro_source
        from looped_internals.models._compat.raven import prepare_raven_source

        math_sdp_enabled = torch.backends.cuda.math_sdp_enabled()
        _load_or_skip(case.repo_id, lambda: prepare_raven_source(case.repo_id))
        _load_or_skip(case.repo_id, lambda: prepare_ouro_source(case.repo_id))
        assert torch.backends.cuda.math_sdp_enabled() == math_sdp_enabled

    config = _load_or_skip(
        case.repo_id,
        lambda: AutoConfig.from_pretrained(
            case.repo_id, trust_remote_code=case.trust_remote_code
        ),
    )
    for key, value in case.hf_config_overrides.items():
        setattr(config, key, value)
    if config.model_type == "ouro":
        from looped_internals.models._compat.ouro import prepare_ouro_config

        prepare_ouro_config(config)
    if case.attn_implementation is not None:
        setattr(config, "attn_implementation", case.attn_implementation)
        setattr(config, "_attn_implementation", case.attn_implementation)

    model_kwargs = {
        "config": config,
        "dtype": case.dtype,
        "trust_remote_code": case.trust_remote_code,
    }
    if case.attn_implementation is not None:
        model_kwargs["attn_implementation"] = case.attn_implementation

    return _load_or_skip(
        case.repo_id,
        lambda: AutoModelForCausalLM.from_pretrained(
            case.repo_id,
            **model_kwargs,
        ).to(DEVICE),
    )


def _extract_logits(output, case: PretrainedMatchCase) -> torch.Tensor:
    if case.output_key is not None:
        return output[case.output_key]
    if hasattr(output, "logits"):
        return output.logits
    return output


def _assert_parameter_dtype(
    case: PretrainedMatchCase,
    model: torch.nn.Module,
    label: str,
) -> None:
    parameter = next(model.parameters())
    assert parameter.dtype == case.dtype, (
        f"{case.name} {label} parameter dtype mismatch: "
        f"expected {case.dtype}, got {parameter.dtype}"
    )


def _seed_forward_rng(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _fork_forward_rng():
    devices = [torch.device(DEVICE)] if torch.device(DEVICE).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        yield


def _empty_device_cache() -> None:
    if torch.device(DEVICE).type == "cuda":
        torch.cuda.empty_cache()


def _assert_hook_iterations(
    cache, structure: ModelLayerStructure, recurrences: int
) -> None:
    """Each recurrent layer records one activation per recurrence."""
    assert any(".attn.hook_pattern.iter_" in key for key in cache)
    for group in structure.groups:
        if not group.n_layers:
            continue
        expected_steps = recurrences if group.is_recurrent else 1
        assert (
            cache.get_iterations(f"{group.hook_prefix}.0.hook_out").shape[0]
            == expected_steps
        )


def _bridge_config_overrides(case: PretrainedMatchCase) -> dict[str, object]:
    overrides = dict(case.hf_config_overrides)
    if case.attn_implementation is not None:
        overrides["attn_implementation"] = case.attn_implementation
        overrides["_attn_implementation"] = case.attn_implementation
    return overrides


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_models_loader_matches_full_pretrained_huggingface_logits(
    case: PretrainedMatchCase,
):
    input_ids = INPUT_IDS.to(DEVICE)

    hf_model = _load_hf_model(case)
    hf_model.eval()
    _assert_parameter_dtype(case, hf_model, "HF")
    with torch.no_grad():
        with _fork_forward_rng():
            _seed_forward_rng(case.forward_seed)
            hf_logits = (
                _extract_logits(hf_model(input_ids, **case.forward_kwargs), case)
                .detach()
                .float()
                .cpu()
            )

    del hf_model
    gc.collect()
    _empty_device_cache()

    bridge_overrides = _bridge_config_overrides(case)
    bridge, structure = _load_or_skip(
        case.repo_id,
        lambda: case.load_bridge(
            device=DEVICE,
            dtype=case.dtype,
            config_overrides=bridge_overrides,
        ),
    )
    bridge.eval()
    _assert_parameter_dtype(case, bridge.original_model, "bridge")
    with torch.no_grad():
        with _fork_forward_rng():
            _seed_forward_rng(case.forward_seed)
            bridge_kwargs = dict(case.forward_kwargs)
            if case.bridge_return_type is not None:
                bridge_kwargs["return_type"] = case.bridge_return_type
            bridge_logits = _extract_logits(bridge(input_ids, **bridge_kwargs), case)
            bridge_logits = bridge_logits.detach().float().cpu()
        with _fork_forward_rng():
            _seed_forward_rng(case.forward_seed)
            cached_output, cache = bridge.run_with_cache(
                input_ids,
                accumulate=True,
                names_filter=metrics_names_filter,
                device="cpu",
                **bridge_kwargs,
            )
            cached_logits = _extract_logits(cached_output, case).detach().float().cpu()
        torch.testing.assert_close(
            cached_logits, bridge_logits, atol=case.atol, rtol=case.rtol
        )
        _assert_hook_iterations(cache, structure, case.recurrences)

    del bridge
    gc.collect()
    _empty_device_cache()

    assert hf_logits.shape == bridge_logits.shape, (
        f"Shape mismatch for {case.name}: "
        f"HF {hf_logits.shape} vs Bridge {bridge_logits.shape}"
    )
    assert torch.allclose(hf_logits, bridge_logits, atol=case.atol, rtol=case.rtol), (
        f"Logit mismatch for {case.name}: "
        f"max diff = {(hf_logits - bridge_logits).abs().max().item()}"
    )


# ``None`` loads the registered default, which must equal the checkpoint's
# mean training recurrence: 8 for Parcae 770M and 1 for its feedforward control.
PARCAE_CASES = [
    pytest.param("parcae_770m", 32, id="parcae_770m_32_steps"),
    pytest.param("parcae_770m", None, id="parcae_770m_default_steps"),
    pytest.param("parcae_770m_rec1", None, id="parcae_770m_rec1"),
]


@pytest.mark.parametrize(("name", "num_steps"), PARCAE_CASES)
def test_parcae_loader_matches_native_pretrained_logits(
    name: str, num_steps: int | None, monkeypatch
):
    checkpoint = MODEL_SPECS[name].source
    repo_id = checkpoint.weights
    config = json.loads(
        _load_or_skip(repo_id, lambda: _resolve(repo_id, "config.json")).read_text()
    )
    config.pop("_class_name")
    steps = config["mean_recurrence"] if num_steps is None else num_steps
    input_ids = INPUT_IDS.to(DEVICE)

    # The bridge always uses SDPA; use it here too so only the wrapping differs.
    monkeypatch.setattr(flash_attention, "_override_impl", "sdpa")
    native = ParcaeConfig(**config).construct_model()
    state = torch.load(
        _resolve(repo_id, "pytorch_model.bin"), map_location="cpu", weights_only=True
    )
    native.load_state_dict(state, strict=True)
    native.to(device=DEVICE, dtype=DTYPE).eval()
    with torch.no_grad(), _fork_forward_rng():
        # Parcae draws its initial recurrent state at random.
        _seed_forward_rng(FORWARD_SEED)
        pair = torch.tensor([steps, 0])
        native_output = native(input_ids, return_logits=True, num_steps_pair=pair)
        native_logits = native_output["logits"].float().cpu()

    del native, state, native_output
    gc.collect()
    _empty_device_cache()

    bridge, structure = load_model(
        name, device=DEVICE, dtype=DTYPE, num_steps=num_steps
    )
    with torch.no_grad():
        with _fork_forward_rng():
            _seed_forward_rng(FORWARD_SEED)
            bridge_logits = bridge(input_ids, return_type="logits").float().cpu()
        with _fork_forward_rng():
            _seed_forward_rng(FORWARD_SEED)
            cached_logits, cache = bridge.run_with_cache(
                input_ids,
                return_type="logits",
                accumulate=True,
                names_filter=metrics_names_filter,
                device="cpu",
            )
    torch.testing.assert_close(
        cached_logits.float().cpu(), bridge_logits, atol=1e-2, rtol=1e-2
    )
    _assert_hook_iterations(cache, structure, steps)

    del bridge, cache
    gc.collect()
    _empty_device_cache()

    torch.testing.assert_close(bridge_logits, native_logits, atol=1e-2, rtol=1e-2)
