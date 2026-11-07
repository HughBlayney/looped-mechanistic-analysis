"""Loader contracts and inference defaults, without checkpoint downloads."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from looped_internals.models import ModelSpec, load_model
from looped_internals.models import loading as models
from looped_internals.models.minimal import MinimalRecurrentConfig
from looped_internals.structure import LayerGroup


@pytest.fixture
def hf_loads(monkeypatch):
    calls = []

    class StubBridge(torch.nn.Module):
        tokenizer = models.TransformerBridge.tokenizer

    for name, spec in models.MODEL_SPECS.items():
        if spec.prepare_source is not None:
            monkeypatch.setitem(
                models.MODEL_SPECS,
                name,
                replace(spec, prepare_source=lambda name, **kwargs: None),
            )

    def boot(name, **kwargs):
        calls.append((name, kwargs))
        config = {
            "mean_recurrence": 7,
            "num_hidden_layers": 6,
            "n_layers_in_prelude": 1,
            "n_layers_in_recurrent_block": 2,
            "n_layers_in_coda": 1,
            **(kwargs["hf_config_overrides"] or {}),
        }
        bridge = StubBridge()
        bridge.original_model = torch.nn.Linear(1, 1)
        bridge.original_model.config = SimpleNamespace(**config)
        bridge.cfg = SimpleNamespace(n_layers=config["num_hidden_layers"])
        return bridge

    monkeypatch.setattr(models.TransformerBridge, "boot_transformers", boot)
    monkeypatch.setattr(
        models.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(
            encode=lambda text, add_special_tokens=True: [3, 4],
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=2,
        ),
    )
    return calls


@pytest.mark.parametrize(
    "name, repo, dtype, overrides, trusted",
    [
        ("gpt2", "gpt2", torch.float32, None, False),
        ("raven", "tomg-group-umd/huginn-0125", torch.bfloat16, None, True),
        (
            "ouro_2_6b",
            "ByteDance/Ouro-2.6B",
            torch.bfloat16,
            {"total_ut_steps": 4},
            True,
        ),
        (
            "retrofitted_llama_4",
            "smcleish/Recurrent-Llama-3.2-train-recurrence-4",
            torch.bfloat16,
            {"mean_recurrence": 32},
            True,
        ),
        ("llama_8b", "meta-llama/Meta-Llama-3.1-8B", torch.bfloat16, None, False),
        ("pythia_410m", "EleutherAI/pythia-410m-deduped", torch.bfloat16, None, False),
        ("gemma_7b", "google/gemma-7b", torch.bfloat16, None, False),
        (
            "olmo2_instruct",
            "allenai/OLMo-2-0425-1B-Instruct",
            torch.bfloat16,
            None,
            True,
        ),
    ],
)
def test_checkpoint_selection_and_existing_defaults(
    hf_loads, name, repo, dtype, overrides, trusted
):
    bridge, _ = load_model(name, device="cpu")
    assert hf_loads == [
        (
            repo,
            {
                "device": "cpu",
                "dtype": dtype,
                "hf_config_overrides": overrides,
                "trust_remote_code": trusted,
                "revision": None,
            },
        )
    ]
    assert not bridge.training
    assert not bridge.original_model.training
    assert bridge.tokenizer.encode("hello") == [3, 4]
    assert not bridge.cfg.default_prepend_bos
    assert not bridge.cfg.tokenizer_prepends_bos


@pytest.mark.parametrize(
    "name, step_key, groups",
    [
        (
            "raven",
            "mean_recurrence",
            [
                LayerGroup("prelude_blocks", 1, False),
                LayerGroup("core_blocks", 2, True),
                LayerGroup("coda_blocks", 1, False),
            ],
        ),
        (
            "retrofitted_olmo_16",
            "mean_recurrence",
            [
                LayerGroup("prelude_blocks", 1, False),
                LayerGroup("core_blocks", 2, True),
                LayerGroup("coda_blocks", 1, False),
            ],
        ),
        ("ouro_1_4b", "total_ut_steps", [LayerGroup("blocks", 6, True)]),
    ],
)
def test_uniform_steps_and_block_layouts(hf_loads, name, step_key, groups):
    bridge, structure = load_model(name, device="cpu", dtype=torch.float32, num_steps=5)
    assert getattr(bridge.original_model.config, step_key) == 5
    assert structure.groups == groups
    assert hf_loads[0][1]["dtype"] == torch.float32


def test_native_overrides_win_and_structure_uses_loaded_config(hf_loads):
    overrides = {"mean_recurrence": 3, "n_layers_in_recurrent_block": 5}
    bridge, structure = load_model(
        "raven", device="cpu", num_steps=8, config_overrides=overrides
    )
    assert bridge.original_model.config.mean_recurrence == 3
    assert structure.groups[1] == LayerGroup("core_blocks", 5, True)
    assert overrides == {"mean_recurrence": 3, "n_layers_in_recurrent_block": 5}
    assert hf_loads[0][1]["hf_config_overrides"] is not overrides


def test_standard_structure_uses_loaded_layer_count(hf_loads):
    _, structure = load_model(
        "gpt2", device="cpu", config_overrides={"num_hidden_layers": 3}
    )
    assert structure.groups == [LayerGroup("blocks", 3, False)]


def test_custom_checkpoint_spec(hf_loads):
    spec = ModelSpec(
        "local/custom-checkpoint",
        layout="recurrent",
        recurrence_parameter="total_ut_steps",
        default_steps=2,
        default_dtype=torch.float64,
    )
    bridge, structure = load_model(spec, device="cpu")
    assert hf_loads[0][0] == spec.source
    assert hf_loads[0][1]["dtype"] == torch.float64
    assert bridge.original_model.config.total_ut_steps == 2
    assert structure.groups == [LayerGroup("blocks", 6, True)]


@pytest.mark.parametrize("steps", [0, -1, 2.5, True, "3"])
def test_invalid_steps_fail_before_loading(hf_loads, steps):
    with pytest.raises(ValueError, match="positive integer"):
        load_model("raven", num_steps=steps)
    assert hf_loads == []


def test_non_recurrent_steps_are_rejected(hf_loads):
    with pytest.raises(ValueError, match="only to recurrent"):
        load_model("gpt2", num_steps=2)
    assert hf_loads == []


def test_unknown_name_is_actionable(hf_loads):
    with pytest.raises(ValueError, match=r"Unknown model.*ModelSpec"):
        load_model("unknown")
    assert hf_loads == []


def test_preserve_pythia_float16_restriction(hf_loads):
    with pytest.raises(ValueError, match="does not support dtype"):
        load_model("pythia_410m", dtype=torch.float16)
    assert hf_loads == []


@pytest.mark.parametrize("native_steps, expected", [(None, 3), (4, 4)])
@pytest.mark.filterwarnings("error:Hook alias:UserWarning")
def test_minimal_inference_steps_dtype_and_source_isolation(native_steps, expected):
    config = MinimalRecurrentConfig(
        d_model=16, n_heads=2, n_layers=2, d_vocab=32, num_cycles=2, expansion_factor=2
    )
    overrides = {} if native_steps is None else {"num_cycles": native_steps}
    bridge, structure = load_model(
        ModelSpec(config),
        device="cpu",
        dtype=torch.float64,
        num_steps=3,
        config_overrides=overrides,
    )
    with torch.no_grad():
        logits, cache = bridge.run_with_cache(
            torch.tensor([[1, 2, 3]]), accumulate=True
        )
    assert logits.shape == (1, 3, 32)
    assert torch.isfinite(logits).all()
    assert cache.get_iterations("blocks.0.hook_out").shape[0] == expected
    assert structure.groups == [LayerGroup("blocks", 2, True)]
    assert not bridge.original_model.training
    assert {p.dtype for p in bridge.parameters()} == {torch.float64}
    assert config.num_cycles == 2
    assert overrides == ({} if native_steps is None else {"num_cycles": native_steps})
    default_bridge, _ = load_model(ModelSpec(config), device="cpu")
    assert default_bridge.original_model.config.num_cycles == 2
    assert next(default_bridge.parameters()).dtype == torch.float32


@pytest.mark.parametrize(
    "name, weights, steps, expected_steps",
    [
        ("parcae_140m", "SandyResearch/parcae-140m", None, 8),
        ("parcae_1_3b", "SandyResearch/parcae-1.3b", 32, 32),
        (
            "parcae_140m_rec1",
            f"{models.PARCAE_MODELS_ORG}/parcae-140m-rec1",
            None,
            1,
        ),
    ],
)
def test_parcae_checkpoint_selection(monkeypatch, name, weights, steps, expected_steps):
    calls = []

    def load(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        bridge = torch.nn.Module()
        bridge.original_model = torch.nn.Linear(1, 1)
        return bridge, None

    monkeypatch.setattr(models, "load_parcae", load)
    load_model(name, device="cpu", num_steps=steps)
    assert calls == [
        (
            models.ParcaeCheckpoint(weights),
            {
                "device": "cpu",
                "dtype": torch.bfloat16,
                "num_steps": expected_steps,
                "config_overrides": {},
            },
        )
    ]


def test_registered_minimal_accepts_config_overrides():
    source = models.MODEL_SPECS["minimal"].source
    bridge, _ = load_model(
        "minimal",
        device="cpu",
        config_overrides={"d_model": 16, "n_heads": 2, "n_layers": 1, "d_vocab": 32},
    )
    assert bridge.cfg.d_model == 16
    assert source.d_model == 256


def test_custom_geometry_and_preparation_are_independent(hf_loads, monkeypatch):
    """A new layout and native step key need no Raven-specific assumptions."""
    from looped_internals.structure import ModelLayerStructure

    order = []
    boot = models.TransformerBridge.boot_transformers

    def construct(*args, **kwargs):
        assert order == ["source"]
        order.append("config-and-model")
        return boot(*args, **kwargs)

    monkeypatch.setattr(models.TransformerBridge, "boot_transformers", construct)
    layout = ModelLayerStructure([LayerGroup("custom_core", 3, True)])
    spec = ModelSpec(
        "local/another-family",
        layout=layout,
        recurrence_parameter="num_iterations",
        prepare_source=lambda _, **kwargs: order.append("source"),
    )
    _, structure = load_model(spec, device="cpu", num_steps=5)
    assert structure is layout
    assert order == ["source", "config-and-model"]
    assert hf_loads[0][1]["hf_config_overrides"] == {"num_iterations": 5}
