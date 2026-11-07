"""Weight-only notebook checks using tiny local models, without inference."""

import gc
import json
import weakref
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from test_raven_adapter import FakeRetrofittedModel
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.supported_architectures.gpt2 import (
    GPT2ArchitectureAdapter,
)
from transformers import GPT2Config, GPT2LMHeadModel

from experiments import paper_models, support, weight_support
from looped_internals.analysis.neurons import (
    classify_neurons,
    compare_neuron_vocab_vectors,
    neuron_vocab_cosine_moments,
    prediction_suppression_fractions,
)
from looped_internals.models import MinimalRecurrentConfig, ModelSpec, load_model
from looped_internals.models.adapters.raven import RavenArchitectureAdapter
from looped_internals.structure import LayerGroup, ModelLayerStructure

NOTEBOOK = Path(__file__).parents[1] / "experiments" / "neuron_detection.ipynb"


def _cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def _setup(**parameters):
    namespace = {}
    for cell in _cells():
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            assert namespace["mode"] == "quick"
            assert not {"dataset", "limit", "num_steps", "seed"} & namespace.keys()
            namespace.update(parameters)
        if "summarize_neurons" in namespace:
            return namespace
    raise AssertionError("Missing notebook reduction")


def _forbid_forward(*args, **kwargs):
    pytest.fail("Weight analysis must not run a forward pass or collect inputs")


def _tiny_model(kind):
    """Exercise Conv1D, linear, and sandwich layouts with known neuron classes."""
    cfg = TransformerBridgeConfig(
        d_model=8, d_head=4, n_heads=2, n_layers=4, n_ctx=32, d_vocab=64
    )
    if kind == "gpt2":
        model = GPT2LMHeadModel(
            GPT2Config(
                n_embd=8,
                n_head=2,
                n_layer=4,
                n_inner=16,
                vocab_size=64,
                n_positions=32,
            )
        )
        output_weights = [block.mlp.c_proj.weight for block in model.transformer.h]
        adapter = GPT2ArchitectureAdapter(cfg)
        structure = ModelLayerStructure([LayerGroup("blocks", 4, False)])
    elif kind == "raven":
        model = FakeRetrofittedModel(
            d_model=8, d_vocab=64, n_prelude=1, n_core=2, n_coda=1
        )
        output_weights = [
            block.mlp.proj.weight.T
            for group in (
                model.transformer.prelude,
                model.transformer.core_block,
                model.transformer.coda,
            )
            for block in group
        ]
        adapter = RavenArchitectureAdapter(cfg, n_norms=2)
        structure = ModelLayerStructure(
            [
                LayerGroup("prelude_blocks", 1, False),
                LayerGroup("core_blocks", 2, True),
                LayerGroup("coda_blocks", 1, False),
            ]
        )
    else:
        bridge, structure = load_model(
            ModelSpec(
                MinimalRecurrentConfig(
                    d_model=8,
                    n_heads=2,
                    n_layers=4,
                    d_vocab=64,
                    expansion_factor=2,
                    num_cycles=13,
                )
            ),
            device="cpu",
        )
        model = bridge.original_model
        output_weights = [
            block.mlp.down_proj.weight.T for block in model.transformer_layers
        ]

    with torch.no_grad():
        # A single vocabulary outlier makes +/- e0 prediction/suppression neurons.
        model.lm_head.weight.zero_()
        model.lm_head.weight[:, 1] = 1
        model.lm_head.weight[0] = 0
        model.lm_head.weight[0, 0] = 1
        for layer, weights in enumerate(output_weights):
            weights.zero_()
            weights[: layer + 1, 0] = 2
            weights[layer + 1 : layer + 3, 0] = -3
    raw_out = torch.stack([w.detach().clone() for w in output_weights])
    raw_unembed = model.lm_head.weight.detach().T.clone()
    if kind != "minimal":
        bridge = TransformerBridge(model, adapter, tokenizer=None)
    model.config = SimpleNamespace(_commit_hash="tiny-local-revision")
    model.forward = _forbid_forward
    bridge.forward = _forbid_forward
    return bridge, structure, raw_out, raw_unembed


@pytest.mark.parametrize("kind", ["gpt2", "raven", "minimal"])
@pytest.mark.parametrize("chunk_size", [1, 7, 128])
def test_weight_reduction_matches_dense_comparison(kind, chunk_size):
    torch.set_num_threads(1)
    namespace = _setup(device="cpu", neuron_chunk_size=chunk_size)
    bridge, structure, weights, unembed = _tiny_model(kind)
    dense = compare_neuron_vocab_vectors(weights, unembed).cosine_sim
    moments = neuron_vocab_cosine_moments(dense)
    labels = classify_neurons(moments["kurtosis"], moments["skew"])
    expected = prediction_suppression_fractions(labels)
    chunks = []

    def record_moments(cosine):
        chunks.append(tuple(cosine.shape))
        return neuron_vocab_cosine_moments(cosine)

    namespace["neuron_vocab_cosine_moments"] = record_moments
    with torch.inference_mode():
        summary = namespace["summarize_neurons"](bridge, structure)
    np.testing.assert_allclose(summary["prediction_fraction"], expected[0])
    np.testing.assert_allclose(summary["suppression_fraction"], expected[1])
    assert summary["prediction_count"] == [1, 2, 3, 4]
    assert summary["suppression_count"] == [2, 2, 2, 2]
    assert summary["neurons_per_layer"] == [16] * 4
    assert all(len(shape) == 2 and shape[0] <= chunk_size for shape in chunks)
    assert sum(shape[0] for shape in chunks) == 4 * 16


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_notebook_loads_once_releases_models_and_exports(mode, tmp_path, monkeypatch):
    paper_root = tmp_path / "paper"
    destination = paper_root / "neuron_detection"
    destination.mkdir(parents=True)
    for name in ("feedforward", "looped"):
        (destination / f"{name}.pdf").write_bytes(b"previous paper figure")
    monkeypatch.setattr(support, "PAPER_PLOTS_DIR", paper_root)
    monkeypatch.setattr(support, "collect_metrics", _forbid_forward)
    monkeypatch.setattr(plt, "show", lambda: None)
    loads, references, figures = [], [], {}

    def tiny_loader(name, *, device, dtype):
        assert device == "cpu" and dtype == torch.float32
        assert all(reference() is None for reference in references)
        loads.append(name)
        kind = (
            "raven"
            if name == "raven" or name.startswith(("retrofitted", "parcae"))
            else "gpt2"
        )
        bridge, structure, _, _ = _tiny_model(kind)
        references.extend([weakref.ref(bridge), weakref.ref(bridge.original_model)])
        return bridge, structure

    save = support.save_figure

    def record_figure(fig, directory, name, **kwargs):
        figures[name] = fig
        save(fig, directory, name, **kwargs)

    monkeypatch.setattr(weight_support, "load_model", tiny_loader)
    monkeypatch.setattr(support, "save_figure", record_figure)
    namespace = {}
    for index, cell in enumerate(_cells()):
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), f"{NOTEBOOK}:{index}", "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(mode=mode, device="cpu", output_dir=str(tmp_path))
    gc.collect()
    assert all(reference() is None for reference in references)
    models = list(namespace["config"].models)
    assert loads == [support.MODEL_CHECKPOINTS.get(name, name) for name in models]
    assert len(loads) == len(set(loads)) == 15
    assert namespace["looped_models"] == list(paper_models.PAPER_MODEL_GROUPS["all"])
    assert namespace["feedforward_models"] == [
        "gpt2",
        "llama_1b",
        "tinyllama",
        "olmo2",
        "parcae_770m_rec1",
        "parcae_140m_rec1",
        "parcae_370m_rec1",
    ]
    directory = namespace["run_dir"]
    assert directory.parent == tmp_path / "neuron_detection"
    assert directory.name.startswith(f"{mode}-")
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["mode"] == mode and metadata["models"] == models
    assert metadata["looped_models"] == namespace["looped_models"]
    assert metadata["feedforward_models"] == namespace["feedforward_models"]
    assert metadata["kurtosis_threshold"] == 10
    assert not {"dataset", "task", "limit", "num_steps", "seed"} & metadata.keys()
    overview = torch.load(directory / "overview.pt", weights_only=True)
    assert set(overview["summaries"]) == set(overview["run"]) == set(models)
    for name, info in overview["run"].items():
        checkpoint = support.MODEL_CHECKPOINTS.get(name, name)
        source = weight_support.MODEL_SPECS[checkpoint].source
        assert info["checkpoint"] == (
            asdict(source) if is_dataclass(source) else source
        )
        assert info["revision"] == "tiny-local-revision"
        assert (info["d_model"], info["d_vocab"]) == (8, 64)
        assert len(overview["summaries"][name]["prediction_fraction"]) == 4
    assert len(figures["feedforward"].axes) == 7
    assert len(figures["looped"].axes) == 8
    assert figures["looped"].axes[3].get_title() == "Parcae 770M"
    assert figures["looped"].axes[7].get_title() == "Parcae 370M"
    # Metadata places boundaries halfway between physical layers 0/1 and 2/3.
    raven_axes = figures["looped"].axes[1]
    np.testing.assert_allclose(
        [line.get_xdata()[0] for line in raven_axes.lines[2:]],
        [100 / 6, 500 / 6],
    )
    for name, fig in figures.items():
        np.testing.assert_allclose(
            fig.get_size_inches(), support.get_figsize(True, (2, 4), 1.8)
        )
        assert [i for i, ax in enumerate(fig.axes) if ax.get_ylabel()] == [0, 4]
        pdf = directory / f"{name}.pdf"
        assert pdf.read_bytes().startswith(b"%PDF")
        expected = b"previous paper figure" if mode == "quick" else pdf.read_bytes()
        assert (destination / pdf.name).read_bytes() == expected
    assert not list(directory.glob("*.png"))

    # Figure reruns must work with both model loading and reduction forbidden.
    namespace["collect_weight_summary"] = _forbid_forward
    namespace["summarize_neurons"] = _forbid_forward
    for cell in _cells():
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith("plot_neuron_fractions("):
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    assert len(loads) == 15
    # Editing mode after collection cannot promote a quick run to paper output.
    namespace["mode"] = "full" if mode == "quick" else "quick"
    exec(compile("".join(_cells()[-1]["source"]), str(NOTEBOOK), "exec"), namespace)
    assert (namespace["paper_dir"] is None) == (mode == "quick")


def test_weight_modes_keep_the_same_models_and_no_sampling_settings():
    quick = _setup(device="cpu")["config"]
    full = _setup(device="cpu", mode="full")["config"]
    assert asdict(quick) | {"mode": "full"} == asdict(full)


def test_weight_selection_tracks_shared_parcae_switch(monkeypatch):
    models = list(paper_models.PAPER_MODEL_GROUPS["all"])
    models[3] = "parcae_770m"
    monkeypatch.setitem(paper_models.PAPER_MODEL_GROUPS, "all", tuple(models))
    namespace = _setup(device="cpu")
    assert namespace["looped_models"] == models
    assert namespace["feedforward_models"][-3:] == [
        "parcae_770m_rec1",
        "parcae_140m_rec1",
        "parcae_370m_rec1",
    ]
    assert len(namespace["config"].models) == 15


def test_weight_selection_respects_parameter_overrides():
    namespace = _setup(device="cpu", looped_models=["parcae_140m"])
    assert namespace["looped_models"] == ["parcae_140m"]
    assert namespace["feedforward_models"][-1:] == ["parcae_140m_rec1"]
    assert len(namespace["config"].models) == 6
    namespace = _setup(
        device="cpu", looped_models=["parcae_140m"], feedforward_models=["gpt2"]
    )
    assert namespace["config"].models == ("gpt2", "parcae_140m")
