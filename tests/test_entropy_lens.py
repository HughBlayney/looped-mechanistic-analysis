"""Entropy reductions and notebook exports using tiny, local model forwards."""

import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_match_parcae
import torch
from test_coda_lens import tiny_raven
from test_notebooks import TinyTokenizer

from experiments import readout_support, support
from experiments.cache import ExperimentCache
from experiments.paper_models import PAPER_MODEL_GROUPS
from looped_internals.evaluation.logit_lens import logit_lens
from looped_internals.models import MinimalRecurrentConfig, ModelSpec, load_model
from looped_internals.structure import LayerGroup, ModelLayerStructure

# Reuse the exported tiny Parcae checkpoint for native coda readouts.
parcae_checkpoint = test_match_parcae.checkpoint

NOTEBOOK = Path(__file__).parents[1] / "experiments" / "entropy_lens.ipynb"


def cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def setup(**parameters):
    namespace = {}
    for cell in cells():
        if cell["cell_type"] != "code":
            continue
        exec("".join(cell["source"]), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(parameters)
        if "configs" in namespace:
            return namespace
    raise AssertionError("Missing notebook configuration")


def test_shared_groups_select_controls_and_follow_the_parcae_switch(monkeypatch):
    namespace = setup()
    recurrent = [
        (name, 8 if name == "ouro_1_4b" else 32)
        for name in dict.fromkeys(PAPER_MODEL_GROUPS["all"])
    ]
    controls = list(
        dict.fromkeys((support.BASELINE_FOR[name], 1) for name, _ in recurrent)
    )
    assert namespace["model_recurrences"] == recurrent + controls
    assert namespace["coda_pairs"] == recurrent[1:] + [
        pair for pair in controls if pair[0].startswith("parcae_")
    ]
    assert namespace["figure_pairs"]["all"][3] == ("parcae_770m", 32)
    assert namespace["figure_pairs"]["all"][7] == ("parcae_370m", 32)
    main = (*PAPER_MODEL_GROUPS["main"][:3], "parcae_770m")
    monkeypatch.setitem(PAPER_MODEL_GROUPS, "main", main)
    monkeypatch.setitem(
        PAPER_MODEL_GROUPS, "all", main + PAPER_MODEL_GROUPS["appendix"]
    )
    namespace = setup(recurrences=4, ouro_recurrences=2)
    assert namespace["figure_pairs"]["main"][-1] == ("parcae_770m", 4)
    assert namespace["figure_pairs"]["main"][0] == ("ouro_1_4b", 2)
    assert namespace["figure_pairs"]["appendix"][-1] == ("parcae_370m", 4)
    assert ("parcae_770m_rec1", 1) in namespace["baseline_pairs"]
    assert ("parcae_770m", 4) in namespace["coda_pairs"]
    assert ("parcae_770m_rec1", 1) in namespace["coda_pairs"]


def test_projection_changes_only_parcae_cache_requests():
    namespace = setup(device="cpu")
    cache = ExperimentCache(mode="off")
    changed = []
    for pair, config in namespace["configs"].items():
        original = cache.key(
            config,
            pair[0],
            readout_options={
                "compute_coda": pair in namespace["recurrent_pairs"]
                and pair[0] != "ouro_1_4b",
                "token_chunk_size": namespace["token_chunk_size"],
            },
        )
        current = cache.key(
            config, pair[0], readout_options=namespace["readout_options"][pair]
        )
        if current != original:
            changed.append(pair)
    assert changed == namespace["projected_pairs"]
    assert len(changed) == 6


def science(chunk_size=2):
    from looped_internals.evaluation import coda_hidden_states

    namespace = dict(
        np=np,
        torch=torch,
        token_chunk_size=chunk_size,
        logit_lens=logit_lens,
        coda_hidden_states=coda_hidden_states,
    )
    cell = next(
        c for c in cells() if "def mean_vocabulary_entropy" in "".join(c["source"])
    )
    exec("".join(cell["source"]), namespace)
    return namespace


@pytest.mark.parametrize("chunk_size", [1, 3, 32])
def test_entropy_is_mean_of_token_entropies_with_full_vocabulary(chunk_size):
    readout = science(chunk_size)["mean_vocabulary_entropy"]
    bridge = SimpleNamespace(ln_final=lambda h: h, unembed=lambda h: h)
    # Disjoint concentrated predictions have low mean entropy, high pooled entropy.
    logits = torch.tensor([[[40.0, -40.0], [-40.0, 40.0], [0.0, 0.0]]])
    result = readout(bridge, logits)
    assert result["entropy_nats"] == pytest.approx(np.log(2) / 3)
    assert result["vocab_size"] == 2
    uniform = readout(bridge, torch.zeros(1, 5, 7))
    assert uniform["entropy_nats"] == pytest.approx(np.log(7))


def test_entropy_aggregation_weights_examples_and_preserves_coda_depth():
    aggregate = science()["aggregate_entropy"]
    layout = [{"layer_index": i, "iteration": 0 if i == 1 else None} for i in range(3)]
    examples = [
        [
            {
                "entropy_nats": x,
                "coda_entropy_nats": x if i < 2 else None,
                "vocab_size": 8,
            }
            for i in range(3)
        ]
        for x in (1.0, 3.0)
    ]
    result = aggregate(examples, {"virtual_layers": layout}, "coda_entropy_nats")
    np.testing.assert_allclose(result["mean"], [2, 2])
    np.testing.assert_allclose(result["std"], np.sqrt(2))
    assert result["depth_max"] == 2
    np.testing.assert_array_equal(result["layers"], [0, 1])
    np.testing.assert_array_equal(
        aggregate(examples[:1], {"virtual_layers": layout})["std"], 0
    )


def test_parcae_projects_prelude_and_core_once_and_preserves_final_prediction(
    parcae_checkpoint,
):
    torch.set_num_threads(1)
    bridge, structure = load_model(
        ModelSpec(parcae_checkpoint, default_steps=1),
        device="cpu",
        dtype=torch.float32,
    )
    model = bridge.original_model.parcae
    tokens = torch.tensor([[3, 4, 5, 6]])
    reduce = science()["summarize_readouts"]
    projection_calls = []
    with torch.inference_mode():
        output, cache = bridge.run_with_cache(
            tokens, accumulate=True, names_filter=structure.hook_names()
        )
        hook = model.transformer.C.register_forward_hook(
            lambda *args: projection_calls.append(1)
        )
        try:
            rows = []
            for layer in structure.virtual_layers(1):
                residual = cache[layer.residual_key]
                original = residual.clone()
                before_coda = layer.layer_index < 3
                expected_hidden = (
                    torch.nn.functional.linear(residual, model.transformer.C.weight)
                    if before_coda
                    else residual
                )
                expected_logits = model.lm_head(model.transformer.ln_f(expected_hidden))
                log_p = expected_logits.float().log_softmax(-1)
                before = len(projection_calls)
                result = reduce(
                    bridge,
                    residual,
                    layer,
                    structure,
                    input_ids=tokens,
                    compute_coda=True,
                    project_pre_coda=True,
                )
                assert len(projection_calls) - before == int(before_coda)
                torch.testing.assert_close(residual, original)
                assert result["entropy_nats"] == pytest.approx(
                    -(log_p.exp() * log_p).sum(-1).mean().item(), abs=1e-6
                )
                if before_coda:
                    # Independent native suffix: C was already applied above.
                    h = model.transformer.coda[0](
                        expected_hidden,
                        model.freqs_cis[:, : tokens.shape[1]],
                        None,
                        ve=model.value_embeds["3"](tokens),
                    )
                    log_p = (
                        model.lm_head(model.transformer.ln_f(h)).float().log_softmax(-1)
                    )
                    assert result["coda_entropy_nats"] == pytest.approx(
                        -(log_p.exp() * log_p).sum(-1).mean().item(), abs=1e-6
                    )
                else:
                    assert result["coda_entropy_nats"] is None
                rows.append(result)
        finally:
            hook.remove()
        log_p = output.float().log_softmax(-1)
        final_entropy = -(log_p.exp() * log_p).sum(-1).mean().item()
        assert rows[-1]["entropy_nats"] == pytest.approx(final_entropy, abs=1e-6)
        assert rows[-2]["coda_entropy_nats"] == pytest.approx(final_entropy, abs=1e-6)


@pytest.mark.parametrize(
    "dataset,mode,limit", [("gsm8k", "full", 1), ("hellaswag", "quick", 2)]
)
def test_entropy_notebook_runs_once_per_pair_and_reuses_figures(
    tmp_path, monkeypatch, parcae_checkpoint, dataset, mode, limit
):
    import datasets
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    docs = datasets.Dataset.from_list(
        [
            {
                "question": q,
                "answer": "w10",
                "activity_label": "Counting",
                "ctx_a": q,
                "ctx_b": "",
                "endings": ["w10", "w11"],
                "label": "0",
            }
            for q in ["w3 w4", "w5 w6", "w7 w8 w9"]
        ]
    )
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, **kw: datasets.DatasetDict(
            {s: docs for s in ["train", "test", "validation"]}
        ),
    )
    loads, forwards, refs, final_entropies = [], [], [], {}

    def model(name, config):
        assert all(ref() is None for ref in refs)
        loads.append((name, config.num_steps))
        if name.startswith("parcae_"):
            bridge, structure = load_model(
                ModelSpec(parcae_checkpoint, default_steps=config.num_steps),
                device="cpu",
                dtype=torch.float32,
            )
        elif name == "raven" or name.startswith("retrofitted_"):
            bridge = tiny_raven(config.num_steps)
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
                        d_model=16,
                        n_heads=2,
                        n_layers=2,
                        d_vocab=64,
                        num_cycles=config.num_steps,
                        initialization_variance=0,
                    )
                ),
                device="cpu",
            )
            if name in support.BASELINES:
                structure = ModelLayerStructure([LayerGroup("blocks", 2, False)])
        if not name.startswith("parcae_"):
            bridge.tokenizer = TinyTokenizer()
        refs.append(weakref.ref(bridge))
        final_entropies[name] = []

        def final_readout(module, args, output):
            forwards.append(name)
            logits = output if isinstance(output, torch.Tensor) else output.logits
            log_p = logits.float().log_softmax(-1)
            final_entropies[name].append(-(log_p.exp() * log_p).sum(-1).mean().item())

        bridge.original_model.register_forward_hook(final_readout)
        return bridge, structure

    monkeypatch.setattr(readout_support, "load_experiment_model", model)
    monkeypatch.setattr(plt, "show", lambda: None)
    paper_root = tmp_path / "paper"
    monkeypatch.setattr(support, "PAPER_PLOTS_DIR", paper_root)
    figures = {}
    original_save = support.save_figure

    def save(fig, directory, name, **kwargs):
        figures[name] = fig
        with plt.rc_context({"savefig.bbox": None}):
            fig.savefig(tmp_path / f"{name}.png", dpi=160)
        original_save(fig, directory, name, **kwargs)

    monkeypatch.setattr(support, "save_figure", save)
    namespace = {}
    for cell in cells():
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(
                dataset=dataset,
                mode=mode,
                limit=limit,
                device="cpu",
                output_dir=str(tmp_path),
                token_chunk_size=3,
            )

    expected_recurrent = [
        (name, 8 if name == "ouro_1_4b" else 32)
        for name in dict.fromkeys(PAPER_MODEL_GROUPS["all"])
    ]
    expected_controls = list(
        dict.fromkeys((support.BASELINE_FOR[name], 1) for name, _ in expected_recurrent)
    )
    expected_pairs = expected_recurrent + expected_controls
    assert loads == expected_pairs
    assert len(forwards) == len(loads) * limit
    assert all(ref() is None for ref in refs)
    output = namespace["run_dir"]
    assert output.name.startswith(mode + "-")
    expected_files = {
        "entropy_lens_main",
        "entropy_lens_appendix",
        "entropy_lens_normalized_main",
        "entropy_lens_normalized_appendix",
        "entropy_coda_lens_all",
        "entropy_coda_lens_normalized_all",
    }
    assert {p.stem for p in output.glob("*.pdf")} == expected_files
    assert paper_root.exists() == (mode == "full")
    saved = torch.load(output / "overview.pt", weights_only=False)
    assert list(saved["summaries"]) == expected_pairs
    indices = [info["indices"] for info in saved["run"].values()]
    assert all(i == indices[0] for i in indices)
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["mode"] == mode and metadata["limit"] == limit
    assert metadata["model_recurrences"] == [list(p) for p in expected_pairs]
    assert saved["figure_pairs"] == namespace["figure_pairs"]
    assert metadata["figure_pairs"] == {
        group: [list(pair) for pair in pairs]
        for group, pairs in namespace["figure_pairs"].items()
    }
    assert metadata["coda_pairs"] == [list(pair) for pair in namespace["coda_pairs"]]
    assert metadata["parcae_projection"] == "prelude_and_core"

    for pair, examples in saved["summaries"].items():
        name, steps = pair
        expected_depth = (
            2 * steps + 2
            if name == "raven" or name.startswith(("retrofitted_", "parcae_"))
            else 2 * steps
        )
        for i, example in enumerate(examples):
            assert len(example) == expected_depth
            assert example[-1]["entropy_nats"] == pytest.approx(
                final_entropies[name][i], abs=1e-5
            )
            if pair in namespace["coda_pairs"]:
                assert example[-1]["coda_entropy_nats"] is None
                assert example[-2]["coda_entropy_nats"] == pytest.approx(
                    final_entropies[name][i], abs=1e-5
                )
                assert all(v["coda_entropy_nats"] is not None for v in example[:-1])
            else:
                assert all(v["coda_entropy_nats"] is None for v in example)

    for filename, fig in figures.items():
        coda = "coda" in filename
        normalized = "normalized" in filename
        group = filename.rsplit("_", 1)[-1]
        pairs = namespace["figure_pairs"][group]
        axes = fig.axes[:-1]  # Last axis is the recurrence colourbar.
        assert len(axes) == (8 if coda else 4)
        assert [ax.get_title() for ax in axes] == [
            support.MODEL_LABELS[name]
            + (
                "*"
                if not coda and (name, steps) in namespace["projected_pairs"]
                else ""
            )
            for name, steps in pairs
        ]
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        assert [t.get_text() for t in legend.get_texts()] == (
            ["Prelude", "Control"] if coda else ["Prelude", "Coda", "Control"]
        )
        assert legend._ncols == (2 if coda else 3)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bounds = legend.get_window_extent(renderer)
        selected = namespace["coda_profiles" if coda else "profiles"]
        for ax, pair in zip(axes, pairs, strict=True):
            assert ax.get_legend() is None
            assert legend_bounds.y0 > ax.title.get_window_extent(renderer).y1
            if coda and pair[0] == "ouro_1_4b":
                assert not ax.lines and [t.get_text() for t in ax.texts] == ["No coda"]
                continue
            assert ax.lines[-1].get_color() == "red"
            control_pair = (support.BASELINE_FOR[pair[0]], 1)
            control = namespace[
                "coda_profiles"
                if coda and control_pair in namespace["coda_pairs"]
                else "profiles"
            ][control_pair]
            scale = np.log(control["vocab_size"]) if normalized else 1
            np.testing.assert_allclose(
                ax.lines[-1].get_ydata(), control["mean"] / scale
            )
            data = selected[pair]
            scale = np.log(data["vocab_size"]) if normalized else 1
            np.testing.assert_allclose(
                np.concatenate([line.get_ydata() for line in ax.lines[:-1]]),
                data["mean"] / scale,
            )
            if coda:
                assert all(line.get_label() != "Coda" for line in ax.lines)
                assert all(max(line.get_xdata()) < 100 for line in ax.lines[:-1])
            connectors = [
                c for c in ax.collections if c.get_gid() == "prelude_coda_connectors"
            ]
            if pair[0] == "ouro_1_4b":
                assert not connectors
            else:
                assert len(connectors) == 1
                segments = connectors[0].get_segments()
                assert len(segments) == (1 if coda else 2)
                np.testing.assert_allclose(
                    segments[0],
                    [ax.lines[0].get_xydata()[-1], ax.lines[1].get_xydata()[0]],
                )
                if not coda:
                    np.testing.assert_allclose(
                        segments[-1],
                        [ax.lines[-3].get_xydata()[-1], ax.lines[-2].get_xydata()[0]],
                    )
        if coda:
            for main, appendix in zip(axes[3].lines, axes[7].lines, strict=True):
                np.testing.assert_array_equal(main.get_ydata(), appendix.get_ydata())
        np.testing.assert_allclose(
            fig.get_size_inches(), support.get_figsize(True, (2 if coda else 1, 4), 1.9)
        )
    # Figure reruns never invoke collection, including the additional coda panels.
    for cell in cells():
        source = "".join(cell["source"])
        if source.startswith("plot_entropy_profiles("):
            exec(source, namespace)
    assert len(forwards) == len(loads) * limit
