"""Scientific reductions and offline execution of the injection-radius sweep."""

import gc
import json
import math
import weakref
from pathlib import Path
from types import SimpleNamespace

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pytest
import test_match_parcae
import torch
import torch.nn.functional as F

from experiments import support
from experiments.cache import ExperimentCache
from looped_internals.models import ModelSpec, load_model
from looped_internals.structure import LayerGroup, ModelLayerStructure

parcae_checkpoint = test_match_parcae.checkpoint
NOTEBOOK = Path(__file__).parents[1] / "experiments/isolating_stability.ipynb"
CELLS = json.loads(NOTEBOOK.read_text())["cells"]


def execute(cell, namespace):
    exec(
        compile("".join(cell["source"]), f"{NOTEBOOK}:{cell['id']}", "exec"), namespace
    )


def setup(**parameters):
    namespace = {}
    for cell in CELLS:
        if cell["cell_type"] != "code":
            continue
        if cell["id"] == "collection":
            break
        execute(cell, namespace)
        if cell["id"] == "parameters":
            namespace.update(parameters)
    return namespace


def test_presets_keep_full_figure_order_and_128_step_reference():
    quick, full = setup(), setup(mode="full")
    assert quick["figure_arms"] == full["figure_arms"]
    assert quick["RHO_KEYS"] == ("0_25", "0_5", "0_75", "1_0", "learned")
    assert quick["SCHEDULES"] == ("poisson", "fixed")
    assert len(quick["arms"]) == 6 and len(full["arms"]) == 10
    for namespace, count, limit, ppl_limit in [(quick, 6, 1, 1), (full, 10, 256, 32)]:
        assert namespace["config"].limit == limit
        assert namespace["ppl_examples"] == ppl_limit
        assert namespace["analysis_pairs"] == [(arm, 128) for arm in namespace["arms"]]
        assert namespace["ppl_recurrences"] == [1, 2, 4, 8, 16, 32, 64]
        assert len(namespace["model_recurrences"]) == count * 8
    assert setup(mode="quick", limit=256)["config"].mode == "quick"


@pytest.mark.parametrize(
    "parameters",
    [
        {"arms": ["ouro_1_4b"]},
        {"arms": ["stab_fixed_0_25"] * 2},
        {"ppl_recurrences": []},
        {"ppl_recurrences": [1, 2]},
        {"ppl_limit": 0},
        {"analysis_recurrences": 8},
    ],
)
def test_invalid_parameters_fail_before_collection(parameters):
    with pytest.raises(ValueError):
        setup(**parameters)


def test_convergence_uses_each_layers_final_state_and_token_mean_distances():
    namespace = setup()
    structure = ModelLayerStructure(
        [
            LayerGroup("prelude", 1, False),
            LayerGroup("core", 2, True),
            LayerGroup("coda", 1, False),
        ]
    )
    layout = list(structure.virtual_layers(3))
    cache = {}
    # Opposite token directions catch norm-of-mean instead of mean-of-norm.
    for layer in layout:
        radius = (layer.iteration + 1) ** 2 if layer.iteration is not None else 1000
        scale = layer.layer_index + 1
        cache[layer.residual_key] = torch.tensor([[[radius, 0], [-radius, 0]]]) * scale
    actual = namespace["summarize_convergence"](cache, layout)
    np.testing.assert_allclose(actual["to_final"], [[16, 10, 0], [24, 15, 0]])
    np.testing.assert_allclose(actual["successive"], [[6, 10], [9, 15]])

    def example(scale):
        return {
            "layers": [1, 1, 1],
            "virtual_layers": [0, 1, 2],
            "recurrences": [0, 1, 2],
            "recurrent_layers": [1],
            "curves": {"residual_norms_mean": [scale] * 3},
            "convergence": {
                key: np.asarray(value) * scale for key, value in actual.items()
            },
        }

    aggregate = namespace["aggregate_examples"]([example(1), example(3)])
    np.testing.assert_allclose(aggregate["curves"]["residual_norms_mean"]["mean"], 2)
    np.testing.assert_allclose(
        aggregate["curves"]["residual_norms_mean"]["std"], np.sqrt(2)
    )
    np.testing.assert_allclose(
        aggregate["convergence"]["to_final"]["mean"], np.asarray(actual["to_final"]) * 2
    )
    single = namespace["aggregate_examples"]([example(1)])
    assert not single["convergence"]["successive"]["std"].any()


def test_attention_distance_uses_training_reference_and_averages_per_head_norms():
    namespace = setup()
    structure = ModelLayerStructure(
        [
            LayerGroup("prelude", 1, False),
            LayerGroup("core", 2, True),
            LayerGroup("coda", 1, False),
        ]
    )
    layout = list(structure.virtual_layers(3))
    cache = {}
    base = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    # Opposite changes across heads cancel if patterns are averaged before norms.
    direction = torch.tensor([[[0.0, 0.0], [1.0, -1.0]], [[0.0, 0.0], [-1.0, 1.0]]])
    for layer in layout:
        cache[layer.residual_key] = torch.zeros(1, 2, 2)
        if layer.iteration is not None:
            scale = 0.1 * layer.layer_index * (layer.iteration - 1)
            cache[layer.attn_key] = (base + scale * direction).unsqueeze(0)
    measured = namespace["summarize_convergence"](cache, layout, attention_reference=2)
    expected = np.sqrt(2) * np.array([[0.1, 0, 0.1], [0.2, 0, 0.2]])
    np.testing.assert_allclose(
        measured["attention_to_training_max"], expected, rtol=1e-6
    )
    assert "attention_to_training_max" not in namespace["summarize_convergence"](
        cache, layout
    )


def test_summary_averages_token_mean_norms_over_only_recurrent_layers():
    namespace = setup()
    examples = [
        {
            "recurrences": [None, 0, 0, 1, 1, None],
            "curves": {"residual_norms_mean": [1000, 1, 3, 5, 7, 1000]},
        },
        {
            "recurrences": [None, 0, 0, 1, 1, None],
            "curves": {"residual_norms_mean": [2000, 3, 5, 7, 9, 2000]},
        },
    ]
    steps, mean = namespace["recurrent_norm_by_recurrence"](examples)
    np.testing.assert_array_equal(steps, [1, 2])
    np.testing.assert_allclose(mean, [3, 7])


def test_summary_mixing_profiles_keep_only_extrapolation_and_actual_control_coda():
    namespace = setup()
    execute(next(c for c in CELLS if c["id"] == "drawing"), namespace)
    structure = ModelLayerStructure(
        [
            LayerGroup("prelude", 1, False),
            LayerGroup("core", 2, True),
            LayerGroup("coda", 1, False),
        ]
    )
    layout = list(structure.virtual_layers(18))
    values = np.linspace(0.1, 0.9, len(layout))
    data = {
        "layers": [layer.layer_index for layer in layout],
        "virtual_layers": [layer.virtual_layer_index for layer in layout],
        "recurrences": [layer.iteration for layer in layout],
        "recurrent_layers": [1, 2],
        "curves": {"mixing_scores": {"mean": values, "std": np.zeros(len(values))}},
    }
    control_layout = list(structure.virtual_layers(16))
    # The actual 16-step coda is maximally mixing; earlier layers attend only to self.
    cache = {layer.attn_key: torch.eye(2)[None, None] for layer in control_layout}
    cache[control_layout[-1].attn_key] = torch.tensor([[[[1.0, 0.0], [0.5, 0.5]]]])
    control = namespace["summarize_training_control"](cache, structure, 16)
    assert control["recurrences"] == [None, 15, 15, None]
    np.testing.assert_allclose(control["curves"]["mixing_scores"], [0, 0, 0, 1])
    name = "stab_poisson_0_25"
    namespace.update(
        stages={name: data}, controls={name: namespace["aggregate_examples"]([control])}
    )
    fig, ax = plt.subplots()
    namespace["draw_mixing_extrapolation"](ax, name)
    assert len(ax.lines) == 5  # Prelude/coda, two extrapolated recurrences, control.
    for line, recurrence in zip(ax.lines[2:4], (16, 17), strict=True):
        indices = [i for i, layer in enumerate(layout) if layer.iteration == recurrence]
        np.testing.assert_allclose(line.get_ydata(), values[indices])
    np.testing.assert_allclose(ax.lines[-1].get_ydata(), [0, 0, 0, 1])
    connectors = next(
        c for c in ax.collections if c.get_gid() == "prelude_coda_connectors"
    )
    assert len(connectors.get_segments()) == 1
    np.testing.assert_allclose(
        connectors.get_segments()[0],
        [ax.lines[3].get_xydata()[-1], ax.lines[1].get_xydata()[0]],
    )
    plt.close(fig)


@pytest.mark.parametrize("preexisting_analysis", [False, True])
def test_training_controls_share_ppl_forwards_and_preserve_analysis_cache(
    preexisting_analysis, parcae_checkpoint, tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    loads, forwards = [], []
    name = "stab_poisson_0_25"
    spec = ModelSpec(parcae_checkpoint, default_steps=4)
    monkeypatch.setattr(
        ExperimentCache, "model", lambda self, name: (spec, {"source": "tiny"})
    )

    def model(name, config, *, steps=None, cache=None):
        loads.append(name)
        bridge, structure = load_model(spec, device="cpu", dtype=torch.float32)
        bridge.original_model.parcae.register_forward_hook(
            lambda module, args, kwargs, result: forwards.append(
                int(kwargs["num_steps_pair"].sum())
            ),
            with_kwargs=True,
        )
        return bridge, structure

    monkeypatch.setattr(support, "load_experiment_model", model)
    namespace = setup(
        arms=[name],
        analysis_recurrences=4,
        max_training_recurrence=2,
        training_recurrence=2,
        ppl_recurrences=[1, 2, 4],
        limit=2,
        ppl_limit=1,
        device="cpu",
        dtype="float32",
        output_dir=str(tmp_path / "runs"),
        cache_dir=str(tmp_path / "cache"),
        cache_mode="reuse",
    )
    task = SimpleNamespace(
        eval_docs=["w3 w4 ", "w5 w6 "],
        doc_to_text=lambda doc: doc,
        doc_to_target=lambda doc: "w9",
    )
    namespace["get_task_manager"] = lambda: SimpleNamespace(
        load=lambda names: {"tasks": {n: task for n in names}}
    )
    cell = next(c for c in CELLS if c["id"] == "collection")
    if preexisting_analysis:
        namespace["SUMMARY_ARMS"] = ()
        execute(cell, namespace)
        assert forwards == [1, 2, 4, 4]
        artifact = next(
            (tmp_path / "cache" / "isolating_stability").glob("*/result.pt")
        )
        original = artifact.read_bytes()
        namespace["SUMMARY_ARMS"] = (name,)
        forwards.clear()
    execute(cell, namespace)
    assert forwards == ([2, 2] if preexisting_analysis else [1, 2, 2, 4, 4])
    assert len(loads) == (2 if preexisting_analysis else 1)
    if preexisting_analysis:
        assert artifact.read_bytes() == original
    assert namespace["training_controls"][name]["run"]["num_steps"] == 2
    assert len(namespace["training_controls"][name]["summaries"]) == 2
    # With both artifacts populated, require mode performs no model loading.
    forwards.clear()
    namespace["result_cache"].mode = "require"
    execute(cell, namespace)
    assert not forwards
    assert len(loads) == (2 if preexisting_analysis else 1)


@pytest.mark.parametrize("schedule, training_limit", [("poisson", 16), ("fixed", 8)])
def test_profile_rows_split_training_range_and_keep_only_adjacent_boundaries(
    schedule, training_limit
):
    namespace = setup()
    execute(next(cell for cell in CELLS if cell["id"] == "drawing"), namespace)
    structure = ModelLayerStructure(
        [
            LayerGroup("prelude", 1, False),
            LayerGroup("core", 2, True),
            LayerGroup("coda", 1, False),
        ]
    )
    layout = list(structure.virtual_layers(128))
    values = np.arange(1, len(layout) + 1, dtype=float)
    data = {
        "layers": [layer.layer_index for layer in layout],
        "virtual_layers": [layer.virtual_layer_index for layer in layout],
        "recurrences": [layer.iteration for layer in layout],
        "recurrent_layers": [1, 2],
        "curves": {
            field: {"mean": values, "std": np.zeros(len(layout))}
            for field in namespace["STAGE_METRICS"]
        },
    }
    name = f"stab_{schedule}_0_25"
    namespace["stages"] = {name: data}
    for field in namespace["STAGE_METRICS"]:
        fig, axes = plt.subplots(2, 1)
        for ax, within_training in zip(axes, (True, False), strict=True):
            namespace["draw_stage_profiles"](
                ax, name, field, within_training=within_training
            )
            assert all(line.get_linestyle() == "-" for line in ax.lines)
            labelled = {
                line.get_label()
                for line in ax.lines
                if not line.get_label().startswith("_")
            }
            assert labelled == ({"Prelude"} if within_training else {"Coda"})
            recurrences = (
                range(1, training_limit + 1)
                if within_training
                else range(training_limit + 1, 129)
            )
            recurrent = ax.lines[1:] if within_training else ax.lines[:-1]
            assert len(recurrent) == len(recurrences)
            for line, recurrence in zip(recurrent, recurrences, strict=True):
                indices = [
                    i
                    for i, layer in enumerate(layout)
                    if layer.iteration == recurrence - 1
                ]
                np.testing.assert_array_equal(line.get_ydata(), values[indices])
                # Colours remain on one absolute scale across both rows.
                expected = plt.colormaps[namespace["COLORMAPS"]["recurrences"]](
                    (recurrence - 1) / 127
                )
                np.testing.assert_allclose(line.get_color(), expected)
            connectors = next(
                c for c in ax.collections if c.get_gid() == "prelude_coda_connectors"
            )
            assert len(connectors.get_segments()) == 1
            before, after = (
                (ax.lines[0], recurrent[0])
                if within_training
                else (recurrent[-1], ax.lines[-1])
            )
            np.testing.assert_allclose(
                connectors.get_segments()[0],
                [before.get_xydata()[-1], after.get_xydata()[0]],
            )
        plt.close(fig)


@pytest.mark.parametrize("mode, dataset", [("quick", "gsm8k"), ("full", "hellaswag")])
def test_notebook_collects_once_and_exports_all_figures(
    mode, dataset, parcae_checkpoint, tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    paper_root = tmp_path / "paper"
    monkeypatch.setattr(support, "PAPER_PLOTS_DIR", paper_root)
    monkeypatch.setattr(plt, "show", lambda: None)

    def load_dataset(path, *args, **kwargs):
        if dataset == "hellaswag":
            assert path == "Rowan/hellaswag"
            docs = datasets.Dataset.from_list(
                [
                    {
                        "activity_label": "Activity",
                        "ctx_a": text,
                        "ctx_b": "they",
                        "endings": ["w10", "w11", "w9", "w12"],
                        "label": "2",
                    }
                    for text in ["w3 w4", "w5 w6 w7 w8"]
                ]
            )
            return datasets.DatasetDict({"train": docs, "validation": docs})
        assert path == "openai/gsm8k"
        docs = datasets.Dataset.from_list(
            [{"question": text, "answer": "w9"} for text in ["w3 w4", "w5 w6 w7 w8"]]
        )
        return datasets.DatasetDict({"train": docs, "test": docs})

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    loads, forwards, references = [], [], []

    def model(name, config, *, steps=None):
        loads.append((name, steps))
        bridge, structure = load_model(
            ModelSpec(parcae_checkpoint, default_steps=steps),
            device="cpu",
            dtype=torch.float32,
        )
        references.append(weakref.ref(bridge))

        def record_forward(module, args, kwargs, result):
            ids = args[0]
            logits = result["logits"]
            forwards.append(
                {
                    "name": name,
                    "steps": int(kwargs["num_steps_pair"].sum()),
                    "ids": ids.clone(),
                    "nll": F.cross_entropy(
                        logits[0, :-1].float(), ids[0, 1:], reduction="sum"
                    ).item(),
                }
            )

        bridge.original_model.parcae.register_forward_hook(
            record_forward, with_kwargs=True
        )
        return bridge, structure

    monkeypatch.setattr(support, "load_experiment_model", model)
    figures = {}
    save = support.save_figure

    def record_figure(fig, directory, name, **kwargs):
        figures[name] = fig
        save(fig, directory, name, **kwargs)

    monkeypatch.setattr(support, "save_figure", record_figure)
    arms = ["stab_poisson_0_25", "stab_fixed_learned"]
    namespace = {}
    for cell in CELLS:
        if cell["cell_type"] != "code":
            continue
        execute(cell, namespace)
        if cell["id"] == "parameters":
            namespace.update(
                mode=mode,
                dataset=dataset,
                arms=arms,
                analysis_recurrences=4,
                ppl_recurrences=[1, 2, 4],
                training_recurrence=2,
                max_training_recurrence=4,
                limit=2,
                ppl_limit=1,
                device="cpu",
                dtype="float32",
                output_dir=str(tmp_path),
            )

    assert loads == [(arm, 4) for arm in arms]
    # Two short PPL passes plus two shared analysis/PPL passes per arm.
    assert len(forwards) == 8
    gc.collect()
    assert all(ref() is None for ref in references)
    directory = namespace["run_dir"]
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["dataset"] == dataset and directory.parent.name == dataset
    assert metadata["mode"] == mode and metadata["limit"] == 2
    assert directory.name.startswith(f"{mode}-")
    assert metadata["analysis_pairs"] == [list(pair) for pair in loads]
    overview = torch.load(directory / "overview.pt", weights_only=False)
    assert set(overview["summaries"]) == {name for name, _ in loads}
    for name, steps in loads:
        assert (directory / "models" / f"{name}.pt").exists()
        examples = overview["summaries"][name]
        assert len(examples) == 2
        assert all(len(example["layers"]) == 2 + 2 * steps for example in examples)
        for example in examples:
            assert all(
                np.isfinite(values).all() for values in example["curves"].values()
            )
            if name in arms:
                assert np.shape(example["convergence"]["successive"]) == (2, 3)
                np.testing.assert_array_equal(
                    np.asarray(example["convergence"]["to_final"])[:, -1], 0
                )
            if name.startswith("stab_poisson_"):
                distances = np.asarray(
                    example["convergence"]["attention_to_training_max"]
                )
                assert distances.shape == (2, 4) and np.isfinite(distances).all()
                np.testing.assert_array_equal(distances[:, 3], 0)
                assert np.any(distances[:, 1] > 0)
            else:
                assert "attention_to_training_max" not in example["convergence"]
        info = overview["run"][name]
        assert info["attention_reference_recurrence"] == (
            4 if name.startswith("stab_poisson_") else None
        )
        for recurrence, totals in info["nll_totals"].items():
            input_length = next(
                item["length"]
                for item in info["inputs"]
                if item["dataset_index"] == info["ppl_indices"][0]
            )
            forward = next(
                run
                for run in forwards
                if run["name"] == name
                and run["steps"] == recurrence
                and run["ids"].numel() == input_length
            )
            assert totals == {
                "total_nll": forward["nll"],
                "total_tokens": input_length - 1,
            }
            assert overview["perplexity"][name][recurrence] == pytest.approx(
                math.exp(forward["nll"] / (input_length - 1))
            )

    assert len(figures) == len(list(directory.rglob("*.pdf"))) == 28
    assert len(list(paper_root.rglob("*.pdf"))) == (0 if mode == "quick" else 28)
    if mode == "full":
        assert (
            len(list((paper_root / "isolating_stability" / dataset).rglob("*.pdf")))
            == 28
        )
    for name, fig in figures.items():
        half_width = name in {
            "perplexity_degradation",
            "stability_summary_scaling",
            "stability_summary_mixing",
        }
        assert fig.get_figwidth() == pytest.approx(
            support.get_figsize(not half_width, (1, 1))[0]
        )
    convergence = figures["convergence/successive_poisson"]
    assert len(convergence.axes) == 6  # Five rho slots plus the shared colourbar.
    assert [ax.get_title() for ax in convergence.axes[:5]] == [
        namespace["rho_label"](rho) for rho in namespace["RHOS"]
    ]
    for ax in convergence.axes[1:5]:
        assert [text.get_text() for text in ax.texts] == ["skipped"]
    np.testing.assert_array_equal(convergence.axes[0].lines[0].get_xdata(), [2, 3, 4])
    distances = figures["convergence/attention_to_training_max_poisson"]
    assert distances.axes[0].get_gridspec().get_geometry() == (2, 3)
    assert len(distances.axes) == 6
    marker, *layer_lines = distances.axes[0].lines
    np.testing.assert_array_equal(marker.get_xdata(), [4, 4])
    assert len(layer_lines) == 2
    for line in layer_lines:
        np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3, 4])
        assert line.get_ydata()[3] == 0
    for ax in distances.axes[1:5]:
        assert [text.get_text() for text in ax.texts] == ["skipped"]
    for metric in ("to_final", "successive"):
        fixed = figures[f"convergence/{metric}_fixed"]
        assert fixed.axes[0].get_gridspec().get_geometry() == (3, 2)
        assert [bool(ax.get_ylabel()) for ax in fixed.axes[:5]] == [True, False] * 2 + [
            True
        ]
        assert (
            fixed.axes[0].get_position().width
            > convergence.axes[0].get_position().width
        )

    attention = figures["recurrent/attention_poisson_rho_0_25"]
    assert [text.get_text() for text in attention.legends[0].texts] == [
        "Prelude",
        "Coda",
    ]
    assert attention.axes[0].get_gridspec().get_geometry() == (2, 3)
    assert attention.axes[0].texts[0].get_text() == "Within train distribution"
    assert attention.axes[3].texts[-1].get_text() == "Out of train distribution"
    # This tiny run ends at the Poisson training cap; the second row has no recurrences.
    assert all(not ax.lines for ax in attention.axes[3:6])
    for part, rhos in [(1, namespace["RHOS"][:3]), (2, namespace["RHOS"][3:])]:
        residual = figures[f"recurrent/residual_norms_mean_poisson_part_{part}"]
        assert residual.axes[0].get_gridspec().get_geometry() == (2, len(rhos))
        assert [ax.get_title() for ax in residual.axes[: 2 * len(rhos)]] == [
            namespace["rho_label"](rho) for rho in rhos
        ] * 2
    for ax in attention.axes[:3]:
        connectors = next(
            collection
            for collection in ax.collections
            if collection.get_gid() == "prelude_coda_connectors"
        )
        assert len(connectors.get_segments()) == 2
    growth = figures["residual_norm_growth"]
    assert len(growth.axes) == 2 and all(len(ax.lines) == 1 for ax in growth.axes)
    assert all(ax.get_yscale() == "log" for ax in growth.axes)
    assert growth.axes[0].get_gridspec().get_geometry() == (1, 2)
    assert all(ax.get_xlabel() == "Virtual layer depth (%)" for ax in growth.axes)

    # Original PPL axes, clipped training distribution, and long/train ratio.
    ppl = figures["perplexity_vs_recurrence"]
    assert [line.get_linestyle() for line in ppl.legends[0].get_lines()] == ["--", "-"]
    assert all(
        ax.get_xscale() == "linear" and ax.get_yscale() == "log" for ax in ppl.axes[:2]
    )
    _, probabilities = namespace["clipped_poisson_distribution"](8, 16)
    assert probabilities[0] == pytest.approx(9 * math.exp(-8))
    assert probabilities[-1] == pytest.approx(
        1 - sum(math.exp(-8) * 8**k / math.factorial(k) for k in range(16))
    )
    assert probabilities.sum() == pytest.approx(1)
    bars = figures["perplexity_degradation"].axes[0].patches
    for bar, arm in zip(bars, arms, strict=True):
        assert bar.get_height() == pytest.approx(
            overview["perplexity"][arm][4] / overview["perplexity"][arm][2]
        )
    np.testing.assert_allclose(namespace["rho_positions"](list(reversed(arms))), [4, 0])

    scaling = figures["stability_summary_scaling"]
    mixing = figures["stability_summary_mixing"]
    for fig in (scaling, mixing):
        assert fig.axes[0].get_gridspec().get_geometry() == (1, 2)
        np.testing.assert_allclose(
            fig.get_size_inches(), support.get_figsize(False, (1, 2), 1.5)
        )
        assert len(fig.axes) == 3
    assert scaling.axes[0].get_yscale() == "log"
    assert scaling.axes[1].get_yscale() == "linear"
    np.testing.assert_allclose(scaling.axes[2].get_ylim(), [0, 0.2])
    assert [text.get_text() for text in scaling.legends[0].texts] == [
        namespace["rho_label"]("0.25")
    ]
    assert [text.get_text() for text in mixing.legends[0].texts] == [
        "Prelude",
        "Coda",
        "Control",
    ]
    assert mixing.axes[0].get_shared_x_axes().joined(*mixing.axes[:2])
    assert mixing.axes[0].get_shared_y_axes().joined(*mixing.axes[:2])
    np.testing.assert_allclose(mixing.axes[0].get_xlim(), mixing.axes[1].get_xlim())
    np.testing.assert_allclose(mixing.axes[0].get_ylim(), mixing.axes[1].get_ylim())
    x, mean = namespace["recurrent_norm_by_recurrence"](overview["summaries"][arms[0]])
    np.testing.assert_allclose(scaling.axes[0].lines[0].get_xdata(), x)
    np.testing.assert_allclose(scaling.axes[0].lines[0].get_ydata(), mean)
    # The fixed-trained arm must not fill the missing Poisson summary panel.
    assert [text.get_text() for text in mixing.axes[1].texts] == ["skipped"]
    assert not mixing.axes[1].lines
    control = overview["training_controls"][arms[0]]
    assert control["run"]["num_steps"] == 4
    assert control["run"]["indices"] == overview["run"][arms[0]]["analysis_indices"]
    control_line = next(
        line for line in mixing.axes[0].lines if line.get_label() == "Control"
    )
    expected_control = np.array(
        [example["curves"]["mixing_scores"] for example in control["summaries"]]
    ).mean(axis=0)
    np.testing.assert_allclose(control_line.get_ydata(), expected_control)
    assert control_line.get_linestyle() == "--" and control_line.get_color() == "red"
    assert len(scaling.axes[1].lines) == 1
    np.testing.assert_allclose(
        scaling.axes[1].lines[0].get_ydata(),
        [overview["perplexity"][arms[0]][t] for t in (1, 2, 4)],
    )
    _, probability = namespace["clipped_poisson_distribution"](2, 4)
    np.testing.assert_allclose(
        [bar.get_height() for bar in scaling.axes[2].patches], probability
    )

    # Every figure can be redrawn after releasing all models.
    namespace["save_figure"] = lambda fig, *args, **kwargs: plt.close(fig)
    started = False
    for cell in CELLS:
        if cell["id"] == "convergence-to_final-poisson":
            started = True
        if started and cell["cell_type"] == "code" and cell["id"] != "export":
            execute(cell, namespace)
    assert len(forwards) == 8
