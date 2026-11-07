"""Check multiple-choice sampling, training, depth coordinates, and paper figures."""

import csv
import importlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from experiments import support
from experiments.paper_models import PAPER_MODEL_GROUPS
from looped_internals import evaluation
from looped_internals.evaluation import probes as probe_module
from looped_internals.evaluation.datasets import MCQExample
from looped_internals.evaluation.logit_lens import logit_lens
from looped_internals.models import MinimalRecurrentConfig, ModelSpec, load_model
from looped_internals.results import DepthAccuracy, DepthEvaluationResult
from looped_internals.structure import LayerGroup, ModelLayerStructure

NOTEBOOK = Path(__file__).parents[1] / "experiments" / "multiple_choice.ipynb"


def cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def setup(**parameters):
    namespace = {}
    for cell in cells():
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(parameters)
        if cell["id"] == "depth-coordinates":
            return namespace
    raise AssertionError("Missing accuracy definitions")


def test_modes_keep_paper_pairs_and_separate_training_from_test_limits(
    tmp_path, caplog
):
    papermill = pytest.importorskip("papermill")
    prepared = papermill.execute_notebook(
        NOTEBOOK,
        tmp_path / "prepared.ipynb",
        parameters={"mode": "full"},
        prepare_only=True,
    )
    injected = next(
        c for c in prepared.cells if "injected-parameters" in c.metadata.get("tags", [])
    )
    parameters = {}
    exec(injected.source, parameters)
    quick, full = setup(), setup(**parameters)
    recurrent = [
        (name, 8 if name == "ouro_1_4b" else 32)
        for name in dict.fromkeys(PAPER_MODEL_GROUPS["all"])
    ]
    controls = list(
        dict.fromkeys((support.BASELINE_FOR[name], 1) for name, _ in recurrent)
    )
    expected = recurrent + controls
    assert list(quick["configs"]) == list(full["configs"]) == expected
    assert (quick["train_limit"], quick["config"].limit) == (8, 1)
    assert (full["train_limit"], full["config"].limit) == (5000, 256)
    assert all(config.mode == "full" for config in full["configs"].values())
    assert full["feedforward_pairs"] == controls
    assert (
        quick["coda_model_recurrences"]
        == full["coda_model_recurrences"]
        == recurrent[1:]
    )
    assert full["probe_learning_rate"] == 5e-4
    assert full["probe_batch_size"] == 4
    assert full["probe_batch_sizes"]["raven"] == 1
    assert full["probe_batch_sizes"]["retrofitted_llama"] == 4
    assert full["probe_batch_sizes"]["retrofitted_olmo"] == 8
    assert full["probe_batch_sizes"]["parcae_370m_rec1"] == 16
    assert "unknown parameter" not in caplog.text
    overridden = setup(mode="quick", limit=0, max_train=0)
    assert overridden["config"].limit is None and overridden["train_limit"] == 0
    assert overridden["config"].mode == "quick"
    overridden = setup(recurrences=4, ouro_recurrences=2)
    assert overridden["recurrent_pairs"][0] == ("ouro_1_4b", 2)
    assert all(steps == 4 for _, steps in overridden["recurrent_pairs"][1:])


def test_shared_parcae_switch_updates_collection_controls_and_all_figures(monkeypatch):
    main = (*PAPER_MODEL_GROUPS["main"][:3], "parcae_770m")
    monkeypatch.setitem(PAPER_MODEL_GROUPS, "main", main)
    monkeypatch.setitem(
        PAPER_MODEL_GROUPS, "all", main + PAPER_MODEL_GROUPS["appendix"]
    )
    namespace = setup()
    assert namespace["figure_pairs"]["main"][-1] == ("parcae_770m", 32)
    assert namespace["figure_pairs"]["all"][-1] == ("parcae_370m", 32)
    assert ("parcae_770m_rec1", 1) in namespace["feedforward_pairs"]
    assert ("parcae_770m", 32) in namespace["coda_model_recurrences"]
    assert namespace["PAPER_NAMES"]["parcae_770m"] == "parcae-770m"
    assert len(namespace["configs"]) == 14


def test_accuracy_curves_preserve_physical_depth_and_prelude_coda():
    namespace = setup()
    layers = [0, 1, *([2, 3, 4] * 3), 5, 6]
    result = DepthEvaluationResult(
        per_virtual_layer_metrics=[
            DepthAccuracy(
                layer_index=layer,
                virtual_layer_index=i,
                probe_accuracy=torch.tensor([i / 20]),
                logitlens_accuracy=torch.tensor([1 - i / 20]),
                logitlens_norm_accuracy=torch.tensor([0.5]),
                coda_accuracy=torch.tensor([i / 20 if layer <= 4 else float("nan")]),
            )
            for i, layer in enumerate(layers)
        ],
        model_name="synthetic",
        dataset_name="arc_easy",
        n_train=8,
        n_test=20,
    )
    for metric in namespace["METRICS"]:
        values = result.get_metric(metric).numpy()
        depth, accuracy = namespace["accuracy_curves"](result, metric)["all"]
        np.testing.assert_allclose(depth, np.linspace(0, 100, 13))
        np.testing.assert_array_equal(accuracy, values)
        curves = namespace["accuracy_curves"](result, metric, by_recurrence=True)
        assert list(curves) == ["Prelude", "R0", "R1", "R2", "Coda"]
        np.testing.assert_allclose(curves["Prelude"][0], [0, 100 / 6])
        np.testing.assert_allclose(curves["Coda"][0], [500 / 6, 100])
        for recurrence in range(3):
            depth, accuracy = curves[f"R{recurrence}"]
            np.testing.assert_allclose(depth, [200 / 6, 50, 400 / 6])
            np.testing.assert_array_equal(
                accuracy, values[2 + 3 * recurrence : 5 + 3 * recurrence]
            )
    coda = namespace["accuracy_curves"](result, "coda_accuracy", by_recurrence=True)
    np.testing.assert_allclose(coda["R2"][0], [200 / 6, 50, 400 / 6])
    assert np.isnan(coda["Coda"][1]).all()


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_notebook_trains_once_per_pair_dataset_and_exports_paper_figures(
    tmp_path,
    monkeypatch,
    mode,
):
    torch.set_num_threads(1)
    loads, dataset_loads, training, evaluations, figures = [], [], [], [], []
    forwards, coda_calls = [], []

    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 0
        pad_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return ([1] if add_special_tokens else []) + [
                2 + sum(map(ord, word)) % 60 for word in text.split()
            ]

    def model(name, config):
        loads.append((name, config.num_steps))
        bridge, structure = load_model(
            ModelSpec(
                MinimalRecurrentConfig(
                    d_model=16,
                    n_heads=2,
                    n_layers=2,
                    num_cycles=config.num_steps,
                    d_vocab=64,
                )
            ),
            device="cpu",
        )
        bridge.tokenizer = Tokenizer()
        if name.startswith("parcae_"):
            # A stand-in for C that changes the decoded option.
            bridge.core_projection = lambda hidden: hidden.roll(1, dims=-1)
        if name in support.BASELINES:
            structure = ModelLayerStructure([LayerGroup("blocks", 2, False)])
        forward = bridge.run_with_cache

        def cached_forward(*args, **kwargs):
            forwards.append(name)
            return forward(*args, **kwargs)

        bridge.run_with_cache = cached_forward
        return bridge, structure

    def dataset(name, split):
        dataset_loads.append((name, split))
        n_choices = 2 if name == "winogrande" else 4
        return [
            MCQExample(
                f"{split} question {index}",
                [f"{split} answer {choice}" for choice in range(n_choices)],
                index % n_choices,
                full_sentence_choices=name == "winogrande",
            )
            for index in range(12)
        ]

    original_train = probe_module.train_probes

    def train(
        bridge, tokenizer, examples, structure, recurrences, n_classes, device, **kwargs
    ):
        before = [p.detach().clone() for p in bridge.parameters()]
        training.append(
            ([ex.context for ex in examples], recurrences, n_classes, kwargs)
        )
        result = original_train(
            bridge,
            tokenizer,
            examples,
            structure,
            recurrences,
            n_classes,
            device,
            **kwargs,
        )
        for old, new in zip(before, bridge.parameters(), strict=True):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
            assert new.grad is None
        assert len(result) == len(list(structure.virtual_layers(recurrences)))
        return result

    original_evaluate = evaluation.evaluate_multiple_choice_by_depth

    def evaluate(*args, **kwargs):
        assert kwargs["probes"]  # Evaluation must reuse the probes just trained.
        assert kwargs["compute_coda"] == (
            args[-1] == "arc_easy"
            and args[-2]
            in {
                "raven",
                "retrofitted_llama",
                "retrofitted_olmo",
                "retrofitted_tinyllama",
                "parcae_140m",
                "parcae_370m",
                "parcae_770m",
            }
        )
        assert kwargs["project_pre_coda"] == (
            args[-1] == "arc_easy" and args[-2] in {"parcae_770m", "parcae_770m_rec1"}
        )
        evaluations.append([ex.context for ex in args[4]])
        return original_evaluate(*args, **kwargs)

    def toy_coda(bridge, hidden, *, input_ids, from_core):
        assert input_ids.shape == hidden.shape[:2] and from_core
        coda_calls.append(hidden.shape)
        assert hidden.shape[0] == 1
        return logit_lens(bridge, hidden.roll(1, dims=-1))

    original_save = support.save_figure

    def save(fig, directory, name, **kwargs):
        figures.append((name, fig))
        original_save(fig, directory, name, **kwargs)

    monkeypatch.setattr(support, "load_experiment_model", model)
    monkeypatch.setattr(evaluation, "load_mcq_dataset", dataset)
    monkeypatch.setattr(probe_module, "train_probes", train)
    monkeypatch.setattr(evaluation, "evaluate_multiple_choice_by_depth", evaluate)
    monkeypatch.setattr(
        importlib.import_module("looped_internals.evaluation.coda_lens"),
        "coda_lens",
        toy_coda,
    )
    monkeypatch.setattr(support, "save_figure", save)
    monkeypatch.setattr(plt, "show", lambda: None)
    paper_root = tmp_path / "paper"
    monkeypatch.setattr(support, "PAPER_PLOTS_DIR", paper_root)
    paper_dir = paper_root / "multiple_choice"
    paper_dir.mkdir(parents=True)
    previous = paper_dir / "linear_probe_summary.pdf"
    previous.write_bytes(b"Existing paper figure")
    namespace = {}
    for cell in cells():
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(
                mode=mode,
                device="cpu",
                output_dir=str(tmp_path / "results"),
                max_train=2,
                limit=2,
                probe_learning_rate=1e-3,
            )
    directory = namespace["run_dir"]
    assert directory.parent.name == "benchmarks"
    assert directory.name.startswith(f"{mode}-")
    assert loads == namespace["model_recurrences"]
    assert dataset_loads == [
        (name, split)
        for name in namespace["datasets"]
        for split in ("train", "test" if name.startswith("arc_") else "validation")
    ]
    assert len(training) == len(evaluations) == len(namespace["results"]) == 56
    # Probe training uses the resolved batches; each test example uses two forwards.
    n_training_forwards = len(namespace["datasets"]) * sum(
        (2 + namespace["probe_batch_sizes"][name] - 1)
        // namespace["probe_batch_sizes"][name]
        for name, _ in namespace["model_recurrences"]
    )
    assert len(forwards) == n_training_forwards + 56 * 2 * 2
    assert (
        len(coda_calls) == 7 * 2 * 64 * 4
    )  # Models × test examples × depths × options.
    expected_batch_sizes = [
        namespace["probe_batch_sizes"][name]
        for name, _ in namespace["model_recurrences"]
        for _ in namespace["datasets"]
    ]
    assert [call[-1] for call in training] == [
        dict(batch_size=batch_size, lr=1e-3, resume=None, checkpoint=None)
        for batch_size in expected_batch_sizes
    ]
    indices = support.sample_indices(12, 2, 42)
    assert all(
        call[0] == [f"train question {index}" for index in indices] for call in training
    )
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["mode"] == mode and metadata["probe_learning_rate"] == 1e-3
    assert metadata["probe_batch_sizes"] == namespace["probe_batch_sizes"]
    assert metadata["coda_dataset"] == "arc_easy"
    assert metadata["projected_model_recurrences"] == [
        ["parcae_770m", 32],
        ["parcae_770m_rec1", 1],
    ]
    assert (
        "models" not in metadata
        and "num_steps" not in metadata
        and "task" not in metadata
    )
    for name in namespace["datasets"]:
        sample = metadata["sampling"][name]
        assert sample["train_indices"] == sample["test_indices"] == indices
        assert sample["n_train"] == sample["n_test"] == 2
    for i, held_out in enumerate(evaluations):
        name = namespace["datasets"][i % 4]
        split = metadata["sampling"][name]["test_split"]
        assert held_out == [f"{split} question {index}" for index in indices]
    for (pair, dataset_name), result in namespace["results"].items():
        name, steps = pair
        stem = f"{name}-{steps}-steps-{dataset_name}"
        restored = DepthEvaluationResult.load(directory / f"{stem}.pt")
        assert restored.n_train == restored.n_test == 2
        for metric in namespace["METRICS"]:
            torch.testing.assert_close(
                restored.get_metric(metric), result.get_metric(metric)
            )
        if pair in namespace["coda_model_recurrences"] and dataset_name == "arc_easy":
            torch.testing.assert_close(
                restored.get_metric("coda_accuracy"), result.get_metric("coda_accuracy")
            )
        else:
            assert all(
                block.coda_accuracy is None
                for block in restored.per_virtual_layer_metrics
            )
        if pair in namespace["projected_pairs"] and dataset_name == "arc_easy":
            torch.testing.assert_close(
                restored.get_metric("projected_logitlens_accuracy"),
                result.get_metric("projected_logitlens_accuracy"),
            )
        else:
            assert all(
                block.projected_logitlens_accuracy is None
                for block in restored.per_virtual_layer_metrics
            )
        probes = torch.load(directory / f"{stem}-probes.pt", weights_only=True)
        assert probes["metadata"]["n_virtual_layers"] == len(
            result.per_virtual_layer_metrics
        )
    with (directory / "accuracies.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert (
        len(rows)
        == sum(
            len(r.per_virtual_layer_metrics) * 2 for r in namespace["results"].values()
        )
        + 7 * 64
        + 64  # Projected Parcae readout; the toy control has two layers.
        + 2
    )
    expected = {
        "feedforward_comparison",
        "huginn_recurrence",
        "ouro_recurrence",
        "retrofitted-llama_recurrence",
        "retrofitted-olmo_recurrence",
        "retrofitted-tinyllama_recurrence",
        "parcae-140m_recurrence",
        "parcae-370m_recurrence",
        "parcae-770m_recurrence",
        "linear_probe_summary",
        "logitlens_summary",
        "logitlens_projected_summary",
        "coda_lens_summary",
    }
    assert {name for name, _ in figures} == expected
    assert {p.stem for p in directory.glob("*.pdf")} == expected
    for filename, fig in figures:
        height = (
            3.6
            if filename == "coda_lens_summary"
            else (1.8 if filename.endswith("summary") else 2.6)
        )
        assert fig.get_size_inches() == pytest.approx((6.75, height))
        if filename == "feedforward_comparison":
            assert len(fig.axes) == 8
            assert all(
                [line.get_label() for line in ax.lines]
                == [
                    "Llama 3.2 1B",
                    "OLMo2",
                    "TinyLlama",
                    "Parcae 370M feedforward",
                    "Parcae 770M feedforward",
                ]
                for ax in fig.axes
            )
            assert all(ax.get_legend() is None for ax in fig.axes)
            assert len(fig.legends) == 1
            legend = fig.legends[0]
            assert legend._ncols == 3
            # Matplotlib lays out figure legends column-first. This interleaved
            # input produces standard controls on row one and Parcae on row two.
            assert [text.get_text() for text in legend.get_texts()] == [
                "Llama 3.2 1B",
                "Parcae 370M feedforward",
                "OLMo2",
                "Parcae 770M feedforward",
                "TinyLlama",
            ]
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend_bounds = legend.get_window_extent(renderer)
            assert legend_bounds.y0 > max(
                ax.get_window_extent(renderer).y1 for ax in fig.axes[:4]
            )
        elif filename.endswith("summary"):
            coda = filename == "coda_lens_summary"
            metric = {
                "linear_probe_summary": "probe_accuracy",
                "logitlens_summary": "logitlens_accuracy",
                "logitlens_projected_summary": "logitlens_accuracy",
                "coda_lens_summary": "coda_accuracy",
            }[filename]
            projected = filename == "logitlens_projected_summary"

            def readout(pair, metric):
                if projected and pair in namespace["projected_pairs"]:
                    return "projected_logitlens_accuracy"
                return metric

            pairs = namespace["figure_pairs"]["all" if coda else "main"]
            assert len(fig.axes) == len(pairs) + 1
            assert len(fig.legends) == 1
            legend = fig.legends[0]
            assert [t.get_text() for t in legend.get_texts()] == (
                ["Prelude", "Control"] if coda else ["Prelude", "Coda", "Control"]
            )
            assert legend._ncols == (2 if coda else 3)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend_bounds = legend.get_window_extent(renderer)
            for ax, pair in zip(fig.axes[:-1], pairs, strict=True):
                assert ax.get_legend() is None
                assert ax.get_title() == support.MODEL_LABELS[pair[0]] + (
                    "*" if projected and pair[0] == "parcae_770m" else ""
                )
                assert legend_bounds.y0 > ax.title.get_window_extent(renderer).y1
                assert all(
                    spine.get_edgecolor() == (0, 0, 0, 1)
                    for spine in ax.spines.values()
                )
                if coda and pair[0] == "ouro_1_4b":
                    assert not ax.lines and [t.get_text() for t in ax.texts] == [
                        "No coda"
                    ]
                    continue
                baseline_pair = (support.BASELINE_FOR[pair[0]], 1)
                baseline = namespace["results"][baseline_pair, "arc_easy"]
                baseline_metric = readout(
                    baseline_pair, "logitlens_accuracy" if coda else metric
                )
                np.testing.assert_array_equal(
                    ax.lines[0].get_ydata(),
                    baseline.get_metric(baseline_metric).numpy(),
                )
                assert ax.lines[0].get_linestyle() == "--"
                recurrent = namespace["results"][pair, "arc_easy"]
                np.testing.assert_array_equal(
                    np.concatenate([line.get_ydata() for line in ax.lines[1:]]),
                    recurrent.get_metric(readout(pair, metric)).numpy(),
                )
            if coda:
                assert len(fig.axes[3].lines) == len(fig.axes[7].lines)
                for main, appendix in zip(
                    fig.axes[3].lines, fig.axes[7].lines, strict=True
                ):
                    np.testing.assert_array_equal(
                        main.get_ydata(), appendix.get_ydata()
                    )
        else:
            assert len(fig.axes) == 9
            steps = 8 if filename == "ouro_recurrence" else 32
            assert all(len(ax.lines) == steps for ax in fig.axes[:8])
    if mode == "quick":
        assert list(paper_dir.iterdir()) == [previous]
        assert previous.read_bytes() == b"Existing paper figure"
    else:
        assert {p.stem for p in paper_dir.glob("*.pdf")} == expected
        for pdf in directory.glob("*.pdf"):
            assert (paper_dir / pdf.name).read_bytes() == pdf.read_bytes()
    namespace["save_figure"] = lambda fig, *args, **kwargs: plt.close(fig)
    for cell in cells():
        if cell["id"] in {
            "feedforward-figure",
            "recurrence-figures",
            "summary-figure",
            "logitlens-summary-figure",
            "projected-logitlens-summary-figure",
            "coda-summary-figure",
        }:
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
    assert len(loads) == 14 and len(training) == len(evaluations) == 56
    assert len(forwards) == n_training_forwards + 56 * 2 * 2
