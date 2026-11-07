"""Notebook integration with real tiny-model activations and local task fixtures."""

from __future__ import annotations

import json
import random
import re
import weakref
from dataclasses import replace
from pathlib import Path

import datasets
import numpy as np
import pytest
import torch

from experiments import support as experiments
from experiments.paper_models import MAIN_PARCAE_MODEL, PAPER_MODEL_GROUPS
from looped_internals.evaluation.lm_eval import BridgeLM
from looped_internals.models import ModelSpec, load_model
from looped_internals.models.minimal import MinimalRecurrentConfig
from looped_internals.structure import LayerGroup, ModelLayerStructure

NOTEBOOK = Path(__file__).parents[1] / "experiments" / "convergence.ipynb"


def _cells():
    """Read the convergence notebook cells from disk.

    Returns
    -------
    list of dict
        Notebook cells in execution order, including code and Markdown cells.
    """
    return json.loads(NOTEBOOK.read_text())["cells"]


def _notebook_science():
    """Load the specialized convergence reduction from its notebook cell.

    Returns
    -------
    dict
        Reduction-cell namespace, including the cache reducer and name filter.
    """
    from looped_internals.analysis import (
        compare_residuals,
        compute_attn_pattern_diff_norm,
    )
    from looped_internals.capture.extraction import _cached_virtual_layers

    namespace = {
        "torch": torch,
        "compare_residuals": compare_residuals,
        "compute_attn_pattern_diff_norm": compute_attn_pattern_diff_norm,
        "_cached_virtual_layers": _cached_virtual_layers,
    }
    source = next(
        "".join(cell["source"])
        for cell in _cells()
        if "def reduce_convergence_cache(" in "".join(cell["source"])
    )
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


def _generic_summary(run):
    """Reference the previous generic reduction for cache-infrastructure tests."""
    virtual_indices = {}
    for block in run.per_virtual_layer_metrics:
        virtual_indices.setdefault(block.layer_index, []).append(
            block.virtual_layer_index
        )
    layers = [layer for layer, indices in virtual_indices.items() if len(indices) > 1]
    metrics = (
        "to_final_state_residual_diff_norm",
        "to_final_state_residual_cosine_sim",
        "to_final_state_attn_pattern_diff_norm",
        "successive_recurrence_residual_diff_norm",
        "successive_recurrence_residual_cosine_sim",
        "successive_recurrence_attn_pattern_diff_norm",
    )
    curves = {}
    for metric in metrics:
        values = getattr(run, metric)
        curves[metric] = []
        for layer in layers:
            selected = (
                values[virtual_indices[layer], layer]
                if metric.startswith("to_final_state_")
                else values[layer]
            )
            curves[metric].append(selected.float().mean(-1).cpu().tolist())
    return {"layers": layers, "curves": curves}


def test_mode_defaults_and_overrides():
    """Verify quick and full presets and explicit parameter overrides."""
    quick = experiments.configure_experiment()
    full = experiments.configure_experiment(mode="full", dataset="hellaswag")
    assert (quick.limit, quick.num_steps, quick.models) == (
        2,
        8,
        ("ouro_1_4b", "retrofitted_tinyllama"),
    )
    assert (full.limit, full.num_steps, full.models) == (
        256,
        128,
        tuple(experiments.MODELS),
    )
    assert full.task == "hellaswag_likelihood"
    explicit = experiments.configure_experiment(
        mode="full", limit=1, num_steps=3, models="raven, ouro_1_4b"
    )
    assert (explicit.limit, explicit.num_steps, explicit.models) == (
        1,
        3,
        ("raven", "ouro_1_4b"),
    )
    assert experiments.configure_experiment(limit=0).limit is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "typo"},
        {"dataset": "typo"},
        {"limit": -1},
        {"num_steps": 1},
        {"models": "typo"},
        {"models": []},
        {"models": "raven,raven"},
    ],
)
def test_invalid_parameters_fail_early(kwargs):
    """Reject invalid experiment parameters before model loading.

    Parameters
    ----------
    kwargs : dict
        Invalid keyword arguments passed to ``configure_experiment``.
    """
    with pytest.raises(ValueError):
        experiments.configure_experiment(**kwargs)


def test_sample_selection_matches_legacy_shuffle():
    """Verify legacy shuffle order, full-split sampling, and oversized limits."""
    expected = list(range(1319))
    random.Random(42).shuffle(expected)
    assert experiments.sample_indices(1319, 256, 42) == expected[:256]
    assert experiments.sample_indices(1319, None, 42) == expected
    with pytest.raises(ValueError, match="split contains"):
        experiments.sample_indices(3, 4, 42)


@pytest.mark.parametrize("dataset", ["gsm8k", "hellaswag", None])
def test_paper_exports_preserve_archives_and_other_figures(
    tmp_path, monkeypatch, dataset
):
    """Only full runs refresh matching PDFs, including nested rollout figures."""
    paper_root = tmp_path / "paper-plots"
    monkeypatch.setattr(experiments, "PAPER_PLOTS_DIR", paper_root)
    config = experiments.configure_experiment(device="cpu")
    run_dir = tmp_path / "run"
    (run_dir / "eight-step").mkdir(parents=True)
    (run_dir / "curve.pdf").write_bytes(b"primary PDF")
    (run_dir / "eight-step" / "curve.pdf").write_bytes(b"eight-step PDF")
    (run_dir / "run.json").write_text("{}")

    assert experiments.export_paper_figures(config, run_dir, dataset=dataset) is None
    assert not paper_root.exists()

    config = replace(config, mode="full")
    destination = paper_root / "convergence"
    if dataset is not None:
        destination /= dataset
    assert (
        experiments.export_paper_figures(config, run_dir, dataset=dataset)
        == destination
    )
    assert {path.name for path in destination.iterdir()} == {
        "curve.pdf",
        "eight-step-curve.pdf",
    }
    assert (destination / "curve.pdf").read_bytes() == b"primary PDF"
    assert (destination / "eight-step-curve.pdf").read_bytes() == b"eight-step PDF"

    # A new full run replaces matching figures while preserving absent plots
    # and figures belonging to the other dataset.
    next_run = tmp_path / "next-run"
    next_run.mkdir()
    (next_run / "curve.pdf").write_bytes(b"updated PDF")
    other_dataset = "hellaswag" if dataset == "gsm8k" else "gsm8k"
    other_dir = paper_root / "convergence" / other_dataset
    other_dir.mkdir(parents=True, exist_ok=True)
    (other_dir / "curve.pdf").write_bytes(b"other dataset PDF")
    experiments.export_paper_figures(config, next_run, dataset=dataset)
    assert (destination / "curve.pdf").read_bytes() == b"updated PDF"
    assert (destination / "eight-step-curve.pdf").read_bytes() == b"eight-step PDF"
    assert (other_dir / "curve.pdf").read_bytes() == b"other dataset PDF"
    assert (run_dir / "curve.pdf").read_bytes() == b"primary PDF"
    assert (run_dir / "eight-step" / "curve.pdf").read_bytes() == b"eight-step PDF"
    assert (run_dir / "run.json").read_text() == "{}"

    experiments.export_paper_figures(
        replace(config, mode="quick"), run_dir, dataset=dataset
    )
    assert (destination / "curve.pdf").read_bytes() == b"updated PDF"


def test_papermill_parameter_injection(tmp_path, caplog):
    """Verify that Papermill injects the declared notebook parameters.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the prepared notebook.
    caplog : pytest.LogCaptureFixture
        Captured logs used to detect parameter parsing warnings.
    """
    papermill = pytest.importorskip("papermill")
    prepared = papermill.execute_notebook(
        NOTEBOOK,
        tmp_path / "prepared.ipynb",
        parameters={
            "dataset": "hellaswag",
            "model_recurrences": [["ouro_1_4b", 3]],
            "limit": 1,
        },
        prepare_only=True,
    )
    injected = next(
        cell
        for cell in prepared.cells
        if "injected-parameters" in cell.metadata.get("tags", [])
    )
    namespace = {}
    exec(injected.source, namespace)
    assert (
        namespace["dataset"],
        namespace["model_recurrences"],
        namespace["limit"],
    ) == (
        "hellaswag",
        [["ouro_1_4b", 3]],
        1,
    )
    assert "Unable to parse" not in caplog.text
    assert "unknown parameter" not in caplog.text


@pytest.fixture
def local_inference(monkeypatch):
    """Provide local tiny-model inference with real evaluation and metrics.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing model and task loading with local implementations.

    Returns
    -------
    dict of str to int
        Mutable ``loads`` and ``forwards`` counters updated during collection.
    """
    from lm_eval.tasks import TaskManager

    counts = {"loads": 0, "forwards": 0}
    run_refs = []

    # Index only the packaged tasks used here, not every lm-eval benchmark.
    task_path = NOTEBOOK.parents[1] / "src/looped_internals/evaluation/tasks"
    monkeypatch.setattr(
        experiments,
        "get_task_manager",
        lambda: TaskManager(include_path=task_path, include_defaults=False),
    )

    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=True):
            """Map characters to token IDs with an optional leading token.

            Parameters
            ----------
            text : str
                Text to encode.
            add_special_tokens : bool, optional
                Whether to prepend token ID one, defaulting to True.

            Returns
            -------
            list of int
                Encoded characters with the optional leading token.
            """
            return ([1] if add_special_tokens else []) + [2 + ord(c) % 90 for c in text]

    def load_dataset(*args, **kwargs):
        # Exercise the real packaged GSM8K and HellaSwag tasks with local data.
        docs = datasets.Dataset.from_list(
            [
                {
                    "question": text,
                    "answer": "!",
                    "activity_label": "Test",
                    "ctx_a": text,
                    "ctx_b": "",
                    "endings": ["!", "?"],
                    "label": "0",
                }
                for text in ["hi", "longer text", "hello"]
            ]
        )
        return datasets.DatasetDict(
            {split: docs for split in ["train", "validation", "test"]}
        )

    def load(name, config):
        """Load a tiny local model and increment the shared load counter.

        Parameters
        ----------
        name : str
            Registry name determining whether to use a single baseline cycle.
        config : experiments.ExperimentConfig
            Recurrence count and analysis settings for the evaluation wrapper.

        Returns
        -------
        BridgeLM
            CPU model with a local tokenizer and inference counters.
        """
        counts["loads"] += 1
        bridge, structure = load_model(
            ModelSpec(
                MinimalRecurrentConfig(
                    d_model=16,
                    n_heads=2,
                    n_layers=2,
                    num_cycles=1 if name in experiments.BASELINES else config.num_steps,
                    d_vocab=100,
                    expansion_factor=2,
                )
            ),
            device="cpu",
        )
        bridge.tokenizer = Tokenizer()
        forward = bridge.run_with_cache

        def counted_forward(*args, **kwargs):
            assert all(ref() is None for ref in run_refs)
            counts["forwards"] += 1
            output, cache = forward(*args, **kwargs)
            run_refs.append(weakref.ref(cache))
            return output, cache

        monkeypatch.setattr(bridge, "run_with_cache", counted_forward)
        return BridgeLM(
            bridge,
            structure,
            compute_differences=config.experiment == "convergence",
            compute_residual_entropy=config.compute_residual_entropy,
        )

    monkeypatch.setattr(experiments, "_load_model", load)
    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    torch.set_num_threads(1)
    return counts


def test_collection_always_recomputes_and_keeps_only_curves(tmp_path, local_inference):
    """Verify recomputation, metric release, and separate timestamped exports.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root for exported results.
    local_inference : dict of str to int
        Shared load and forward counters from the local inference fixture.
    """
    config = experiments.configure_experiment(
        models="ouro_1_4b,retrofitted_tinyllama",
        limit=2,
        num_steps=3,
        seed=0,
        device="cpu",
        output_dir=str(tmp_path / "results"),
    )
    reduce = _generic_summary
    observed_tokens = []

    def summarize(run):
        observed_tokens.append(run.input_ids.flatten().tolist())
        return reduce(run)

    results, run_info = experiments.collect_metrics(config, summarize)
    assert local_inference == {"loads": 2, "forwards": 4}
    assert run_info["indices"] == [0, 2]
    expected_tokens = [
        [1] + [2 + ord(c) % 90 for c in f"Question: {question}\nAnswer:!"]
        for question in ["hi", "hello"]
    ]
    assert observed_tokens == expected_tokens * len(config.models)
    assert all(len(examples) == 2 for examples in results.values())
    # Collection does no result-file I/O; export is a separate notebook step.
    assert not list(tmp_path.iterdir())
    output = experiments.save_results(results, run_info, config.output_dir)
    assert json.loads((output / "summaries.json").read_text()) == results
    assert json.loads((output / "run.json").read_text())["num_steps"] == 3

    repeated, repeated_info = experiments.collect_metrics(config, summarize)
    assert local_inference == {"loads": 4, "forwards": 8}
    assert (repeated, repeated_info) == (results, run_info)
    assert (
        experiments.save_results(repeated, repeated_info, config.output_dir) != output
    )


def test_failed_collection_restarts_all_examples(tmp_path, local_inference):
    """Verify that retrying a failed reduction recomputes every example.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root checked for unexpected result files.
    local_inference : dict of str to int
        Shared load and forward counters from the local inference fixture.
    """
    config = experiments.configure_experiment(
        models="ouro_1_4b",
        limit=2,
        num_steps=3,
        device="cpu",
        output_dir=str(tmp_path / "results"),
    )
    summarize = _generic_summary
    calls = 0

    def interrupted(run):
        """Reduce a model run, failing on the second call.

        Parameters
        ----------
        run : ModelRunData
            Per-example metrics to reduce.

        Returns
        -------
        dict
            Reduced convergence curves when no interruption is triggered.

        Raises
        ------
        RuntimeError
            On the second invocation to simulate an interrupted collection.
        """
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return summarize(run)

    with pytest.raises(RuntimeError, match="interrupted"):
        experiments.collect_metrics(config, interrupted)
    assert not list(tmp_path.iterdir())
    experiments.collect_metrics(config, summarize)
    assert local_inference == {"loads": 2, "forwards": 4}


@pytest.mark.parametrize(
    "tokens,batch_size,dtype", [(3, 1, torch.float32), (7, 2, torch.bfloat16)]
)
@pytest.mark.parametrize("accumulated", [False, True])
def test_same_layer_reduction_handles_prelude_and_variable_lengths(
    tokens, batch_size, dtype, accumulated
):
    """Match the generic curves while excluding prelude, coda, and cross-layer work."""
    from types import SimpleNamespace

    from looped_internals.capture import compute_model_run_data

    structure = ModelLayerStructure(
        groups=[
            LayerGroup("prelude_blocks", 1, False),
            LayerGroup("core_blocks", 2, True),
            LayerGroup("coda_blocks", 1, False),
        ]
    )
    torch.manual_seed(4)
    cache_dict = {}
    suffix = ".iter_0" if accumulated else ""
    for prefix in ("prelude_blocks.0", "coda_blocks.0"):
        cache_dict[f"{prefix}.hook_out{suffix}"] = torch.randn(
            batch_size, tokens, 4, dtype=dtype
        )
        cache_dict[f"{prefix}.attn.hook_pattern{suffix}"] = torch.randn(
            batch_size, 3, tokens, tokens
        )
    for recurrence in range(3):
        for layer in range(2):
            suffix = f".iter_{recurrence}"
            prefix = f"core_blocks.{layer}"
            cache_dict[f"{prefix}.hook_out{suffix}"] = torch.randn(
                batch_size, tokens, 4, dtype=dtype
            )
            cache_dict[f"{prefix}.attn.hook_pattern{suffix}"] = torch.randn(
                batch_size, 3, tokens, tokens
            )
    for key, value in cache_dict.items():
        if "hook_out" in key:
            value[:, 0] = 0  # Preserve the zero-vector cosine convention.
    if accumulated:
        # run_with_cache(accumulate=True) numbers even single-pass layers and
        # also retains unsuffixed entries containing each hook's last value.
        for key, value in list(cache_dict.items()):
            cache_dict[key.rsplit(".iter_", 1)[0]] = value
    cache = SimpleNamespace(cache_dict=cache_dict)
    input_ids = torch.arange(tokens).expand(batch_size, -1)

    generic = compute_model_run_data(
        cache,
        input_ids,
        structure,
        compute_pairwise=False,
        compute_differences=True,
    )
    expected = [_generic_summary(run) for run in generic]
    science = _notebook_science()
    reduced = science["reduce_convergence_cache"](cache, input_ids, structure)

    assert len(reduced) == batch_size
    for batch, record in enumerate(reduced):
        torch.testing.assert_close(record["input_ids"], input_ids[batch : batch + 1])
        assert record["summary"] == expected[batch]
        assert expected[batch]["layers"] == [1, 2]
    selected = science["recurrent_metric_names"](structure)
    assert selected("core_blocks.0.hook_out")
    assert selected("core_blocks.1.attn.hook_pattern")
    assert not selected("prelude_blocks.0.hook_out")
    assert not selected("coda_blocks.0.attn.hook_pattern")


@pytest.mark.parametrize("mode, limit", [("quick", 1), ("full", 2)])
def test_notebook_runs_all_cells_and_exports_layerwise_figures(
    tmp_path, monkeypatch, local_inference, mode, limit
):
    """Check real reduction, recurrence indexing, uncertainty bands, and exports."""
    import matplotlib.pyplot as plt

    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    paper_root = tmp_path / "paper-plots"
    monkeypatch.setattr(experiments, "PAPER_PLOTS_DIR", paper_root)
    paper_dir = paper_root / "convergence" / "hellaswag"
    paper_dir.mkdir(parents=True)
    existing_figure = paper_dir / "fixed_point_norms.pdf"
    existing_figure.write_bytes(b"Previous full-run figure")
    figures = {}
    pairs = [
        (name, 2 if name == "ouro_1_4b" else 3)
        for name in dict.fromkeys(PAPER_MODEL_GROUPS["all"])
    ]
    save_figure = experiments.save_figure

    def record_figure(fig, run_dir, name, **kwargs):
        save_figure(fig, run_dir, name, **kwargs)
        figures[name] = fig

    monkeypatch.setattr(experiments, "save_figure", record_figure)
    namespace = {}
    for cell in _cells():
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            assert namespace["model_recurrences"] is None
            assert namespace["recurrences"] == 128
            # Same ordering as Papermill's injected-parameters cell.
            namespace.update(
                mode=mode,
                dataset="hellaswag",
                model_recurrences=[list(pair) for pair in pairs],
                limit=limit,
                device="cpu",
                output_dir=str(tmp_path / "results"),
            )
    output = namespace["run_dir"]
    assert output.name.startswith(f"{mode}-")
    filenames = {
        "to_final_state_residual_diff_norm": "fixed_point_norms",
        "to_final_state_residual_cosine_sim": "fixed_point_cosine_similarities",
        "to_final_state_attn_pattern_diff_norm": "fixed_point_attention_differences",
        "successive_recurrence_residual_diff_norm": "successive_recurrence_residual_diff_norm",
        "successive_recurrence_residual_cosine_sim": "successive_recurrence_residual_cosine_sim",
        "successive_recurrence_attn_pattern_diff_norm": "successive_recurrence_attn_pattern_diff_norm",
    }
    assert {path.stem for path in output.glob("*.pdf")} == {
        f"{filename}_{group}"
        for filename in filenames.values()
        for group in PAPER_MODEL_GROUPS
    }
    assert len(figures) == 18
    assert not list(output.rglob("*.png"))
    if mode == "full":
        assert {path.name for path in paper_dir.iterdir()} == {
            path.name for path in output.glob("*.pdf")
        } | {existing_figure.name}
        for figure in output.glob("*.pdf"):
            assert (paper_dir / figure.name).read_bytes() == figure.read_bytes()
    else:
        assert list(paper_dir.iterdir()) == [existing_figure]
        assert existing_figure.read_bytes() == b"Previous full-run figure"
        assert namespace["paper_dir"] is None
    assert existing_figure.read_bytes() == b"Previous full-run figure"
    overview = torch.load(output / "overview.pt", weights_only=False)
    assert overview["summaries"] == namespace["summaries"]
    assert list(overview["summaries"]) == pairs
    assert all(
        overview["run"][pair]["indices"] == overview["run"][pairs[0]]["indices"]
        for pair in pairs
    )
    assert overview["figure_pairs"] == namespace["figure_pairs"]
    assert [pair[0] for pair in overview["figure_pairs"]["all"]].count(
        "parcae_370m"
    ) == 1 + (MAIN_PARCAE_MODEL == "parcae_370m")
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["mode"] == mode and metadata["limit"] == limit
    assert metadata["model_recurrences"] == [list(pair) for pair in pairs]
    assert metadata["figure_pairs"] == {
        group: [list(pair) for pair in selected]
        for group, selected in namespace["figure_pairs"].items()
    }
    assert "models" not in metadata and "num_steps" not in metadata
    assert local_inference == {"loads": len(pairs), "forwards": len(pairs) * limit}

    for metric, filename in filenames.items():
        for group, names in PAPER_MODEL_GROUPS.items():
            stem = f"{filename}_{group}"
            fig = figures[stem]
            height = 3 if group == "all" else 1.5
            assert fig.get_size_inches() == pytest.approx((6.75, height))
            panels = [ax for ax in fig.axes if ax.lines]
            assert len(panels) == len(names)
            assert [ax.get_title() for ax in panels] == [
                experiments.MODEL_LABELS[name].replace("Retrofitted ", "R.")
                for name in names
            ]
            assert sum(bool(ax.get_ylabel()) for ax in panels) == (
                2 if group == "all" else 1
            )
            fig.canvas.draw()
            bounds = fig.get_tightbbox(fig.canvas.get_renderer())
            assert bounds.x0 >= 0 and bounds.y0 >= 0
            assert bounds.x1 <= 6.75 and bounds.y1 <= height
            media_box = re.search(
                rb"/MediaBox\s*\[([^]]+)\]", (output / f"{stem}.pdf").read_bytes()
            )
            assert media_box is not None
            assert [float(value) for value in media_box[1].split()] == pytest.approx(
                [0, 0, 6.75 * 72, height * 72]
            )
            for ax, pair in zip(panels, namespace["figure_pairs"][group], strict=True):
                examples = overview["summaries"][pair]
                values = np.asarray([example["curves"][metric] for example in examples])
                means = values.mean(0)
                stds = values.std(0, ddof=int(limit > 1))
                start = 1 if metric.startswith("to_final_state_") else 2
                x = np.arange(start, pair[1] + 1)
                for line, band, mean, std in zip(
                    ax.lines, ax.collections, means, stds, strict=True
                ):
                    np.testing.assert_array_equal(line.get_xdata(), x)
                    np.testing.assert_allclose(line.get_ydata(), mean)
                    vertices = band.get_paths()[0].vertices
                    assert np.isfinite(vertices).all()
                    for recurrence, average, spread in zip(x, mean, std, strict=True):
                        y = vertices[vertices[:, 0] == recurrence, 1]
                        assert y.min() == pytest.approx(average - spread)
                        assert y.max() == pytest.approx(average + spread)
                if start == 1:
                    expected = 1 if metric.endswith("cosine_sim") else 0
                    np.testing.assert_allclose(means[:, -1], expected, atol=1e-6)

    # Rerunning the figure cells reuses the reductions and creates no new run.
    with plt.rc_context():
        for cell in _cells():
            source = "".join(cell["source"])
            if cell["cell_type"] == "code" and source.startswith("plot_convergence("):
                exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["run_dir"] == output
    assert local_inference == {"loads": len(pairs), "forwards": len(pairs) * limit}
