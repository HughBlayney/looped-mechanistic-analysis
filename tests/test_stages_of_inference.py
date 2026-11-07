"""Verify the reduced stage curves against the old aggregation and run all cells."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_experiments import local_inference as _local_inference

from experiments import support as experiments
from experiments.paper_models import MAIN_PARCAE_MODEL, PAPER_MODEL_GROUPS
from looped_internals import models
from looped_internals.results import (
    ModelRunData,
    PerVirtualLayerMetrics,
    aggregate_values,
)

NOTEBOOK = Path(__file__).parents[1] / "experiments" / "stages_of_inference.ipynb"


@pytest.fixture
def local_inference(monkeypatch):
    """Reuse the local tiny-model inference fixture for stage analyses.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing model and task loading with local implementations.

    Returns
    -------
    dict of str to int
        Mutable model-load and forward-call counters from the shared fixture.
    """
    counts = _local_inference.__wrapped__(monkeypatch)
    task_manager = experiments.get_task_manager()
    monkeypatch.setattr(experiments, "get_task_manager", lambda: task_manager)
    original_load = experiments._load_model

    def load(name, config):
        lm = original_load(name, config)
        if name == "retrofitted_llama":
            from test_match_raven import RavenConfig, RavenForCausalLM
            from transformer_lens.config import TransformerBridgeConfig
            from transformer_lens.model_bridge import TransformerBridge

            from looped_internals.models.adapters.raven import RavenArchitectureAdapter
            from looped_internals.structure import LayerGroup, ModelLayerStructure

            # Exercise the exact prelude hook on a tiny model with real layer groups.
            model = RavenForCausalLM(
                RavenConfig(
                    n_embd=16,
                    n_heads=2,
                    n_layers=5,
                    block_size=64,
                    vocab_size=100,
                    padding_multiple=1,
                    intermediate_size=32,
                    n_layers_in_prelude=2,
                    n_layers_in_recurrent_block=2,
                    n_layers_in_coda=1,
                    mean_recurrence=config.num_steps,
                    mean_backprop_depth=0,
                    tie_embeddings=False,
                )
            )
            cfg = TransformerBridgeConfig(
                d_model=16,
                d_head=8,
                n_heads=2,
                n_layers=5,
                n_ctx=64,
                d_vocab=100,
                architecture="RavenForCausalLM",
            )
            bridge = TransformerBridge(
                model, RavenArchitectureAdapter(cfg), tokenizer=None
            ).eval()
            bridge.tokenizer = lm.bridge.tokenizer
            forward = bridge.run_with_cache

            def counted_forward(*args, **kwargs):
                counts["forwards"] += 1
                return forward(*args, **kwargs)

            monkeypatch.setattr(bridge, "run_with_cache", counted_forward)
            lm.bridge = bridge
            lm.structure = ModelLayerStructure(
                [
                    LayerGroup("prelude_blocks", 2, False),
                    LayerGroup("core_blocks", 2, True),
                    LayerGroup("coda_blocks", 1, False),
                ]
            )
        return lm

    monkeypatch.setattr(experiments, "_load_model", load)
    return counts


def cells():
    """Read the stages-of-inference notebook cells from disk.

    Returns
    -------
    list of dict
        Notebook cells in execution order, including code and Markdown cells.
    """
    return json.loads(NOTEBOOK.read_text())["cells"]


def science():
    """Load stage reductions and grouping helpers without running collection.

    Returns
    -------
    dict
        Execution namespace containing metric names and notebook analysis helpers.
    """
    namespace = {"np": np, "ablation_position": 0}
    for cell in cells():
        source = "".join(cell["source"])
        if "def aggregate_stages(" in source:
            # Only the function definitions; aggregation calls occur after them.
            source = source.split("\nprimary =")[0]
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("fail", [False, True])
def test_ablation_changes_only_first_token_and_hooks_are_scoped(
    monkeypatch, local_inference, fail
):
    config = experiments.configure_experiment(
        experiment="stages_of_inference",
        models=["retrofitted_llama"],
        limit=2,
        num_steps=2,
        device="cpu",
    )
    lm = experiments._load_model("retrofitted_llama", config)
    monkeypatch.setattr(experiments, "_load_model", lambda *args: lm)
    hook_name = "prelude_blocks.1.mlp.hook_out"
    ablate = science()["zero_first_token_mlp"]
    calls = []

    def checked_ablation(value, hook):
        before = value.clone()
        result = ablate(value, hook)
        assert torch.count_nonzero(before[:, 0]) > 0
        assert torch.count_nonzero(result[:, 0]) == 0
        torch.testing.assert_close(result[:, 1:], before[:, 1:])
        calls.append(hook.name)
        if fail:
            raise RuntimeError("ablation interrupted")
        return result

    if fail:
        with pytest.raises(RuntimeError, match="ablation interrupted"):
            experiments.collect_metrics(
                config,
                experiments.summarize_stages,
                fwd_hooks=[(hook_name, checked_ablation)],
            )
    else:
        results, info = experiments.collect_metrics(
            config,
            experiments.summarize_stages,
            fwd_hooks=[(hook_name, checked_ablation)],
        )
        assert len(results["retrofitted_llama"]) == 2
        assert info["forward_hooks"] == [hook_name]
    assert calls == [hook_name] * (1 if fail else 2)
    assert all(not point.fwd_hooks for point in lm.bridge.hook_dict.values())
    assert not lm.all_model_data
    # The released intervention must not change a subsequent ordinary forward.
    with torch.no_grad():
        _, cache = lm.bridge.run_with_cache(
            torch.tensor([[1, 2, 3], [4, 5, 6]]), names_filter=[hook_name]
        )
    assert torch.count_nonzero(cache[hook_name][:, 0]) > 0


def test_stages_defaults_keep_convergence_defaults():
    """Verify stage-specific recurrence defaults and baseline restrictions."""
    stages = experiments.configure_experiment(
        experiment="stages_of_inference",
        mode="full",
        include_baselines=True,
    )
    assert stages.num_steps == 8
    assert stages.limit == 256
    assert stages.models == tuple(experiments.MODELS)
    assert experiments.configure_experiment(mode="full").num_steps == 128
    with pytest.raises(ValueError, match="Feedforward"):
        experiments.configure_experiment(include_baselines=True)


@pytest.mark.parametrize("baseline", [False, True])
@pytest.mark.parametrize("entropy", [False, True])
def test_loader_enables_entropy_without_convergence_or_baseline_recurrences(
    monkeypatch, baseline, entropy
):
    """Verify stage metric flags and recurrence settings for each model type.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing the model loader with an argument-capturing stub.
    baseline : bool
        Whether to load a feedforward baseline instead of a recurrent model.
    """
    captured = {}

    def load(name, **kwargs):
        """Capture loader arguments and return a minimal bridge stub.

        Parameters
        ----------
        **kwargs : dict
            Loader settings copied into the enclosing test's captured dictionary.

        Returns
        -------
        bridge : types.SimpleNamespace
            Stub exposing a small PyTorch model for device discovery.
        structure : types.SimpleNamespace
            Placeholder block structure for wrapper initialization.
        """
        captured.update(name=name, **kwargs)
        bridge = SimpleNamespace(original_model=torch.nn.Linear(1, 1))
        return bridge, SimpleNamespace()

    monkeypatch.setattr(models, "load_model", load)
    config = experiments.configure_experiment(
        experiment="stages_of_inference",
        device="cpu",
        num_steps=3,
        compute_residual_entropy=entropy,
    )
    lm = experiments._load_model(
        "llama_1b" if baseline else "retrofitted_llama", config
    )
    assert lm.compute_residual_entropy is entropy
    assert not lm.compute_differences and not lm.compute_pairwise
    assert lm.residual_entropy_components is None
    assert lm.forward_kwargs == {}
    assert captured["num_steps"] == (None if baseline else 3)
    assert captured["name"] == ("llama_1b" if baseline else "retrofitted_llama_32")


def make_run(layers, offset):
    """Construct synthetic metrics for a specified sequence of virtual layers.

    Parameters
    ----------
    layers : list of int
        Layer indices in execution order.
    offset : float
        Shift applied to metric values and used to vary the input sequence length.

    Returns
    -------
    ModelRunData
        Synthetic attention and residual metrics for each requested virtual layer.
    """
    blocks = []
    for sequence, layer in enumerate(layers):
        values = torch.tensor([[0.1, 0.3, 0.7]]) + offset + sequence * 0.01
        blocks.append(
            PerVirtualLayerMetrics(
                layer_index=layer,
                virtual_layer_index=sequence,
                sink_scores=values,
                mixing_scores=values,
                colsum_concentrations=values,
                residual_norms=torch.tensor([1.0 + offset + sequence]),
                residual_norms_last=torch.tensor([2.0 + offset + sequence]),
                residual_norms_mean=torch.tensor([1.5 + offset + sequence]),
                residual_entropy=torch.tensor([0.5 + offset + sequence]),
            )
        )
    return ModelRunData(
        input_ids=torch.zeros(1, 5 + int(offset * 10)), per_virtual_layer_metrics=blocks
    )


@pytest.mark.parametrize("layers", [[0, 1, 2, 1, 2, 1, 2, 3], [0, 1, 0, 1], [0, 1, 2]])
def test_reduced_means_spreads_and_group_positions_match_old_figures(layers):
    """Compare reduced stage statistics and positions with legacy aggregation.

    Parameters
    ----------
    layers : list of int
        Layer execution sequence covering recurrent or feedforward layouts.
    """
    ns = science()
    runs = [make_run(layers, 0), make_run(layers, 0.1)]
    reduced = [experiments.summarize_stages(run) for run in runs]
    data = ns["aggregate_stages"](reduced)
    groups = ns["recurrence_groups"](data)
    for field in ns["METRICS"]:
        old_all = aggregate_values(runs, field)["all"]
        np.testing.assert_allclose(
            data["curves"][field]["mean"], old_all.mean, rtol=1e-6
        )
        np.testing.assert_allclose(data["curves"][field]["std"], old_all.std, rtol=1e-6)
        if data["recurrent_layers"]:
            old_groups = aggregate_values(
                runs, field, group_by="recurrence", scale_by="layer"
            )
            assert list(groups) == list(old_groups)
            for key, indices in groups.items():
                np.testing.assert_allclose(
                    data["curves"][field]["mean"][indices],
                    old_groups[key].mean,
                    rtol=1e-6,
                )
                np.testing.assert_allclose(
                    np.asarray(layers)[indices] / max(layers) * 100,
                    old_groups[key].positions,
                    rtol=1e-6,
                )
    assert reduced[0]["curves"]["sink_rates"][0] == pytest.approx(2 / 3)


def setup(**parameters):
    namespace = {}
    for cell in cells():
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(parameters)
        if "configs" in namespace:
            return namespace
    raise AssertionError("Missing notebook configuration")


def test_shared_groups_select_only_required_pairs(monkeypatch):
    ns = setup()
    expected_short = [
        (name, 4 if name == "ouro_1_4b" else 32)
        for name in dict.fromkeys(PAPER_MODEL_GROUPS["all"])
    ]
    expected_long = [(name, 128) for name in PAPER_MODEL_GROUPS["main"]]
    expected_baselines = {
        (experiments.BASELINE_FOR[name], 1) for name, _ in expected_short
    }
    assert ns["short_runs"] == expected_short
    assert ns["long_pairs"] == expected_long
    assert ns["figure_pairs"] == {
        group: [(name, 4 if name == "ouro_1_4b" else 32) for name in names]
        for group, names in PAPER_MODEL_GROUPS.items()
    }
    assert len(ns["model_recurrences"]) == 19
    assert set(ns["model_recurrences"]) == (
        set(expected_short + expected_long)
        | expected_baselines
        | {("retrofitted_llama", 8)}
    )
    assert set(ns["entropy_pairs"]) == set(expected_short) | expected_baselines
    assert ns["model_recurrences"].count((MAIN_PARCAE_MODEL, 32)) == 1
    assert ns["figure_pairs"]["all"].count((MAIN_PARCAE_MODEL, 32)) == 1
    assert ("parcae_140m_rec1", 1) in ns["configs"]
    assert ("parcae_370m_rec1", 1) in ns["configs"]
    main = (*PAPER_MODEL_GROUPS["main"][:3], "parcae_770m")
    monkeypatch.setitem(PAPER_MODEL_GROUPS, "main", main)
    monkeypatch.setitem(
        PAPER_MODEL_GROUPS, "all", main + PAPER_MODEL_GROUPS["appendix"]
    )
    updated = setup()
    assert updated["figure_pairs"]["main"][-1] == ("parcae_770m", 32)
    assert updated["figure_pairs"]["appendix"][-1] == ("parcae_370m", 32)
    assert ("parcae_770m", 128) in updated["configs"]
    assert ("parcae_770m_rec1", 1) in updated["configs"]
    overridden = setup(profile_recurrences=16, ouro_recurrences=6, long_recurrences=64)
    assert overridden["figure_pairs"]["main"][0] == ("ouro_1_4b", 6)
    assert overridden["figure_pairs"]["main"][-1] == ("parcae_770m", 16)
    assert overridden["long_pairs"][-1] == ("parcae_770m", 64)


@pytest.mark.parametrize(
    "dataset, mode, limit", [("gsm8k", "full", 1), ("hellaswag", "quick", 2)]
)
def test_notebook_collects_only_required_runs_and_exports_figures(
    tmp_path, monkeypatch, local_inference, dataset, mode, limit
):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from looped_internals.capture import extraction

    monkeypatch.setattr(plt, "show", lambda: None)
    paper_root = tmp_path / "paper-plots"
    monkeypatch.setattr(experiments, "PAPER_PLOTS_DIR", paper_root)
    paper_dir = paper_root / "stages_of_inference" / dataset
    paper_dir.mkdir(parents=True)
    protected = paper_dir / "residual_norm_comparison.pdf"
    protected.write_bytes(b"Previous full run")

    entropy_calls = 0
    original_entropy = extraction.get_residual_entropy

    def entropy(*args, **kwargs):
        nonlocal entropy_calls
        entropy_calls += 1
        return original_entropy(*args, **kwargs)

    monkeypatch.setattr(extraction, "get_residual_entropy", entropy)
    collected = []
    original_collect = experiments.collect_metrics

    def collect(config, summarize, **kwargs):
        before = entropy_calls
        result = original_collect(config, summarize, **kwargs)
        assert (entropy_calls > before) is config.compute_residual_entropy
        collected.append(config)
        return result

    monkeypatch.setattr(experiments, "collect_metrics", collect)

    figures = {}
    original_save = experiments.save_figure

    def save(fig, run_dir, name, **kwargs):
        figures[name] = fig
        if name in {
            "attention_metrics_in_total_depth_main",
            "residual_norm_comparison_main",
            "recurrent/summarised",
            "recurrent/residual_entropy_all",
            "recurrent/residual_norm_mean_all",
            "ablation_retrofitted_llama_bos_mlp",
        }:
            fig.savefig(tmp_path / (name.replace("/", "-") + ".png"), dpi=180)
        original_save(fig, run_dir, name, **kwargs)

    monkeypatch.setattr(experiments, "save_figure", save)

    ns = {}
    for cell in cells():
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), ns)
        if "parameters" in cell["metadata"].get("tags", []):
            ns.update(
                dataset=dataset,
                mode=mode,
                limit=limit,
                device="cpu",
                output_dir=str(tmp_path),
            )

    expected_pairs = set(ns["short_runs"] + ns["long_pairs"] + ns["baseline_pairs"]) | {
        ("retrofitted_llama", 8)
    }
    assert set(ns["configs"]) == expected_pairs
    assert {(c.models[0], c.num_steps) for c in collected} == expected_pairs
    assert len(collected) == 20
    assert local_inference == {"loads": 20, "forwards": 20 * limit}
    assert collected[-1].models == ("retrofitted_llama",)
    assert collected[-1].num_steps == 32 and not collected[-1].compute_residual_entropy
    assert all(not c.include_baselines for c in collected)
    assert {
        pair
        for pair, config in ns["configs"].items()
        if config.compute_residual_entropy
    } == set(ns["entropy_pairs"])

    expected_figures = {
        "attention_metrics_in_total_depth_main",
        "attention_metrics_in_total_depth_appendix",
        "residual_norm_comparison_main",
        "residual_norm_comparison_appendix",
        "similar_behaviour/similar_behaviour_summarised",
        "similar_behaviour/stability_in_percentage_depth_summarised",
        "recurrent/summarised",
        "recurrent/residual_entropy_all",
        "recurrent/residual_norm_mean_all",
        "ablation_retrofitted_llama_bos_mlp",
        *(
            f"recurrent/{experiments.MODEL_LABELS[name]}"
            for name, _ in ns["short_runs"]
        ),
        *(
            f"recurrent/{experiments.MODEL_LABELS[name]} (128 steps)_to_fixed_point"
            for name in PAPER_MODEL_GROUPS["main"]
        ),
    }
    assert len(expected_figures) == 22 and set(figures) == expected_figures
    output = ns["run_dir"]
    assert output.name.startswith(mode + "-")
    assert {
        p.relative_to(output).with_suffix("").as_posix() for p in output.rglob("*.pdf")
    } == expected_figures
    assert not list(output.rglob("*.png"))

    def panels(fig):
        return [ax for ax in fig.axes if ax.lines]

    for name, fig in figures.items():
        compact = name in {
            "similar_behaviour/similar_behaviour_summarised",
            "similar_behaviour/stability_in_percentage_depth_summarised",
        }
        assert fig.get_figwidth() == pytest.approx(3.25 if compact else 6.75)
        if (
            name.startswith("recurrent/")
            or name == "ablation_retrofitted_llama_bos_mlp"
        ):
            assert fig.axes[-1].get_xlabel() == "Early"
            assert fig.axes[-1].get_title() == "Late"
            assert fig.axes[-1].get_ylabel() == "Recurrence"

    for group in ("main", "appendix"):
        pairs = ns["figure_pairs"][group]
        fig = figures[f"attention_metrics_in_total_depth_{group}"]
        assert [ax.get_ylabel() for ax in panels(fig)] == list(
            ns["ATTENTION_METRICS"].values()
        )
        for ax, field in zip(panels(fig), ns["ATTENTION_METRICS"], strict=True):
            assert len(ax.lines) == 4
            assert [line.get_label() for line in ax.lines] == [
                experiments.MODEL_LABELS[name] for name, _ in pairs
            ]
            for line, pair in zip(ax.lines, pairs, strict=True):
                np.testing.assert_allclose(
                    line.get_ydata(), ns["stages"][pair]["curves"][field]["mean"]
                )
        norm_fig = figures[f"residual_norm_comparison_{group}"]
        first_last, average = panels(norm_fig)
        assert len(first_last.lines) == 8 and len(average.lines) == 4
        assert first_last.get_yscale() == average.get_yscale() == "log"
        for index, pair in enumerate(pairs):
            lines = [*first_last.lines[2 * index : 2 * index + 2], average.lines[index]]
            assert [line.get_linestyle() for line in lines] == ["-", "--", "-"]
            assert lines[0].get_color() == lines[1].get_color() == lines[2].get_color()
            for line, field in zip(
                lines,
                ["residual_norms", "residual_norms_last", "residual_norms_mean"],
                strict=True,
            ):
                np.testing.assert_allclose(
                    line.get_ydata(), ns["stages"][pair]["curves"][field]["mean"]
                )

    # Every attention profile uses the matched baseline, including trained Parcae controls.
    for pair in ns["short_runs"] + ns["long_pairs"]:
        name, recurrences = pair
        filename = f"recurrent/{experiments.MODEL_LABELS[name]}"
        if recurrences == 128:
            filename += " (128 steps)_to_fixed_point"
        axes = panels(figures[filename])
        assert len(axes) == 3
        groups = ns["recurrence_groups"](ns["stages"][pair])
        baseline = ns["stages"][(experiments.BASELINE_FOR[name], 1)]
        for ax, field in zip(axes, ns["ATTENTION_METRICS"], strict=True):
            assert len(ax.lines) == len(groups) + 1
            bands = [c for c in ax.collections if isinstance(c, PolyCollection)]
            assert len(bands) == (1 if recurrences == 128 else len(groups) + 1)
            np.testing.assert_allclose(
                ax.lines[-1].get_ydata(), baseline["curves"][field]["mean"]
            )
            assert (
                ax.lines[-1].get_label()
                == experiments.MODEL_LABELS[experiments.BASELINE_FOR[name]]
            )

    for filename, field in (
        ("recurrent/residual_entropy_all", "residual_entropy"),
        ("recurrent/residual_norm_mean_all", "residual_norms_mean"),
        ("recurrent/summarised", "mixing_scores"),
    ):
        fig = figures[filename]
        group = "main" if filename == "recurrent/summarised" else "all"
        pairs = ns["figure_pairs"][group]
        axes = panels(fig)
        assert len(axes) == len(pairs)
        assert fig.get_figheight() == pytest.approx(1.8 if group == "main" else 3.6)
        assert [ax.get_title() for ax in axes] == [
            experiments.MODEL_LABELS[name] for name, _ in pairs
        ]
        for ax, pair in zip(axes, pairs, strict=True):
            data = ns["stages"][pair]
            for line, indices in zip(
                ax.lines[:-1], ns["recurrence_groups"](data).values(), strict=True
            ):
                np.testing.assert_allclose(
                    line.get_ydata(), data["curves"][field]["mean"][indices]
                )
        if group == "all":
            for first, repeated in zip(axes[3].lines, axes[7].lines, strict=True):
                np.testing.assert_array_equal(first.get_ydata(), repeated.get_ydata())

    if mode == "full":
        assert {p.name for p in paper_dir.iterdir()} == {
            name.replace("/", "-") + ".pdf" for name in expected_figures
        } | {protected.name}
        for figure in output.rglob("*.pdf"):
            assert (
                paper_dir / "-".join(figure.relative_to(output).parts)
            ).read_bytes() == figure.read_bytes()
    else:
        assert list(paper_dir.iterdir()) == [protected]
    assert protected.read_bytes() == b"Previous full run"

    settings = json.loads((output / "run.json").read_text())
    assert settings["mode"] == mode and settings["limit"] == limit
    assert settings["ablation"] == {
        "model": "retrofitted_llama",
        "recurrences": 32,
        "hook": "prelude_blocks.1.mlp.hook_out",
        "position": 0,
        "value": 0.0,
    }
    assert not {"models", "num_steps", "compute_residual_entropy"} & settings.keys()
    overview = torch.load(output / "overview.pt", weights_only=False)
    assert set(overview["summaries"]) == expected_pairs
    assert overview["figure_pairs"] == ns["figure_pairs"]
    ablation = overview["ablation"]
    assert ablation["run"]["ablation"] == settings["ablation"]
    assert ablation["run"]["forward_hooks"] == [settings["ablation"]["hook"]]
    control = overview["run"][("retrofitted_llama", 32)]
    assert ablation["run"]["indices"] == control["indices"]
    assert ablation["run"]["tokenization"] == control["tokenization"]
    assert len(ablation["summaries"]) == limit
    assert all(
        "residual_entropy" not in example["curves"] for example in ablation["summaries"]
    )
    ablated = ns["ablation_stages"]
    axes = panels(figures["ablation_retrofitted_llama_bos_mlp"])
    assert len(axes) == 3
    for ax, field in zip(axes, ns["ATTENTION_METRICS"], strict=True):
        assert ax.get_ylabel() == ns["ATTENTION_METRICS"][field]
        for line, group in zip(
            ax.lines[:-1], ns["recurrence_groups"](ablated).values(), strict=True
        ):
            np.testing.assert_allclose(
                line.get_xdata(), ns["depth_positions"](ablated, by_layer=True)[group]
            )
            np.testing.assert_allclose(
                line.get_ydata(), ablated["curves"][field]["mean"][group]
            )
    indices = []
    for pair, info in overview["run"].items():
        assert (
            info["compute_residual_entropy"]
            is ns["configs"][pair].compute_residual_entropy
        )
        assert not info["compute_differences"] and not info["compute_pairwise"]
        assert not info["forward_hooks"]
        indices.append(info["indices"])
        for example in overview["summaries"][pair]:
            assert ("residual_entropy" in example["curves"]) is info[
                "compute_residual_entropy"
            ]
        for curve in ns["stages"][pair]["curves"].values():
            assert np.isfinite(curve["mean"]).all() and np.isfinite(curve["std"]).all()
            if limit == 1:
                assert np.all(curve["std"] == 0)
    assert all(index_list == indices[0] for index_list in indices)

    def unexpected_collection(*args, **kwargs):
        pytest.fail("Rerunning figure cells repeated inference")

    ns["collect_metrics"] = unexpected_collection
    # The full set of figure cells can rerun directly from saved summaries.
    plotting_started = False
    for cell in cells():
        if cell["id"] == "overall-attention":
            plotting_started = True
        if (
            plotting_started
            and cell["cell_type"] == "code"
            and cell["id"] != "f956f950"
        ):
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), ns)
    assert local_inference == {"loads": 20, "forwards": 20 * limit}
