"""Check the mixing-correlation reduction on synthetic profiles and run all cells."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_notebooks import notebook_models  # noqa: F401  (pytest fixture)

from experiments import support

NOTEBOOK = Path(__file__).parents[1] / "experiments" / "mixing_correlation.ipynb"


def cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def science():
    """Execute only the reduction definitions."""
    namespace = {}
    for cell in cells():
        if cell["id"] in {"root", "imports-and-configuration", "reduction"}:
            source = "".join(cell["source"])
            source = source.split("\nmodel_recurrences =")[0]  # Imports only.
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_profiles_keep_prelude_and_coda_around_each_recurrence():
    ns = science()
    # Prelude 0-1, core 2-3 applied three times, coda 4.
    layers = [0, 1, 2, 3, 2, 3, 2, 3, 4]
    scores = [0.0, 1.0, 20.0, 30.0, 21.0, 31.0, 22.0, 32.0, 4.0]
    example = {
        "layers": layers,
        "recurrent_layers": [2, 3],
        "curves": {"mixing_scores": scores},
    }
    np.testing.assert_array_equal(
        ns["recurrence_profiles"](example, 2),
        [[0, 1, 20, 30, 4], [0, 1, 21, 31, 4]],
    )
    # Without a prelude or coda, each profile is one recurrence.
    example = {
        "layers": [0, 1, 0, 1],
        "recurrent_layers": [0, 1],
        "curves": {"mixing_scores": [1.0, 2.0, 3.0, 4.0]},
    }
    np.testing.assert_array_equal(
        ns["recurrence_profiles"](example, 8), [[1, 2], [3, 4]]
    )


def test_control_interpolation_and_correlation_summaries():
    ns = science()
    control = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
    np.testing.assert_allclose(
        ns["control_at_depths"](control, 5),
        [[0, 0.5, 1, 1.5, 2], [0, 1, 2, 3, 4]],
    )
    np.testing.assert_array_equal(ns["control_at_depths"](control, 3), control)

    control = np.array([[0.1, 0.4, 0.2, 0.9], [0.3, 0.1, 0.8, 0.5]])
    # Recurrence 1: affine copy; 2: monotone, nonlinear; 3: reversed sign.
    looped = np.stack([control * 2 + 1, np.exp(5 * control), -control], axis=1)
    pearson = ns["correlations"](looped, control, "pearson")
    spearman = ns["correlations"](looped, control, "spearman")
    assert pearson["per_example"].shape == (2, 4)
    np.testing.assert_allclose(pearson["per_example"][:, [0, 2]], [[1, -1], [1, -1]])
    assert np.all(pearson["per_example"][:, 1] < 1)
    np.testing.assert_allclose(spearman["per_example"][:, :3], [[1, 1, -1]] * 2)
    for values in (pearson, spearman):
        np.testing.assert_allclose(
            values["per_example"][:, -1], values["per_example"][:, :3].mean(axis=1)
        )
        assert values["of_means"][-1] == pytest.approx(values["of_means"][:3].mean())
    # Correlating mean profiles is not the mean of per-example correlations.
    np.testing.assert_allclose(pearson["of_means"][[0, 2]], [1, -1])


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_notebook_executes_without_repeating_inference(
    notebook_models,  # noqa: F811
    tmp_path,
    monkeypatch,
    mode,
):
    paper_root = tmp_path / "paper-plots"
    monkeypatch.setattr(support, "PAPER_PLOTS_DIR", paper_root)
    collected = []
    collect_metrics = support.collect_metrics

    def record_collection(config, summarize, **kwargs):
        assert config.experiment == "stages_of_inference"
        assert summarize is support.summarize_stages
        collected.append((config.models[0], config.num_steps))
        return collect_metrics(config, summarize, **kwargs)

    monkeypatch.setattr(support, "collect_metrics", record_collection)
    namespace = {}
    for cell in cells():
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(mode=mode, limit=2, device="cpu", output_dir=str(tmp_path))
    pairs = namespace["model_recurrences"]
    controls = [("llama_1b", 1), ("parcae_770m_rec1", 1)]
    assert collected == [*pairs, *controls]
    directory = namespace["run_dir"]
    assert directory.name.startswith(f"{mode}-")
    assert directory.parent == tmp_path / "mixing_correlation" / "gsm8k"
    figures = {"mixing_score_pearson.pdf", "mixing_score_spearman.pdf"}
    assert {path.name for path in directory.glob("*.pdf")} == figures
    if mode == "quick":
        assert not paper_root.exists()
    else:
        exported = paper_root / "mixing_correlation" / "gsm8k"
        assert {path.name for path in exported.glob("*.pdf")} == figures

    overview = torch.load(directory / "overview.pt", weights_only=False)
    for statistic in ("pearson", "spearman"):
        for pair in pairs:
            values = overview["correlation"][statistic][pair]
            shown = min(pair[1], namespace["shown_recurrences"])
            assert values["of_means"].shape == (shown + 1,)
            assert values["per_example"].shape == (2, shown + 1)
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["experiment"] == "mixing_correlation"
    assert metadata["mode"] == mode and metadata["limit"] == 2
    assert (directory / "correlations.csv").exists()

    def unexpected_collection(*args, **kwargs):
        pytest.fail("A figure cell repeated inference")

    namespace["collect_metrics"] = unexpected_collection
    for cell in cells():
        source = "".join(cell["source"])
        if source.startswith("plot_correlations("):
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)
