"""Check paper sizing against the PDF page, including the notebook's layouts."""

import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.ticker import MaxNLocator

from experiments import support
from experiments.paper_models import paper_model_pairs

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "fullpage, subplots, height, fraction, expected",
    [
        (False, (1, 1), 1, 1, (3.25, 1)),
        (True, (1, 3), 2, 1, (6.75, 2)),
        (True, (2, 3), 2, 1, (6.75, 4)),
        (True, (1, 4), 3, 1, (6.75, 3)),
        (False, (2, 2), 0.75, 0.5, (1.625, 1.5)),
    ],
)
def test_paper_figsize(fullpage, subplots, height, fraction, expected):
    assert support.get_figsize(fullpage, subplots, height, fraction) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "group, selection",
    [
        ("main", None),
        ("appendix", None),
        ("all", None),
        ("main", ["raven"]),
        (
            "all",
            [
                "ouro_1_4b",
                "raven",
                "retrofitted_llama",
                "parcae_370m",
                "retrofitted_olmo",
            ],
        ),
    ],
)
def test_cyclic_similarity_paper_pdfs(tmp_path, monkeypatch, group, selection):
    """Check square panels, independent scales, row labels, and actual PDF sizes."""
    notebook = ROOT / "experiments" / "cyclic_similarity.ipynb"
    pairs, figure_pairs = paper_model_pairs(
        8, None if selection is None else [(name, 8) for name in selection]
    )
    namespace = {
        "ROOT": ROOT,
        "plt": plt,
        "np": np,
        "MaxNLocator": MaxNLocator,
        "COLORMAPS": support.COLORMAPS,
        "MODEL_LABELS": support.MODEL_LABELS,
        "model_subplots": support.model_subplots,
        "shorten_label": support.shorten_label,
        "figure_pairs": figure_pairs,
        "run_dir": tmp_path,
    }
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if (
                isinstance(node, ast.FunctionDef) and node.name == "plot_similarity"
            ) or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "PAIRWISE_METRICS"
                    for target in node.targets
                )
            ):
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]), str(notebook), "exec"
                    ),
                    namespace,
                )

    figures = {}

    def save_figure(fig, directory, name, **kwargs):
        support.save_figure(fig, directory, name, **kwargs)
        figures[name] = fig

    namespace["save_figure"] = save_figure
    monkeypatch.setattr(plt, "show", lambda: None)
    data = {}
    for i, pair in enumerate(pairs):
        size = [192, 32, 48, 64, 32, 32, 64, 48][i]
        distance = np.abs(np.arange(size)[:, None] - np.arange(size)[None, :]) / (
            size - 1
        )
        data[pair] = {
            metric: 1 - 2 * distance
            if metric.endswith("cosine_sim")
            else distance * (i + 1) ** 3
            for metric in namespace["PAIRWISE_METRICS"]
        }
    namespace["summaries"] = data
    with plt.rc_context({"font.size": 17, "savefig.bbox": "tight"}):
        for metric in namespace["PAIRWISE_METRICS"]:
            namespace["plot_similarity"](metric, group=group)
        assert plt.rcParams["font.size"] == 17
        assert plt.rcParams["savefig.bbox"] == "tight"

    selected = figure_pairs[group]
    rows = (len(selected) + 3) // 4
    height = rows * 1.75
    assert len(figures) == len(namespace["PAIRWISE_METRICS"]) == 3
    for metric, label in namespace["PAIRWISE_METRICS"].items():
        name = f"{metric}_{group}"
        fig = figures[name]
        assert fig.get_size_inches() == pytest.approx((6.75, height))
        heatmaps = [ax for ax in fig.axes if ax.images]
        colorbars = [ax for ax in fig.axes if not ax.images]
        assert len(heatmaps) == len(colorbars) == len(selected)
        assert [ax.get_title() for ax in heatmaps] == [
            support.MODEL_LABELS[name] for name, _ in selected
        ]
        assert sum(bool(ax.get_ylabel()) for ax in heatmaps) == rows
        assert sum(bool(ax.get_ylabel()) for ax in colorbars) == rows
        assert {ax.get_ylabel() for ax in colorbars if ax.get_ylabel()} == {
            support.shorten_label(label)
        }
        assert all(ax.title.get_fontsize() == 6 for ax in heatmaps)
        assert all(ax.xaxis.label.get_fontsize() == 6 for ax in heatmaps)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = fig.get_tightbbox(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= 6.75 and bounds.y1 <= height
        for ax, pair in zip(heatmaps, selected, strict=True):
            np.testing.assert_array_equal(ax.images[0].get_array(), data[pair][metric])
            expected = (
                (-1, 1)
                if metric.endswith("cosine_sim")
                else (0, data[pair][metric].max())
            )
            assert ax.images[0].get_clim() == pytest.approx(expected)
            panel = ax.get_window_extent(renderer)
            assert panel.width == pytest.approx(panel.height)
        assert len({id(ax.images[0].norm) for ax in heatmaps}) == len(selected)

        media_box = re.search(
            rb"/MediaBox\s*\[([^]]+)\]", (tmp_path / f"{name}.pdf").read_bytes()
        )
        assert media_box is not None
        assert [float(value) for value in media_box[1].split()] == pytest.approx(
            [0, 0, 6.75 * 72, height * 72]
        )
    assert not list(tmp_path.glob("*.png"))
