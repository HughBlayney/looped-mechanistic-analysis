"""Check the trajectory notebook's PCA semantics, figures, and export policy."""

import json
import re
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from experiments import support
from experiments.paper_models import MAIN_PARCAE_MODEL, PAPER_MODEL_GROUPS
from looped_internals import models
from looped_internals.models import MinimalRecurrentConfig, ModelSpec
from looped_internals.results import ModelRunData, PerVirtualLayerMetrics

NOTEBOOK = Path(__file__).parents[1] / "experiments" / "trajectories.ipynb"


def cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def setup(**parameters):
    namespace = {}
    for cell in cells():
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            assert not {"models", "num_steps", "limit", "dataset"} & namespace.keys()
            namespace.update(parameters)
        if "capture_trajectories" in namespace:
            return namespace
    raise AssertionError("Missing trajectory definitions")


def test_trajectory_modes_keep_the_same_required_runs(tmp_path, caplog, monkeypatch):
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
    expected = [(name, 32) for name in PAPER_MODEL_GROUPS["main"]]
    assert list(quick["configs"]) == list(full["configs"]) == expected
    assert quick["config"].mode == "quick" and full["config"].mode == "full"
    assert quick["random_configs"] == full["random_configs"]
    assert (
        quick["gsm8k_token_fractions"] == full["gsm8k_token_fractions"] == [0, 0.5, 1]
    )
    assert quick["plot_recurrences"] == full["plot_recurrences"] == 32
    relu = quick["random_configs"][("relu_no_injection", 64)]
    gated = quick["random_configs"][("gated_input_injection", 64)]
    assert (relu.d_model, relu.n_heads, relu.n_layers) == (512, 4, 24)
    assert relu.activation == "relu" and not relu.gated_mlp
    assert relu.input_injection is None
    assert gated.n_layers == 12 and gated.gated_mlp
    assert gated.input_injection == "projection"
    assert "unknown parameter" not in caplog.text

    # The central Parcae switch must reach all trajectory figure loops.
    main = (*PAPER_MODEL_GROUPS["main"][:3], "parcae_770m")
    monkeypatch.setitem(PAPER_MODEL_GROUPS, "main", main)
    overridden = setup(recurrences=8)
    assert overridden["model_recurrences"] == [(name, 8) for name in main]
    explicit = setup(model_recurrences=[["parcae_140m", 16]])
    assert list(explicit["configs"]) == [("parcae_140m", 16)]


def test_trajectory_pca_matches_per_token_library_projection():
    namespace = setup()
    generator = torch.Generator().manual_seed(5)
    layers = [0, *([1, 2] * 6), 3]
    blocks = [
        PerVirtualLayerMetrics(
            layer_index=layer,
            virtual_layer_index=i,
            sink_scores=None,
            mixing_scores=None,
            colsum_concentrations=None,
            residual_norms=torch.zeros(1),
            raw_residual=torch.randn(1, 2, 5, generator=generator) + i,
        )
        for i, layer in enumerate(layers)
    ]
    run = ModelRunData(
        input_ids=torch.tensor([[1, 2]]), per_virtual_layer_metrics=blocks
    )
    groups = run.get_pca_residual_trajectories(
        num_dimensions=2, separate_across_tokens=True, group_by="recurrence"
    )
    expected = np.stack([v[:, -1] for k, v in groups.items() if k.startswith("R")])
    summary = namespace["summarize_trajectory"](run, -1)
    np.testing.assert_allclose(summary["coordinates"], expected, atol=1e-5)
    assert summary["layers"] == [1, 2]
    assert summary["coordinates"].shape == (6, 2, 2)
    assert summary["token_position"] == 1
    # Other token positions must not change the selected token's PCA basis.
    for block in blocks:
        block.raw_residual[:, 0] *= 1000
    np.testing.assert_array_equal(
        namespace["summarize_trajectory"](run, -1)["coordinates"],
        summary["coordinates"],
    )


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_trajectory_notebook_generates_only_requested_figures(
    tmp_path, monkeypatch, mode
):
    torch.set_num_threads(1)
    original_load = models.load_model
    loads, forwards, figures = [], [], []

    class Tokenizer:
        eos_token_id = 0
        bos_token_id = 1

        def encode(self, text, add_special_tokens=True, return_tensors=None):
            tokens = ([1] if add_special_tokens else []) + [
                2 + sum(map(ord, word)) % 60 for word in text.split()
            ]
            return torch.tensor([tokens]) if return_tensors else tokens

        def __call__(self, text, return_offsets_mapping=False):
            return {
                "input_ids": self.encode(text),
                "offset_mapping": [(0, 0)]
                + [match.span() for match in re.finditer(r"\S+", text)],
            }

        def decode(self, ids):
            return f"token-{ids[0]}"

    def load_dataset(path, name, **kwargs):
        assert (path, name) == ("openai/gsm8k", "main")
        examples = datasets.Dataset.from_list(
            [
                {
                    "question": f"What is {index} plus another three apples?",
                    "answer": "There are three more apples.\n#### 3",
                }
                for index in range(1319)
            ]
        )
        return datasets.DatasetDict(train=examples, test=examples)

    def record_forward(bridge):
        original_run = bridge.run_with_cache

        def run(*args, **kwargs):
            forwards.append(kwargs)
            assert kwargs["accumulate"] and kwargs["device"] == "cpu"
            assert all(
                name.endswith(".hook_out") and "attn" not in name
                for name in kwargs["names_filter"]
            )
            return original_run(*args, **kwargs)

        bridge.run_with_cache = run
        return bridge

    def pretrained(name, config):
        loads.append((name, config.num_steps))
        bridge, structure = original_load(
            ModelSpec(
                MinimalRecurrentConfig(
                    d_model=16,
                    n_heads=4,
                    n_layers=2,
                    num_cycles=config.num_steps,
                    d_vocab=64,
                )
            ),
            device="cpu",
        )
        bridge.tokenizer = Tokenizer()
        return record_forward(bridge), structure

    def random_model(spec, **kwargs):
        loads.append((spec.source.n_layers, spec.source.num_cycles))
        bridge, structure = original_load(spec, **kwargs)
        return record_forward(bridge), structure

    original_save = support.save_figure

    def save(fig, directory, name, **kwargs):
        figures.append((name, fig))
        original_save(fig, directory, name, **kwargs)

    monkeypatch.setattr(support, "load_experiment_model", pretrained)
    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    monkeypatch.setattr(models, "load_model", random_model)
    monkeypatch.setattr(support, "save_figure", save)
    monkeypatch.setattr(plt, "show", lambda: None)
    paper_root = tmp_path / "paper"
    monkeypatch.setattr(support, "PAPER_PLOTS_DIR", paper_root)
    paper_dir = paper_root / "trajectories"
    paper_dir.mkdir(parents=True)
    existing = paper_dir / "pca_quartiles_Ouro 1.4B (8 steps).pdf"
    existing.write_bytes(b"Previous paper figure")

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
                recurrences=8,
                random_model_recurrences=[
                    (name, 8) for name, _ in namespace["random_model_recurrences"]
                ],
                random_width=16,
                random_sequence_length=6,
                plot_recurrences=4,
            )
    directory = namespace["run_dir"]
    assert directory.parent.name == "fixed_sequence"
    assert directory.name.startswith(f"{mode}-")
    assert loads[:4] == [(name, 8) for name in PAPER_MODEL_GROUPS["main"]]
    assert len(loads) == 6
    assert len(forwards) == 18  # Four sequences per pretrained model; two random.
    assert loads[-2:] == [(24, 8), (12, 8)]
    assert len(figures) == len(list(directory.glob("*.pdf"))) == 14
    assert (
        sum(support.MODEL_LABELS[MAIN_PARCAE_MODEL] in name for name, _ in figures) == 3
    )
    assert not list(directory.glob("*floorplan*"))
    overview = torch.load(directory / "overview.pt", weights_only=False)
    summaries = overview["summaries"]
    assert len(summaries) == 6
    assert (MAIN_PARCAE_MODEL, 8) in overview["gsm8k"]
    assert summaries[("relu_no_injection", 8)]["coordinates"].shape == (8, 24, 2)
    assert summaries[("gated_input_injection", 8)]["coordinates"].shape == (8, 12, 2)
    assert (
        summaries[("relu_no_injection", 8)]["input_ids"]
        == summaries[("gated_input_injection", 8)]["input_ids"]
    )
    metadata = json.loads((directory / "run.json").read_text())
    assert (
        metadata["mode"] == mode and "limit" not in metadata and "task" not in metadata
    )
    assert metadata["gsm8k"]["indices"] == [588, 1214, 57]
    assert metadata["model_recurrences"] == [
        [name, 8] for name in PAPER_MODEL_GROUPS["main"]
    ]
    assert metadata["gsm8k"]["token_fractions"] == [0, 0.5, 1]
    for examples in overview["gsm8k"].values():
        assert [example["index"] for example in examples] == [588, 1214, 57]
        for example in examples:
            trajectories = example["trajectories"]
            assert len(trajectories) == 3
            assert [t["token_position"] for t in trajectories] == [2, 5, 8]
            tokenizer = Tokenizer()
            context_ids = tokenizer.encode(example["context"])
            whole_ids = tokenizer.encode(example["context"] + example["answer"])
            expected_ids = context_ids + whole_ids[len(context_ids) :]
            for trajectory in trajectories:
                assert trajectory["input_ids"] == [expected_ids]
                position = trajectory["token_position"]
                assert trajectory["token_text"] == tokenizer.decode(
                    [expected_ids[position]]
                )
                assert trajectory["coordinates"].shape == (8, 2, 2)
            assert not np.array_equal(
                trajectories[0]["coordinates"], trajectories[-1]["coordinates"]
            )

    pairs = list(namespace["configs"]) * 2 + list(namespace["random_configs"])
    for (filename, fig), pair in zip(figures[: len(pairs)], pairs, strict=True):
        panels = 4 if filename.endswith("_extended") else 2
        width = 6.75 if panels == 4 else 3.25
        height = 2 if filename == "initialised_injected_trajectory" else 1.5
        assert fig.get_size_inches() == pytest.approx((width, height))
        box = re.search(
            rb"/MediaBox\s*\[([^]]+)\]", (directory / f"{filename}.pdf").read_bytes()
        )
        assert [float(v) for v in box[1].split()] == pytest.approx(
            [0, 0, width * 72, height * 72]
        )
        axes = fig.axes[:panels]
        lines = [line for ax in axes for line in ax.lines]
        assert len(lines) == 4  # Display truncation happens after the full PCA fit.
        for line, coordinates in zip(
            lines, summaries[pair]["coordinates"][:4], strict=True
        ):
            np.testing.assert_allclose(
                np.column_stack([line.get_xdata(), line.get_ydata()]), coordinates
            )
        assert all(ax.get_xlim() == axes[0].get_xlim() for ax in axes)
        assert all(ax.get_ylim() == axes[0].get_ylim() for ax in axes)
        if panels == 2:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            plot_right = max(ax.get_window_extent(renderer).x1 for ax in axes)
            colorbar = fig.axes[panels]
            assert colorbar.get_xlabel() == "Early"
            assert colorbar.get_title() == "Late"
            assert colorbar.xaxis.label.get_window_extent(renderer).x0 > plot_right
            assert colorbar.title.get_window_extent(renderer).x0 > plot_right

    for (filename, fig), pair in zip(
        figures[len(pairs) :], namespace["configs"], strict=True
    ):
        assert filename.startswith("pca_gsm8k_grid_")
        assert fig.get_size_inches() == pytest.approx((6.75, 5.4))
        assert len(fig.axes) == 19  # Nine pairs plus the shared colour bar.
        for panel, trajectory in zip(
            fig.subfigs,
            [t for example in overview["gsm8k"][pair] for t in example["trajectories"]],
            strict=True,
        ):
            axes = panel.axes
            assert len(axes) == 2
            assert [ax.get_title() for ax in axes] == [
                "Recurrences 1-2",
                "Recurrences 3-4",
            ]
            assert axes[0].get_xlim() == axes[1].get_xlim()
            assert axes[0].get_ylim() == axes[1].get_ylim()
            lines = [line for ax in axes for line in ax.lines]
            for line, coordinates in zip(
                lines, trajectory["coordinates"][:4], strict=True
            ):
                np.testing.assert_allclose(line.get_xydata(), coordinates)

    if mode == "quick":
        assert list(paper_dir.iterdir()) == [existing]
        assert existing.read_bytes() == b"Previous paper figure"
    else:
        assert {p.name for p in paper_dir.iterdir()} == {
            p.name for p in directory.glob("*.pdf")
        }
        for pdf in directory.glob("*.pdf"):
            assert (paper_dir / pdf.name).read_bytes() == pdf.read_bytes()
    # Every figure cell can be rerun without reloading models or running inference.
    for cell in cells():
        if cell["cell_type"] == "code" and cell["id"] in {
            "compact-figures",
            "extended-figures",
            "relu-figure",
            "injected-figure",
            "gsm8k-grid-figures",
        }:
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
    assert len(loads) == 6 and len(forwards) == 18
