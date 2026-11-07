"""Scientific regressions and local execution of the initialisation figures."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments import support
from looped_internals import models
from looped_internals.models.minimal import MinimalRecurrentConfig, MinimalRecurrentLM
from looped_internals.results import ModelRunData, PerVirtualLayerMetrics

NOTEBOOK = Path(__file__).parents[1] / "experiments" / "initialisation.ipynb"


def cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def setup(**parameters):
    namespace = {}
    for cell in cells():
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            assert not {"models", "limit", "dataset"} & namespace.keys()
            namespace.update(parameters)
        if "recurrent_jacobian" in namespace:
            return namespace
    raise AssertionError("Missing scientific definitions")


def test_initialisation_modes_and_papermill_defaults(tmp_path, caplog):
    papermill = pytest.importorskip("papermill")
    prepared = papermill.execute_notebook(
        NOTEBOOK,
        tmp_path / "prepared.ipynb",
        parameters={"mode": "full"},
        prepare_only=True,
    )
    injected = next(
        cell
        for cell in prepared.cells
        if "injected-parameters" in cell.metadata.get("tags", [])
    )
    parameters = {}
    exec(injected.source, parameters)
    quick, full = setup(), setup(**parameters)
    assert quick["seeds"] == [42]
    assert full["seeds"] == list(range(42, 47))
    assert quick["stability_configs"] == full["stability_configs"]
    assert quick["jacobian_configs"] == full["jacobian_configs"]
    assert quick["scale_configs"] == full["scale_configs"]
    assert quick["projection_config"] == full["projection_config"]
    assert quick["summary_layers"] == full["summary_layers"] == 12
    assert len(quick["stability_configs"]) == 24
    for (depth, _, injection), config in quick["stability_configs"].items():
        assert config.n_layers == depth
        assert depth in [4, 8, 12, 16]
        assert config.num_cycles == 128 and config.d_model == 512
        assert config.n_heads == 4 and config.input_injection == injection
        assert config.gated_mlp
    for config in quick["jacobian_configs"].values():
        assert config.n_layers == 1 and not config.norm_post_cycle
    assert list(quick["scale_configs"]) == [0, 0.5]
    for scale, config in quick["scale_configs"].items():
        assert config.n_layers == 6 and config.num_cycles == 256
        assert config.residual_scaling == scale and config.input_injection is None
        assert not config.gated_mlp and config.activation == "relu"
    assert "unknown parameter" not in caplog.text


def test_reference_reduction_uses_layer_metadata_and_worst_mean():
    namespace = setup()
    layers = [0, 1, 2, 1, 2, 1, 2, 3]  # Non-recurrent prelude and coda.
    records = [
        PerVirtualLayerMetrics(
            layer_index=layer,
            virtual_layer_index=i,
            sink_scores=None,
            mixing_scores=None,
            colsum_concentrations=None,
            residual_norms=torch.tensor([float(i)]),
        )
        for i, layer in enumerate(layers)
    ]
    values = torch.arange(8 * 4 * 2, dtype=torch.float32).reshape(8, 4, 2)
    run = ModelRunData(
        input_ids=torch.tensor([[1, 2]]),
        per_virtual_layer_metrics=records,
        to_final_state_residual_cosine_sim=values,
        to_final_state_residual_diff_norm=values + 10,
        successive_recurrence_residual_diff_norm=[
            torch.empty(0, 2),
            torch.ones(2, 2),
            torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
            torch.empty(0, 2),
        ],
    )
    summary = namespace["summarize_stability"](run)
    assert summary["layers"] == [1, 2]
    torch.testing.assert_close(
        summary["curves"]["cosine_sim"],
        values[[1, 3, 5]][:, [1, 2]].mean(-1),
    )
    assert summary["last_step_distance"] == 6
    torch.testing.assert_close(summary["residual_norms"], torch.arange(8).float())

    first = torch.tensor([[0.7, 0.1], [0.8, 0.2], [1.0, 0.3]])
    second = torch.tensor([[0.2, 0.9], [0.6, 0.5], [1.0, 0.4]])
    examples = [{"curves": {"cosine_sim": v, "diff_norm": v}} for v in [first, second]]
    own, worst = namespace["reference_curves"](examples, "cosine_sim")
    expected = torch.stack([first, second])[:, torch.arange(3), torch.tensor([0, 1, 1])]
    torch.testing.assert_close(own[0], torch.stack([first[:, 0], second[:, 0]]).mean(0))
    torch.testing.assert_close(worst[0], expected.mean(0))
    torch.testing.assert_close(worst[1], expected.std(0))
    _, farthest = namespace["reference_curves"](examples, "diff_norm")
    expected = torch.stack([first, second])[:, torch.arange(3), torch.tensor([1, 0, 0])]
    torch.testing.assert_close(farthest[0], expected.mean(0))
    _, single = namespace["reference_curves"](examples[:1], "cosine_sim")
    assert not single[1].any()


@pytest.mark.parametrize("injection", [None, "projection", "sum"])
def test_recurrent_jacobian_is_latent_derivative_and_matches_forward(injection):
    torch.set_num_threads(1)
    namespace = setup()
    torch.manual_seed(13)
    config = MinimalRecurrentConfig(
        d_model=4,
        n_heads=2,
        n_layers=2,
        num_cycles=3,
        d_vocab=8,
        input_injection=injection,
        residual_scaling=0.5,
    )
    model = MinimalRecurrentLM(config).double().eval()
    x = torch.randn(1, 3, 4, dtype=torch.float64)
    z = torch.randn_like(x)
    actual = namespace["recurrent_jacobian"](model, x, z)
    # Compare against central differences on the full sequence, holding x fixed.
    epsilon = 1e-5
    columns = []
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        for coordinate in range(4):
            delta = torch.zeros_like(z)
            delta[0, 0, coordinate] = epsilon
            columns.append(
                (
                    (
                        model.recurrent_step(z + delta, x)
                        - model.recurrent_step(z - delta, x)
                    )
                    / (2 * epsilon)
                )[0, 0]
            )
    torch.testing.assert_close(
        actual, torch.stack(columns, dim=1), atol=1e-7, rtol=1e-6
    )
    # The extracted step must preserve the model's complete recurrent forward pass.
    torch.manual_seed(10)
    expected = model.forward_no_logits(x)
    torch.manual_seed(10)
    state = (
        torch.randn_like(x) * config.initialization_variance if injection else x.clone()
    )
    for _ in range(config.num_cycles):
        state = config.residual_scaling * state
        if injection == "projection":
            state = model.input_proj(torch.cat([x, state], dim=-1))
        elif injection == "sum":
            state = x + state
        for block in model.transformer_layers:
            state = block(state)
        state = model.cycle_norm(state)
    torch.testing.assert_close(state, expected)


def test_projection_jacobian_uses_latent_half_of_weight():
    namespace = setup()
    config = MinimalRecurrentConfig(
        d_model=4,
        n_heads=2,
        n_layers=0,
        d_vocab=8,
        input_injection="projection",
        norm_post_cycle=False,
        residual_scaling=0.5,
    )
    model = MinimalRecurrentLM(config)
    with torch.no_grad():
        model.input_proj.weight[:, :4].copy_(7 * torch.eye(4))
        model.input_proj.weight[:, 4:].copy_(2 * torch.eye(4))
    x, z = torch.randn(1, 3, 4), torch.randn(1, 3, 4)
    actual = namespace["recurrent_jacobian"](model, x, z)
    torch.testing.assert_close(actual, torch.eye(4))
    # This differs from the embedding derivative, 7I, even at the same input/state.
    assert not torch.allclose(actual, model.input_proj.weight[:, :4])


@pytest.mark.parametrize("mode, count", [("quick", 1), ("full", 5)])
def test_initialisation_notebook_figures_measurements_and_exports(
    tmp_path, monkeypatch, mode, count
):
    torch.set_num_threads(1)
    loads, forwards, figures = [], [], []
    original_load, original_save = models.load_model, support.save_figure

    def load(spec, **kwargs):
        loads.append(spec.source)
        bridge, structure = original_load(spec, **kwargs)
        original_run = bridge.run_with_cache

        def run(tokens, **run_kwargs):
            forwards.append((spec.source, tokens.cpu().clone()))
            assert run_kwargs["names_filter"] == structure.hook_names()
            assert run_kwargs["device"] == "cpu" and run_kwargs["accumulate"]
            return original_run(tokens, **run_kwargs)

        bridge.run_with_cache = run
        return bridge, structure

    def save(fig, directory, name, **kwargs):
        figures.append((name, fig))
        original_save(fig, directory, name, **kwargs)

    monkeypatch.setattr(models, "load_model", load)
    monkeypatch.setattr(support, "save_figure", save)
    monkeypatch.setattr(plt, "show", lambda: None)
    paper_root = tmp_path / "paper"
    monkeypatch.setattr(support, "PAPER_PLOTS_DIR", paper_root)
    paper_dir = paper_root / "initialisation"
    paper_dir.mkdir(parents=True)
    existing = paper_dir / "summarised_stability.pdf"
    existing.write_bytes(b"Existing full result")
    unrelated = paper_dir / "historical.pdf"
    unrelated.write_bytes(b"Keep historical artifacts")

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
                width=8,
                num_heads=2,
                sequence_length=3,
                vocab_size=16,
                recurrences=4,
                residual_scale_recurrences=4,
            )
    directory = namespace["run_dir"]
    assert directory.parent.name == "random_initialisation"
    assert directory.name.startswith(f"{mode}-")
    assert len(loads) == 29 * count  # 24 stability, 2 Jacobian, 2 scales, 1 projection.
    assert len(forwards) == 26 * count
    depth_counts = Counter(c.n_layers for c, _ in forwards if c.gated_mlp)
    assert depth_counts == dict.fromkeys([4, 8, 12, 16], 6 * count)
    overview = torch.load(directory / "overview.pt", weights_only=False)
    for key, examples in overview["stability"].items():
        assert len(examples) == count
        for example in examples:
            assert len(example["layers"]) == key[0]
            assert example["curves"]["cosine_sim"].shape == (4, key[0])
            assert example["curves"]["cosine_sim"][-1, 0].item() == pytest.approx(
                1, abs=1e-6
            )
            assert example["curves"]["diff_norm"][-1, 0].item() == pytest.approx(
                0, abs=1e-6
            )
    assert all(len(v) == count for v in overview["spectra"].values())
    assert len(pd.read_csv(directory / "residual_scaling.csv")) == 2 * count
    projection = pd.read_csv(directory / "projection_measurements.csv")
    assert len(projection) == count
    assert (projection.spectral_norm >= projection.spectral_radius).all()
    np.testing.assert_allclose(
        projection.iid_radius_estimate, projection.weight_std * np.sqrt(8)
    )
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["seeds"] == list(range(42, 42 + count))
    assert metadata["mode"] == mode and metadata["recurrences"] == 4
    assert (
        "limit" not in metadata and "task" not in metadata and "models" not in metadata
    )
    # The same seed's random sequence is reused across all stability configurations.
    for offset in range(0, 24 * count, count):
        for i in range(count):
            torch.testing.assert_close(
                forwards[offset + i][1], overview["inputs"][42 + i]
            )

    assert len(figures) == len(list(directory.glob("*.pdf"))) == 4
    assert [name for name, _ in figures] == [
        "stability_initialised_to_final_state_residual_cosine_sim",
        "stability_initialised_to_final_state_residual_diff_norm",
        "stability_initialised_to_final_state_residual_diff_norm_all_layers",
        "summarised_stability",
    ]
    assert [len(fig.axes) for _, fig in figures] == [6, 6, 6, 3]
    for i, (_, fig) in enumerate(figures):
        np.testing.assert_allclose(fig.get_size_inches(), [6.75, 3 if i < 3 else 2])
    depth_figure = figures[2][1]
    assert [text.get_text() for text in depth_figure.legends[0].get_texts()] == [
        "4 layers",
        "8 layers",
        "12 layers",
        "16 layers",
    ]
    for ax, (injection, norm) in zip(
        depth_figure.axes,
        [
            (injection, norm)
            for injection in namespace["INJECTIONS"]
            for norm in namespace["NORMALISATIONS"]
        ],
        strict=True,
    ):
        assert len(ax.lines) == len(ax.collections) == 4
        assert len({line.get_color() for line in ax.lines}) == 4
        for index, (line, depth) in enumerate(
            zip(ax.lines, [4, 8, 12, 16], strict=True)
        ):
            assert line.get_marker() in {None, "None", ""}
            assert line.get_color() == f"C{index}"
            expected = torch.stack(
                [
                    e["curves"]["diff_norm"][:, 0]
                    for e in overview["stability"][depth, norm, injection]
                ]
            ).mean(0)
            np.testing.assert_allclose(line.get_ydata(), expected)
    summary = figures[-1][1]
    assert [ax.get_title() for ax in summary.axes] == [
        "Same Layer",
        "Residual Norms",
        "Eigenvalues at Init",
    ]
    expected, _ = namespace["reference_curves"](
        namespace["stability"][12, "Pre", "projection"], "cosine_sim"
    )
    np.testing.assert_allclose(summary.axes[0].lines[0].get_ydata(), expected[0])
    expected_norms = torch.stack(
        [e["residual_norms"] for e in overview["stability"][12, "Pre", "projection"]]
    ).mean(0)
    np.testing.assert_allclose(summary.axes[1].lines[0].get_ydata(), expected_norms)
    for collection in summary.axes[2].collections:
        assert collection.get_offsets().shape == (8 * count, 2)

    if mode == "quick":
        assert namespace["paper_dir"] is None
        assert existing.read_bytes() == b"Existing full result"
    else:
        assert namespace["paper_dir"] == paper_dir
        assert existing.read_bytes() == (directory / existing.name).read_bytes()
    assert unrelated.read_bytes() == b"Keep historical artifacts"

    # Replotting the completed summaries never reloads a model or reruns inference.
    before = len(loads), len(forwards)
    namespace["plot_summary"]("rerender")
    assert (len(loads), len(forwards)) == before
