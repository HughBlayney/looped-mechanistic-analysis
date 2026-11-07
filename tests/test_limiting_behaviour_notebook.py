"""Check limiting-behaviour reductions, selective collection, and paper exports."""

import json
import re
from collections import Counter
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from experiments import behavior_support, support
from looped_internals import analysis, models
from looped_internals.capture import compute_model_run_data
from looped_internals.models import MinimalRecurrentConfig, ModelSpec
from looped_internals.results import aggregate_values
from looped_internals.structure import LayerGroup, ModelLayerStructure

NOTEBOOK = Path(__file__).parents[1] / "experiments" / "limiting_behaviour.ipynb"


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
        if "orbit_spectrum" in namespace:
            return namespace
    raise AssertionError("Missing scientific definitions")


class Cache(dict):
    @property
    def cache_dict(self):
        return self


def test_cosines_pca_and_stage_groups_match_library_reductions():
    torch.set_num_threads(1)
    ns = setup()
    structure = ModelLayerStructure(
        [
            LayerGroup("prelude", 1, False),
            LayerGroup("core", 2, True),
            LayerGroup("coda", 1, False),
        ]
    )
    entries = list(structure.virtual_layers(8))
    generator = torch.Generator().manual_seed(7)
    cache = Cache()
    for entry in entries:
        cache[entry.residual_key] = torch.randn(1, 5, 8, generator=generator)
        attention = torch.rand(1, 2, 5, 5, generator=generator).tril()
        cache[entry.attn_key] = attention / attention.sum(-1, keepdim=True)
    input_ids = torch.arange(5).unsqueeze(0)
    run = compute_model_run_data(
        cache,
        input_ids,
        structure,
        compute_pairwise=False,
        compute_differences=True,
        compute_residual_entropy=True,
        store_raw_values=True,
    )[0]
    histories = ns["cosine_histories"](cache, entries, 2)
    assert list(histories) == [1, 2]
    for layer, signal in histories.items():
        indices = [e.virtual_layer_index for e in entries if e.layer_index == layer]
        torch.testing.assert_close(
            signal, run.to_final_state_residual_cosine_sim[indices, layer, 2:]
        )
        torch.testing.assert_close(signal[-1], torch.ones(3))

    class Tokenizer:
        def decode(self, token):
            return str(token)

    actual = ns["joint_pca"](cache, entries, input_ids, 2, Tokenizer())
    expected = run.get_pca_residual_trajectories(num_dimensions=6, group_by="layer")
    for layer in actual["layers"]:
        np.testing.assert_allclose(
            actual["coordinates"][layer], expected[f"L{layer}"][:, 2:], atol=1e-5
        )
    assert actual["positions"] == [2, 3, 4]
    # The fitted basis must include prefix tokens and non-recurrent layers.
    cache[entries[0].residual_key][:, :2] *= 1000
    changed = ns["joint_pca"](cache, entries, input_ids, 2, Tokenizer())
    assert not np.allclose(actual["coordinates"][1], changed["coordinates"][1])

    # Restore the prefix changed for the PCA-sensitivity check.
    cache[entries[0].residual_key][:, :2] /= 1000
    stages = ns["stage_curves"](cache, entries)
    for by in ("layer", "recurrence"):
        for field in ns["STAGE_METRICS"]:
            reference = aggregate_values([run], field, group_by=by)
            for key, indices in stages["groups"][by].items():
                np.testing.assert_allclose(
                    stages["curves"][field][indices], reference[key].mean, rtol=1e-6
                )

    late = ns["stage_curves"](cache, entries, first_recurrence=4)
    assert list(late["groups"]["recurrence"]) == [
        "Prelude",
        "R4",
        "R5",
        "R6",
        "R7",
        "Coda",
    ]
    assert late["virtual_layers"].tolist() == [0, *range(9, 18)]
    for field in ns["STAGE_METRICS"]:
        expected_values = aggregate_values([run], field, group_by="recurrence")
        for key, indices in late["groups"]["recurrence"].items():
            np.testing.assert_allclose(
                late["curves"][field][indices], expected_values[key].mean, rtol=1e-6
            )


def test_prevalence_counts_and_conditional_cooccurrence():
    ns = setup()
    rows = []
    for example, token, kinds in [
        (0, 0, ["FixedPoint", "Orbit"]),
        (1, 0, ["FixedPoint", "FixedPoint"]),
        (1, 1, ["Slider", "Unknown"]),
    ]:
        for layer, kind in enumerate(kinds):
            rows.append(
                dict(
                    model="raven",
                    recurrences=128,
                    prompt="no_prompt",
                    example_idx=example,
                    sequence_pos=token,
                    layer_idx=layer,
                    behavior_type=kind,
                )
            )
    frame = pd.DataFrame(rows)
    examples, tokens = ns["prevalence_tables"](frame)
    assert examples.iloc[0]["Non-Fixed-Point %"] == 100
    assert examples.iloc[0]["Orbit %"] == 50
    assert tokens.iloc[0]["Non-Fixed-Point %"] == 50
    assert tokens.iloc[0]["Orbit %"] == pytest.approx(100 / 6)
    cooccurrence = ns["conditional_cooccurrence"](frame)
    assert cooccurrence.loc["Orbit", "FixedPoint"] == 0.5
    assert cooccurrence.loc["FixedPoint", "Orbit"] == 1
    assert cooccurrence.loc["Unknown", "Slider"] == 1
    absent = ns["conditional_cooccurrence"](frame[frame.behavior_type != "Unknown"])
    assert absent["Unknown"].isna().all()


def test_orbit_selection_prioritizes_prevalence_then_coverage_then_amplitude():
    ns = setup()
    rows = []

    def add_case(model, example, total, amplitudes):
        for observation in range(total):
            is_orbit = observation < len(amplitudes)
            rows.append(
                dict(
                    model=model,
                    recurrences=128,
                    prompt="no_prompt",
                    example_idx=example,
                    layer_idx=observation % 4,
                    sequence_pos=observation,
                    token_str=str(observation),
                    behavior_type="Orbit" if is_orbit else "FixedPoint",
                    orbit_frequency=0.125 if is_orbit else np.nan,
                    orbit_amplitude=(amplitudes[observation] if is_orbit else np.nan),
                    slider_gradient=np.nan,
                )
            )

    # Example 1 beats the larger-amplitude example 0 on prevalence. Example 2
    # then beats example 1 on coverage after their prevalence ties.
    add_case("raven", 0, 4, [0.9])
    add_case("raven", 1, 4, [0.2, 0.3, 0.4])
    add_case("raven", 2, 8, [0.05] * 6)
    # The retrofit prevalence and coverage tie, so amplitude breaks the tie.
    add_case("retrofitted_llama", 3, 4, [0.4, 0.5])
    add_case("retrofitted_llama", 4, 4, [0.4, 0.6])
    rows.append(
        dict(
            model="raven",
            recurrences=128,
            prompt="no_prompt",
            example_idx=5,
            layer_idx=0,
            sequence_pos=0,
            token_str="slider",
            behavior_type="Slider",
            orbit_frequency=np.nan,
            orbit_amplitude=np.nan,
            slider_gradient=0.02,
        )
    )

    cases = ns["select_cases"](pd.DataFrame(rows))
    overall = cases["most_prevalent_orbit"]
    assert overall["example_idx"] == 2
    assert overall["orbit_prevalence"] == pytest.approx(0.75)
    assert overall["orbit_observations"] == 6
    assert overall["total_observations"] == 8

    retrofit = cases["retrofitted_most_prevalent_orbit"]
    assert retrofit["example_idx"] == 4
    assert retrofit["orbit_amplitude"] == pytest.approx(0.6)
    assert cases["largest_slider"]["example_idx"] == 5


def test_classifier_and_fft_share_the_same_nonfinal_window():
    ns = setup()
    t = torch.arange(128).float()
    series = torch.stack(
        [
            torch.ones(128),
            0.5 + 0.08 * torch.sin(2 * torch.pi * t / 8),
            0.2 + 0.004 * t,
            torch.full((128,), 0.5),
        ],
        dim=1,
    )
    series[-1] = 1  # The self-comparison must never enter the classifier window.
    record = dict(
        model="raven",
        recurrences=128,
        prompt="no_prompt",
        example_idx=0,
        question_start=2,
        token_strs=list("abcd"),
        similarities={4: series},
    )
    frame = ns["classify_example"](record)
    assert frame.behavior_type.tolist() == ["FixedPoint", "Orbit", "Slider", "Unknown"]
    assert frame.sequence_pos.tolist() == [2, 3, 4, 5]
    trimmed, _, _, frequencies, amplitudes = ns["orbit_spectrum"](series[:, 1])
    torch.testing.assert_close(trimmed, series[-33:-1, 1])
    orbit = frame.iloc[1]
    assert orbit.orbit_amplitude == pytest.approx(float(amplitudes.max()))
    assert orbit.orbit_frequency == pytest.approx(
        float(frequencies[amplitudes.argmax()])
    )
    assert frame.iloc[2].slider_gradient == pytest.approx(0.004, abs=1e-7)


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 2
    bos_token_id = 1
    chat_template = "test chat template"

    def encode(self, text, add_special_tokens=True, return_tensors=None):
        words = re.findall(r"<[^>]+>|[^\s<>]+", text)
        ids = ([1] if add_special_tokens else []) + [
            1 if word == "<bos>" else 3 + sum(map(ord, word)) % 61 for word in words
        ]
        return torch.tensor([ids]) if return_tensors else ids

    def decode(self, token):
        return f"token-{token}"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        return (
            "<bos> "
            + " ".join(f"<{m['role']}> {m['content']} </{m['role']}>" for m in messages)
            + (" <assistant>" if add_generation_prompt else "")
        )


@pytest.mark.parametrize("selection", ["distinct", "shared", "absent"])
def test_notebook_collects_only_needed_examples_and_metrics(
    tmp_path, monkeypatch, selection
):
    torch.set_num_threads(1)
    original_load = models.load_model
    loads, captures, entropy_calls = [], [], []

    def load(name, config):
        bridge, structure = original_load(
            ModelSpec(
                MinimalRecurrentConfig(
                    d_model=8,
                    n_heads=2,
                    n_layers=2,
                    num_cycles=config.num_steps,
                    d_vocab=64,
                    expansion_factor=2,
                )
            ),
            device="cpu",
        )
        bridge.tokenizer = Tokenizer()
        bridge.test_name = name
        loads.append((name, config.num_steps))
        return bridge, structure

    original_capture = behavior_support.capture_example

    def capture(bridge, structure, question, config, device, *, attention=False):
        result = original_capture(
            bridge, structure, question, config, device, attention=attention
        )
        captures.append((bridge.test_name, config.prompt_stem, question, attention))
        input_ids, start, cache = result
        assert 0 < start < input_ids.shape[1]
        assert any("hook_pattern" in key for key in cache) is attention
        if config.replace_prompt_with_pad:
            assert input_ids[0, :start].eq(bridge.tokenizer.pad_token_id).all()
        return result

    original_entropy = analysis.get_residual_entropy

    def entropy(*args, **kwargs):
        assert captures[-1][-1] is True
        entropy_calls.append(captures[-1])
        return original_entropy(*args, **kwargs)

    monkeypatch.setattr(support, "load_experiment_model", load)
    monkeypatch.setattr(behavior_support, "capture_example", capture)
    monkeypatch.setattr(analysis, "get_residual_entropy", entropy)
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, **kw: datasets.Dataset.from_list(
            [
                {
                    "question": f"Question {i} asks how many apples are in this box",
                    "answer": "Five",
                }
                for i in range(20)
            ]
        ),
    )
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(support, "PAPER_PLOTS_DIR", tmp_path / "paper")
    paper = tmp_path / "paper" / "limiting_behaviour" / "gsm8k"
    paper.mkdir(parents=True)
    protected = paper / "orbit_algorithm_examples.pdf"
    protected.write_bytes(b"Previous full run")
    figures = {}
    original_save = support.save_figure

    def save(fig, run_dir, name, **kwargs):
        fig.canvas.draw()
        figures[name] = (fig.get_size_inches().copy(), len(fig.axes))
        if selection == "distinct" and name in {
            "pca_intermediate",
            "orbit_algorithm_examples",
        }:
            fig.savefig(tmp_path / f"{name}.png", dpi=180)
        original_save(fig, run_dir, name, **kwargs)

    monkeypatch.setattr(support, "save_figure", save)
    namespace = {}
    for cell in cells():
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(
                mode="quick" if selection == "absent" else "full",
                limit=1,
                device="cpu",
                output_dir=str(tmp_path),
            )
            if selection == "shared":
                namespace["example_index"] = support.sample_indices(20, 1, 42)[0]
        if "def classify_example(" in source:
            original_classify = namespace["classify_example"]

            def classify(record):
                # Force selection branches; classifier numerics are checked above.
                frame = original_classify(record)
                frame["behavior_type"] = "FixedPoint"
                frame[["orbit_frequency", "orbit_amplitude", "slider_gradient"]] = (
                    np.nan
                )
                if selection != "absent":
                    large = (record["model"] == "raven") == (selection == "distinct")
                    frame.loc[
                        0, ["behavior_type", "orbit_frequency", "orbit_amplitude"]
                    ] = ["Orbit", 0.125, 0.3 if large else 0.2]
                    frame.loc[1, ["behavior_type", "slider_gradient"]] = [
                        "Slider",
                        0.01,
                    ]
                return frame

            namespace["classify_example"] = classify

    stage_count = {"distinct": 2, "shared": 1, "absent": 0}[selection]
    extra_pca = int(selection != "shared")
    assert len(captures) == 8 + extra_pca + stage_count
    assert Counter(c[-1] for c in captures) == Counter(
        {False: 8 + extra_pca, True: stage_count}
    )
    assert len(loads) == 2 + stage_count
    assert set(loads) == {("raven", 128), ("retrofitted_llama", 128)}
    expected_entropy_layers = {
        "distinct": (128 + 64) * 2,
        "shared": 128 * 2,
        "absent": 0,
    }
    assert len(entropy_calls) == expected_entropy_layers[selection]
    assert len(namespace["stage_data"]) == stage_count
    assert namespace["behavior_df"].example_idx.nunique() == 1
    assert set(namespace["behavior_df"].example_idx) == set(namespace["indices"])
    if extra_pca:
        assert namespace["example_index"] not in namespace["indices"]
    assert set(figures) == {
        "behavior_cooccurrence_heatmap",
        "pca_final_layer",
        "pca_intermediate",
        "largest_orbit_stability_in_recurrence",
        "largest_orbit_stability_in_percentage_depth",
        "stability_in_percentage_depth_llama",
        "orbit_algorithm_examples",
    }
    for name, (size, _) in figures.items():
        assert size[0] == pytest.approx(
            3.25 if name == "behavior_cooccurrence_heatmap" else 6.75
        )
    directory = namespace["run_dir"]
    assert len(list(directory.glob("*.pdf"))) == 7
    assert len(list(directory.rglob("*.pt"))) == 8 + extra_pca + 1
    assert not list(directory.rglob("*.png"))
    assert len(list(directory.glob("*.tex"))) == 2
    assert {"selected_cases.csv", "behavior_existence.csv", "behavior_tokens.csv"} <= {
        p.name for p in directory.iterdir()
    }
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["model_recurrences"] == [["raven", 128], ["retrofitted_llama", 128]]
    assert metadata["cohort_metrics"] == ["same_layer_final_residual_cosine"]
    assert metadata["orbit_selection"] == (
        "token_layer_prevalence_then_count_then_amplitude"
    )
    overview = torch.load(directory / "overview.pt", weights_only=False)
    assert len(overview["stages"]) == stage_count
    if selection == "absent":
        assert not overview["signals"]
        assert all(case is None for case in overview["cases"].values())
        assert protected.read_bytes() == b"Previous full run"
        assert list(paper.iterdir()) == [protected]
    else:
        assert set(overview["signals"]) == {
            "most_prevalent_orbit",
            "retrofitted_most_prevalent_orbit",
            "largest_slider",
        }
        for figure in directory.glob("*.pdf"):
            assert (paper / figure.name).read_bytes() == figure.read_bytes()

    # A figure rerun uses the saved reductions and makes no forward pass.
    before = len(captures)
    exec(
        next(
            "".join(c["source"])
            for c in cells()
            if "".join(c["source"]) == 'plot_pca("pca_final_layer")'
        ),
        namespace,
    )
    assert len(captures) == before
