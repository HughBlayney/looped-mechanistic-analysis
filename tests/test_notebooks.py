"""Execute each new notebook with real tiny activations and local data fixtures."""

import json
import os
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import support
from experiments.paper_models import MAIN_PARCAE_MODEL, PAPER_MODEL_GROUPS
from looped_internals.evaluation.datasets import MCQExample
from looped_internals.models import MinimalRecurrentConfig, ModelSpec, load_model
from looped_internals.structure import LayerGroup, ModelLayerStructure


class TinyTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    bos_token_id = 1

    def encode(self, text, add_special_tokens=True, return_tensors=None):
        ids = ([1] if add_special_tokens else []) + [
            2 + sum(map(ord, w)) % 60 for w in text.split()
        ]
        return torch.tensor([ids]) if return_tensors else ids

    def decode(self, token):
        return f"token-{token}"


@pytest.fixture
def notebook_models(monkeypatch):
    import datasets
    from lm_eval.tasks import TaskManager

    import looped_internals.evaluation as evaluation

    def model(name, config, *, steps=None):
        bridge, structure = load_model(
            ModelSpec(
                MinimalRecurrentConfig(
                    d_model=16,
                    n_heads=2,
                    n_layers=2,
                    num_cycles=1
                    if name in support.BASELINES
                    else steps or config.num_steps,
                    d_vocab=64,
                )
            ),
            device="cpu",
        )
        bridge.tokenizer = TinyTokenizer()
        if name in support.BASELINES:
            structure = ModelLayerStructure(groups=[LayerGroup("blocks", 2, False)])
        return bridge, structure

    def load_dataset(path, *args, split=None, **kwargs):
        if path == "Rowan/hellaswag":
            docs = datasets.Dataset.from_list(
                [
                    {
                        "activity_label": "Cooking",
                        "ctx_a": "A person puts a pan on the stove.",
                        "ctx_b": "they",
                        "endings": [
                            "leave.",
                            "sleep.",
                            "turn on the burner. [artifact]",
                            "sit.",
                        ],
                        "label": "2",
                    }
                ]
                * 20
            )
            return (
                docs
                if split
                else datasets.DatasetDict(
                    {name: docs for name in ["train", "validation"]}
                )
            )
        docs = datasets.Dataset.from_list(
            [{"question": "How many apples in a box of five apples", "answer": "Five"}]
            * 20
        )
        return (
            docs
            if split
            else datasets.DatasetDict({name: docs for name in ["train", "test"]})
        )

    def lm_model(name, config):
        from looped_internals.evaluation.lm_eval import BridgeLM

        bridge, structure = model(name, config)
        return BridgeLM(
            bridge,
            structure,
            compute_pairwise=config.experiment
            in {"cyclic_similarity", "bos_comparison"},
            compute_residual_entropy=config.experiment == "bos_comparison",
            bos_policy=config.bos_policy,
        )

    monkeypatch.setattr(support, "load_experiment_model", model)
    monkeypatch.setattr(support, "_load_model", lm_model)
    task_path = Path(__file__).parents[1] / "src/looped_internals/evaluation/tasks"
    monkeypatch.setattr(
        support,
        "get_task_manager",
        lambda: TaskManager(include_path=task_path, include_defaults=False),
    )
    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    monkeypatch.setattr(
        evaluation,
        "load_mcq_dataset",
        lambda *a, **kw: (
            [MCQExample("Some small question", ["one", "two", "three", "four"], 1)] * 8
        ),
    )
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    torch.set_num_threads(1)


@pytest.mark.parametrize("dataset", ["gsm8k", "hellaswag"])
def test_cyclic_similarity_executes(notebook_models, tmp_path, monkeypatch, dataset):
    paper_root = tmp_path / "paper-plots"
    monkeypatch.setattr(support, "PAPER_PLOTS_DIR", paper_root)
    collected = []
    collect_metrics = support.collect_metrics

    def record_collection(config, summarize, **kwargs):
        collected.append((config.models[0], config.num_steps))
        return collect_metrics(config, summarize)

    monkeypatch.setattr(support, "collect_metrics", record_collection)
    notebook = Path(__file__).parents[1] / "experiments" / "cyclic_similarity.ipynb"
    namespace = {}
    for i, cell in enumerate(json.loads(notebook.read_text())["cells"]):
        if cell["cell_type"] != "code":
            continue
        exec(
            compile("".join(cell["source"]), f"{notebook}:cell-{i}", "exec"), namespace
        )
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(
                mode="quick",
                dataset=dataset,
                limit=1,
                device="cpu",
                output_dir=str(tmp_path),
            )
    directory = namespace["run_dir"]
    assert list(directory.glob("*.pdf"))
    assert not list(directory.rglob("*.png"))
    assert not paper_root.exists()
    assert directory.name.startswith("quick-")
    assert (directory / "run.json").exists()
    overview = torch.load(directory / "overview.pt", weights_only=False)
    expected_metrics = {
        "pairwise_residual_cosine_sim",
        "pairwise_residual_diff_norm",
        "pairwise_attn_pattern_diff_norm",
    }
    assert set(overview["summaries"]) == set(namespace["model_recurrences"])
    indices = []
    for pair, matrices in overview["summaries"].items():
        assert set(matrices) == expected_metrics
        steps = pair[1]
        assert steps == 8
        for metric, matrix in matrices.items():
            assert matrix.shape == (2 * steps, 2 * steps)
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)
            np.testing.assert_allclose(
                np.load(directory / f"{pair[0]}-{steps}-steps-{metric}.npy"),
                matrix,
            )
        assert overview["run"][pair]["num_steps"] == steps
        assert overview["run"][pair]["compute_pairwise"]
        assert not overview["run"][pair]["compute_differences"]
        assert not overview["run"][pair]["compute_residual_entropy"]
        indices.append(overview["run"][pair]["indices"])
    assert all(index_list == indices[0] for index_list in indices)
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["dataset"] == dataset and directory.parent.name == dataset
    assert metadata["mode"] == "quick" and metadata["limit"] == 1
    assert metadata["model_recurrences"] == [
        list(pair) for pair in namespace["model_recurrences"]
    ]
    assert "models" not in metadata and "num_steps" not in metadata
    assert collected == namespace["model_recurrences"]
    assert metadata["figure_pairs"] == {
        group: [list(pair) for pair in pairs]
        for group, pairs in namespace["figure_pairs"].items()
    }
    assert overview["figure_pairs"] == namespace["figure_pairs"]
    assert {path.stem for path in directory.glob("*.pdf")} == {
        f"{metric}_{group}"
        for metric in expected_metrics
        for group in PAPER_MODEL_GROUPS
    }
    arrays = {path: path.stat().st_mtime_ns for path in directory.glob("*.npy")}
    assert len(arrays) == 3 * len(collected)

    def unexpected_collection(*args, **kwargs):
        pytest.fail("A figure cell repeated inference")

    namespace["collect_metrics"] = unexpected_collection
    for cell in json.loads(notebook.read_text())["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith("plot_similarity("):
            exec(compile(source, str(notebook), "exec"), namespace)
    assert {path: path.stat().st_mtime_ns for path in arrays} == arrays


def _pair_setup(name="cyclic_similarity", **parameters):
    notebook = Path(__file__).parents[1] / "experiments" / f"{name}.ipynb"
    namespace = {}
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(notebook), "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            assert not {"models", "num_steps"} & namespace.keys()
            assert namespace["mode"] == "quick" and namespace["limit"] is None
            namespace.update(parameters)
        if "configs" in namespace:
            return namespace
    raise AssertionError("Notebook did not configure the selected pairs")


@pytest.mark.parametrize(
    "name, recurrences", [("convergence", 128), ("cyclic_similarity", 8)]
)
def test_notebook_uses_shared_paper_groups(monkeypatch, name, recurrences):
    assert PAPER_MODEL_GROUPS["main"] == (
        "ouro_1_4b",
        "raven",
        "retrofitted_llama",
        MAIN_PARCAE_MODEL,
    )
    assert PAPER_MODEL_GROUPS["appendix"] == (
        "retrofitted_olmo",
        "retrofitted_tinyllama",
        "parcae_140m",
        "parcae_370m",
    )
    assert PAPER_MODEL_GROUPS["all"] == (
        PAPER_MODEL_GROUPS["main"] + PAPER_MODEL_GROUPS["appendix"]
    )
    setup = _pair_setup(name)
    assert setup["model_recurrences"] == [
        (model, recurrences) for model in dict.fromkeys(PAPER_MODEL_GROUPS["all"])
    ]
    assert setup["figure_pairs"] == {
        group: [(model, recurrences) for model in names]
        for group, names in PAPER_MODEL_GROUPS.items()
    }

    # The eventual shared Parcae swap changes collection and figures together.
    main = (*PAPER_MODEL_GROUPS["main"][:3], "parcae_770m")
    monkeypatch.setitem(PAPER_MODEL_GROUPS, "main", main)
    monkeypatch.setitem(
        PAPER_MODEL_GROUPS, "all", main + PAPER_MODEL_GROUPS["appendix"]
    )
    setup = _pair_setup(name, recurrences=4)
    assert setup["model_recurrences"] == [
        (model, 4) for model in PAPER_MODEL_GROUPS["all"]
    ]
    assert setup["figure_pairs"]["main"][-1] == ("parcae_770m", 4)
    assert setup["figure_pairs"]["appendix"][-1] == ("parcae_370m", 4)


@pytest.mark.parametrize(
    "name",
    [
        "cyclic_similarity",
        "convergence",
        "stages_of_inference",
        "mixing_correlation",
        "isolating_stability",
        "bos_comparison",
        "limiting_behaviour",
        "entropy_lens",
        "multiple_choice",
        "trajectories",
        "initialisation",
        "neuron_detection",
    ],
)
def test_pair_slurm_selects_full_mode(tmp_path, caplog, name):
    """Check the actual launcher arguments and Papermill's injected settings."""
    papermill = pytest.importorskip("papermill")
    root = Path(__file__).parents[1]
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "Path('arguments.json').write_text(json.dumps(sys.argv[1:]))\n"
    )
    python.chmod(0o755)
    subprocess.run(
        ["bash", str(root / "slurm_jobs" / "run_notebook.sh"), name],
        cwd=tmp_path,
        env={**os.environ, "SLURM_JOB_ID": "test"},
        check=True,
    )
    args = json.loads((tmp_path / "arguments.json").read_text())
    overrides = args[args.index("--log-output") + 1 :]
    parameters = {
        overrides[i + 1]: overrides[i + 2] for i in range(0, len(overrides), 3)
    }
    expected = {"mode": "full", "device": "cuda"}
    if name not in {
        "initialisation",
        "trajectories",
        "neuron_detection",
        "limiting_behaviour",
    }:
        expected["limit"] = "256"
    if name not in {
        "initialisation",
        "trajectories",
        "multiple_choice",
        "limiting_behaviour",
        "neuron_detection",
    }:
        expected["dataset"] = "gsm8k"
    assert parameters == expected
    if "limit" in parameters:
        parameters["limit"] = int(parameters["limit"])
    prepared = papermill.execute_notebook(
        root / "experiments" / f"{name}.ipynb",
        tmp_path / "prepared.ipynb",
        parameters=parameters,
        prepare_only=True,
    )
    injected = next(
        cell
        for cell in prepared.cells
        if "injected-parameters" in cell.metadata.get("tags", [])
    )
    injected_parameters = {}
    exec(injected.source, injected_parameters)
    assert all(injected_parameters[key] == value for key, value in parameters.items())
    assert "Unable to parse" not in caplog.text
    assert "unknown parameter" not in caplog.text
    if name in {
        "initialisation",
        "trajectories",
        "multiple_choice",
        "neuron_detection",
        "isolating_stability",
        "mixing_correlation",
    }:
        return  # Their distinct full-mode configurations have dedicated notebook tests.
    manual_setup = _pair_setup(name)
    manual = manual_setup["configs"]
    slurm = _pair_setup(name, **parameters)["configs"]
    assert manual.keys() == slurm.keys() == set(manual_setup["model_recurrences"])
    for pair, config in manual.items():
        assert config.dataset == "gsm8k" and config.limit == 1
        assert config.mode == "quick"
        assert config.models == (pair[0],) and config.num_steps == pair[1]
        assert slurm[pair] == replace(
            config,
            mode="full",
            limit=None if name == "limiting_behaviour" else 256,
            device="cuda",
            dtype="bfloat16",
        )


@pytest.mark.parametrize(
    "name",
    [
        "cyclic_similarity",
        "convergence",
        "stages_of_inference",
        "bos_comparison",
        "limiting_behaviour",
        "entropy_lens",
    ],
)
@pytest.mark.parametrize(
    "mode, limit, expected_limit",
    [("quick", None, 1), ("full", None, 256), ("quick", 256, 256), ("full", 1, 1)],
)
def test_pair_modes_control_paper_exports(
    name, mode, limit, expected_limit, tmp_path, monkeypatch
):
    """Notebook modes govern exports independently of the chosen sample count."""
    namespace = _pair_setup(name, mode=mode, limit=limit, output_dir=str(tmp_path))
    config = namespace["config"]
    if name == "limiting_behaviour" and mode == "full" and limit is None:
        expected_limit = None
    assert all(
        c.mode == mode and c.limit == expected_limit
        for c in namespace["configs"].values()
    )
    # Model collection is irrelevant to this check: supply a completed run's PDF.
    run_dir = support.save_results(
        {},
        {"experiment": name, "dataset": config.dataset, "mode": config.mode},
        tmp_path,
    )
    assert run_dir.name.startswith(f"{mode}-")
    (run_dir / "figure.pdf").write_bytes(b"New run figure")
    paper_root = tmp_path / "paper-plots"
    monkeypatch.setattr(support, "PAPER_PLOTS_DIR", paper_root)
    destination = paper_root / name / config.dataset
    destination.mkdir(parents=True)
    (destination / "figure.pdf").write_bytes(b"Previous full-run figure")
    namespace["run_dir"] = run_dir
    notebook = Path(__file__).parents[1] / "experiments" / f"{name}.ipynb"
    export_cell = next(
        cell
        for cell in json.loads(notebook.read_text())["cells"]
        if "paper_dir = export_paper_figures(" in "".join(cell["source"])
    )
    # Export must use the collected run's config, not a subsequently edited parameter.
    namespace["mode"] = "full" if mode == "quick" else "quick"
    exec(compile("".join(export_cell["source"]), str(notebook), "exec"), namespace)
    if mode == "quick":
        assert namespace["paper_dir"] is None
        assert (destination / "figure.pdf").read_bytes() == b"Previous full-run figure"
    else:
        assert namespace["paper_dir"] == destination
        assert (destination / "figure.pdf").read_bytes() == b"New run figure"


@pytest.mark.parametrize("name", ["cyclic_similarity", "bos_comparison", "convergence"])
def test_papermill_accepts_model_recurrence_lists(tmp_path, caplog, name):
    papermill = pytest.importorskip("papermill")
    notebook = Path(__file__).parents[1] / "experiments" / f"{name}.ipynb"
    pairs = [["raven", 8], ["raven", 32], ["ouro_1_4b", 4]]
    if name == "cyclic_similarity":
        pairs = [["raven", 8], ["ouro_1_4b", 8]]
    prepared = papermill.execute_notebook(
        notebook,
        tmp_path / "pairs.ipynb",
        parameters={"model_recurrences": pairs},
        prepare_only=True,
    )
    injected = next(
        cell
        for cell in prepared.cells
        if "injected-parameters" in cell.metadata.get("tags", [])
    )
    namespace = {}
    exec(injected.source, namespace)
    configs = _pair_setup(name, model_recurrences=namespace["model_recurrences"])[
        "configs"
    ]
    assert list(configs) == [tuple(pair) for pair in pairs]
    assert "Unable to parse" not in caplog.text
    assert "unknown parameter" not in caplog.text


@pytest.mark.parametrize(
    "pairs", [[], [["raven", 8], ["raven", 8]], [["typo", 8]], [["raven", 1]]]
)
def test_cyclic_invalid_pairs_fail_before_collection(pairs):
    with pytest.raises(ValueError):
        _pair_setup(model_recurrences=pairs)


def _bos_science():
    notebook = Path(__file__).parents[1] / "experiments" / "bos_comparison.ipynb"
    namespace = {"np": np}
    for cell in json.loads(notebook.read_text())["cells"]:
        source = "".join(cell["source"])
        if (
            "def summarize_bos_comparison(" in source
            or "def check_input_pairs(" in source
        ):
            exec(compile(source, str(notebook), "exec"), namespace)
    return namespace


def test_bos_reductions_weight_examples_equally_and_keep_layer_groups():
    science = _bos_science()
    layers = [0, 1, 2, 1, 2, 3]
    blocks = [
        SimpleNamespace(layer_index=layer, virtual_layer_index=index)
        for index, layer in enumerate(layers)
    ]
    examples = []
    # Different lengths ensure the normalized matrix is reduced per example.
    for length, scale in [(4, 2.0), (9, 6.0)]:
        run = SimpleNamespace(
            input_ids=torch.arange(length)[None],
            per_virtual_layer_metrics=blocks,
            recurrent_layer_indices=[1, 2],
            pairwise_attn_pattern_diff_norm=torch.ones(6, 6, 2) * scale,
            get_metric=lambda field: torch.tensor([[scale, 2 * scale]]).repeat(6, 1),
        )
        examples.append(science["summarize_bos_comparison"](run))
    data = science["aggregate_bos"](examples)
    np.testing.assert_allclose(data[science["ATTENTION"]], 4)
    np.testing.assert_allclose(data[science["NORMALIZED_ATTENTION"]], 1.5)
    for field in science["METRICS"]:
        np.testing.assert_allclose(data["curves"][field]["mean"], 6)
        np.testing.assert_allclose(data["curves"][field]["std"], np.sqrt(18))
        single = science["aggregate_bos"](examples[:1])
        np.testing.assert_array_equal(single["curves"][field]["std"], np.zeros(6))
    groups = science["recurrence_groups"](data)
    assert {key: indices.tolist() for key, indices in groups.items()} == {
        "Prelude": [0],
        "R1": [1, 2],
        "R2": [3, 4],
        "Coda": [5],
    }
    feedforward = science["recurrence_groups"](
        {"layers": [0, 1], "recurrent_layers": []}
    )
    assert feedforward["Feedforward"].tolist() == [0, 1]


@pytest.mark.parametrize(
    "mismatch", ["tokens", "indices", "topology", "missing_example"]
)
def test_bos_audit_rejects_unpaired_inputs(mismatch):
    pair = ("raven", 8)
    example = {
        "input_ids": [2, 3, 0],
        "layers": [0, 0],
        "virtual_layers": [0, 1],
        "recurrent_layers": [0],
    }
    summaries = {"no_bos": {pair: [example]}, "bos": {pair: [deepcopy(example)]}}
    summaries["bos"][pair][0]["input_ids"].insert(0, 1)
    info = {"indices": [7], "tokenization": {"raven": {"bos_token_id": 1}}}
    run_info = {condition: {pair: deepcopy(info)} for condition in summaries}
    check = _bos_science()["check_input_pairs"]
    assert check(pair, summaries, run_info)[0]["sole_difference_is_bos"]
    if mismatch == "tokens":
        summaries["bos"][pair][0]["input_ids"][-1] = 9
    elif mismatch == "indices":
        run_info["bos"][pair]["indices"] = [8]
    elif mismatch == "topology":
        summaries["bos"][pair][0]["layers"] = [0, 1]
    else:
        summaries["bos"][pair].clear()
    with pytest.raises(ValueError):
        check(pair, summaries, run_info)


@pytest.mark.parametrize("mode, dataset", [("quick", "gsm8k"), ("full", "hellaswag")])
def test_bos_notebook_collects_pairs_once_and_reuses_figures(
    mode, dataset, notebook_models, tmp_path, monkeypatch
):
    paper_root = tmp_path / "paper-plots"
    monkeypatch.setattr(support, "PAPER_PLOTS_DIR", paper_root)
    collected, figures = [], {}
    collect = support.collect_metrics
    save = support.save_figure

    def record_collection(config, summarize, **kwargs):
        collected.append(config)
        return collect(config, summarize, **kwargs)

    def record_figure(fig, run_dir, name, **kwargs):
        figures[name] = fig.get_size_inches().copy()
        if name.endswith("attention-similarity"):
            images = [ax.images[0] for ax in fig.axes if ax.images]
            assert len(images) == 2
            assert images[0].get_clim() == images[1].get_clim()
        save(fig, run_dir, name, **kwargs)

    monkeypatch.setattr(support, "collect_metrics", record_collection)
    monkeypatch.setattr(support, "save_figure", record_figure)
    # Two rollout lengths share the same baseline, which must be collected once.
    pairs = [("ouro_1_4b", 2), ("ouro_1_4b", 3), ("llama_1b", 1)]
    notebook = Path(__file__).parents[1] / "experiments" / "bos_comparison.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    namespace = {}
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), f"{notebook}:{index}", "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(
                mode=mode,
                dataset=dataset,
                model_recurrences=pairs,
                limit=2,
                device="cpu",
                output_dir=str(tmp_path),
            )
    directory = namespace["run_dir"]
    expected_runs = {
        (pair, condition) for pair in pairs for condition in ("no_bos", "bos")
    }
    assert len(collected) == len(expected_runs)
    assert {
        ((c.models[0], c.num_steps), c.bos_policy) for c in collected
    } == expected_runs
    assert all(
        c.compute_residual_entropy and not c.include_baselines for c in collected
    )
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["mode"] == mode and metadata["limit"] == 2
    assert metadata["dataset"] == dataset and directory.parent.name == dataset
    assert metadata["model_recurrences"] == [list(pair) for pair in pairs]
    assert metadata["conditions"] == ["no_bos", "bos"]
    assert (
        not {"models", "num_steps", "bos_policy", "include_baselines"} & metadata.keys()
    )
    assert directory.name.startswith(f"{mode}-")
    overview = torch.load(directory / "overview.pt", weights_only=False)
    for condition in ("no_bos", "bos"):
        assert set(overview["summaries"][condition]) == set(pairs)
        for pair in pairs:
            assert len(overview["summaries"][condition][pair]) == 2
            info = overview["run"][condition][pair]
            assert info["num_steps"] == pair[1]
            assert info["compute_pairwise"] and not info["compute_differences"]
            assert info["tokenization"][pair[0]]["bos_policy"] == condition
    checks = json.loads((directory / "paired-input-checks.json").read_text())
    assert len(checks) == 6 and all(check["sole_difference_is_bos"] for check in checks)
    assert (
        (directory / "stage-curves.csv")
        .read_text()
        .startswith("condition,model,recurrences,")
    )
    assert len(list(directory.glob("*.npy"))) == 4 * len(pairs)
    assert len(figures) == 3 * len(pairs) + 5
    assert len(list(directory.glob("*.pdf"))) == len(figures)
    for name, size in figures.items():
        individual = name.endswith(
            ("-attention-similarity", "-attention-length-normalized")
        )
        expected = support.get_figsize(not individual, (2, 1))[0]
        assert size[0] == pytest.approx(expected)
    if mode == "quick":
        assert not paper_root.exists()
    else:
        assert len(list(paper_root.rglob("*.pdf"))) == len(figures)
        assert len(
            list((paper_root / "bos_comparison" / dataset).glob("*.pdf"))
        ) == len(figures)

    # Rerun every figure cell, with collection now forbidden.
    def unexpected_collection(*args, **kwargs):
        pytest.fail("A figure cell repeated inference")

    namespace["collect_metrics"] = unexpected_collection
    for cell in cells:
        source = "".join(cell["source"])
        if (
            cell["cell_type"] == "code"
            and "def " not in source
            and any(
                f"{function}(" in source
                for function in (
                    "plot_attention",
                    "plot_profiles",
                    "plot_profile_summary",
                )
            )
        ):
            exec(compile(source, str(notebook), "exec"), namespace)
    assert len(collected) == len(expected_runs)
