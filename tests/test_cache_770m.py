"""Adding the main 770M model preserves existing experiment cache entries."""

import json
import sys
from pathlib import Path

import pytest
from test_experiment_cache import cached_inference  # noqa: F401
from test_experiments import local_inference  # noqa: F401

from experiments import paper_models, run_cached, support
from experiments.cache import ExperimentCache
from looped_internals.models.loading import MODEL_SPECS


def test_main_paper_uses_uploaded_770m_and_matched_control():
    assert paper_models.MAIN_PARCAE_MODEL == "parcae_770m"
    assert paper_models.PAPER_MODEL_GROUPS["main"][-1] == "parcae_770m"
    assert paper_models.PAPER_MODEL_GROUPS["appendix"][-1] == "parcae_370m"
    assert len(set(paper_models.PAPER_MODEL_GROUPS["all"])) == 8
    assert support.BASELINE_FOR["parcae_770m"] == "parcae_770m_rec1"
    assert MODEL_SPECS["parcae_770m"].source.weights == "SandyResearch/parcae-770m"
    assert MODEL_SPECS["parcae_770m_rec1"].source.weights.endswith("/parcae-770m-rec1")
    assert MODEL_SPECS["parcae_770m_rec1"].default_steps == 1


def select_main(monkeypatch, model, *, small=False):
    main = (
        ("ouro_1_4b", model)
        if small
        else (*paper_models.PAPER_MODEL_GROUPS["main"][:3], model)
    )
    appendix = (
        ("parcae_140m", "parcae_370m")
        if small
        else paper_models.PAPER_MODEL_GROUPS["appendix"]
    )
    monkeypatch.setattr(
        paper_models,
        "PAPER_MODEL_GROUPS",
        {
            "main": main,
            "appendix": appendix,
            "all": main + appendix,
        },
    )


def configuration(name, output_dir):
    path = Path(__file__).parents[1] / "experiments" / f"{name}.ipynb"
    namespace = {"__name__": "__main__"}
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
        if "parameters" in cell.get("metadata", {}).get("tags", []):
            namespace.update(
                mode="full", device="cpu", dtype="float32", output_dir=str(output_dir)
            )
        if "result_cache" in namespace:
            return namespace
    raise AssertionError(f"No configuration found: {name}")


@pytest.mark.parametrize(
    "notebook, expected",
    [
        ("convergence", {("parcae_770m", 128)}),
        ("cyclic_similarity", {("parcae_770m", 8)}),
        (
            "stages_of_inference",
            {("parcae_770m", 32), ("parcae_770m", 128), ("parcae_770m_rec1", 1)},
        ),
        ("entropy_lens", {("parcae_770m", 32), ("parcae_770m_rec1", 1)}),
        ("multiple_choice", {("parcae_770m", 32), ("parcae_770m_rec1", 1)}),
        ("trajectories", {("parcae_770m", 32)}),
        ("neuron_detection", {("parcae_770m", None), ("parcae_770m_rec1", None)}),
    ],
)
def test_paper_switch_adds_only_required_model_pairs(
    monkeypatch, tmp_path, notebook, expected
):
    plans = []
    for main in ("parcae_370m", "parcae_770m"):
        select_main(monkeypatch, main)
        namespace = configuration(notebook, tmp_path)
        configs = namespace.get("configs")
        if configs is None:
            configs = {
                (name, None): namespace["config"] for name in namespace["config"].models
            }
        cache = ExperimentCache(tmp_path / "cache", "off")
        plans.append(
            {
                pair: cache.key(
                    config,
                    pair[0],
                    residual_entropy=getattr(config, "compute_residual_entropy", False),
                    compute_coda=pair
                    in namespace.get(
                        "coda_pairs", namespace.get("coda_model_recurrences", [])
                    ),
                    batch_size=namespace.get("probe_batch_sizes", {}).get(pair[0]),
                )
                for pair, config in configs.items()
            }
        )
    before, after = plans
    assert after.keys() - before.keys() == expected
    for pair in before.keys() & after.keys():
        assert after[pair] == before[pair]
    assert not list(tmp_path.iterdir())


def test_cached_joint_runs_only_compute_770m_and_its_control(
    monkeypatch, tmp_path, request
):
    cache, _config, counts = request.getfixturevalue("cached_inference")
    monkeypatch.setattr(
        ExperimentCache,
        "model",
        lambda self, name: (None, {"revision": "test", "name": name}),
    )
    loads = []
    load = support._load_model

    def tracked(name, config, **kwargs):
        loads.append((config.dataset, name, config.num_steps))
        return load(name, config, **kwargs)

    monkeypatch.setattr(support, "_load_model", tracked)
    common = {"limit": 1, "output_dir": str(tmp_path / "runs")}
    settings = tmp_path / "parameters.json"
    settings.write_text(
        json.dumps(
            {
                "convergence": {**common, "recurrences": 3},
                "stages_of_inference": {
                    **common,
                    "profile_recurrences": 2,
                    "ouro_recurrences": 2,
                    "long_recurrences": 3,
                    "mixing_pair": ["ouro_1_4b", 2],
                    "ablation_pair": ["ouro_1_4b", 2],
                    "ablation_hook": "blocks.0.mlp.hook_out",
                },
            }
        )
    )
    arguments = [
        "run_cached",
        "--notebooks",
        "convergence",
        "stages_of_inference",
        "--datasets",
        "gsm8k",
        "hellaswag",
        "--collect-only",
        "--parameters",
        str(settings),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--cache-dir",
        str(cache.directory),
        "--cache-mode",
        "reuse",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    select_main(monkeypatch, "parcae_370m", small=True)
    run_cached.main()
    saved = {path: path.read_bytes() for path in cache.directory.rglob("*.pt")}
    loads.clear()
    before = counts["forwards"]

    select_main(monkeypatch, "parcae_770m", small=True)
    run_cached.main()
    expected = {
        (dataset, name, steps)
        for dataset in ("gsm8k", "hellaswag")
        for name, steps in (
            ("parcae_770m", 3),
            ("parcae_770m", 2),
            ("parcae_770m_rec1", 1),
        )
    }
    assert len(loads) == len(expected) and set(loads) == expected
    assert counts["forwards"] - before == len(expected)
    assert all(path.read_bytes() == value for path, value in saved.items())
    # A third submission is entirely satisfied by the populated cache.
    loads.clear()
    before = counts["forwards"]
    run_cached.main()
    assert not loads and counts["forwards"] == before
