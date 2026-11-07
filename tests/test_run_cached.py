"""Dataset selection for the cache-population and figure runner."""

import json
import sys

import pytest

from experiments import run_cached
from experiments.support import configure_experiment


@pytest.mark.parametrize(
    "change, shared",
    [
        ({}, True),
        ({"seed": 99}, False),
        ({"limit": 1}, False),
        ({"dataset": "hellaswag"}, False),
        ({"dtype": "bfloat16"}, False),
        ({"compute_residual_entropy": True}, False),
    ],
)
def test_profile_plan_matches_scientific_inputs(monkeypatch, tmp_path, change, shared):
    convergence = configure_experiment(
        models=["ouro_1_4b"], num_steps=3, limit=2, device="cpu", dtype="float32"
    )
    from dataclasses import replace

    stages = replace(convergence, experiment="stages_of_inference", **change)
    if "dataset" in change:
        stages = replace(stages, task="hellaswag_likelihood")

    def setup(name, parameters, *, stop_tag):
        assert stop_tag == "configuration-end"
        config = convergence if name == "convergence" else stages
        return {"configs": {("ouro_1_4b", 3): config}}

    monkeypatch.setattr(run_cached, "_execute_notebook", setup)
    jobs = [
        (name, {"cache_dir": str(tmp_path), "cache_mode": "reuse"})
        for name in ("convergence", "stages_of_inference")
    ]
    run_cached.share_stage_profiles(jobs)
    assert jobs[0][1]["cache_profile_pairs"] == ([("ouro_1_4b", 3)] if shared else [])


def test_profile_plan_uses_notebook_defaults_without_collection(tmp_path):
    jobs = [
        (
            name,
            dict(
                mode="full",
                device="cpu",
                dtype="float32",
                cache_dir=str(tmp_path / "cache"),
                cache_mode="reuse",
            ),
        )
        for name in ("convergence", "stages_of_inference")
    ]
    run_cached.share_stage_profiles(jobs)
    from experiments.paper_models import PAPER_MODEL_GROUPS

    assert set(jobs[0][1]["cache_profile_pairs"]) == {
        (name, 128) for name in PAPER_MODEL_GROUPS["main"]
    }
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "stage_override", [{"cache_mode": "off"}, {"source_run": "saved"}]
)
def test_profile_plan_respects_disabled_or_historical_stages(tmp_path, stage_override):
    parameters = dict(cache_dir=str(tmp_path), cache_mode="reuse")
    jobs = [
        ("convergence", dict(parameters)),
        ("stages_of_inference", {**parameters, **stage_override}),
    ]
    run_cached.share_stage_profiles(jobs)
    assert "cache_profile_pairs" not in jobs[0][1]


@pytest.mark.parametrize(
    "arguments, datasets",
    [
        ([], ["gsm8k"]),
        (["--datasets", "gsm8k", "hellaswag"], ["gsm8k", "hellaswag"]),
        (["--datasets", "hellaswag"], ["hellaswag"]),
    ],
)
def test_dataset_selection_runs_other_notebooks_once(monkeypatch, arguments, datasets):
    runs = []

    def record(name, parameters, *, collect_only):
        assert collect_only
        assert parameters["mode"] == "full"
        runs.append((name, parameters))

    monkeypatch.setattr(run_cached, "run_notebook", record)
    monkeypatch.setattr(sys, "argv", ["run_cached", "--collect-only", *arguments])
    run_cached.main()

    for name in (
        "convergence",
        "stages_of_inference",
        "mixing_correlation",
        "cyclic_similarity",
        "entropy_lens",
        "bos_comparison",
        "isolating_stability",
    ):
        assert [p["dataset"] for n, p in runs if n == name] == datasets
    for name in (
        "multiple_choice",
        "trajectories",
        "neuron_detection",
        "initialisation",
        "limiting_behaviour",
    ):
        parameters = [p for n, p in runs if n == name]
        assert len(parameters) == 1
        assert "dataset" not in parameters[0] and "datasets" not in parameters[0]
    order = [(name, p.get("dataset")) for name, p in runs]
    for dataset in datasets:
        assert order.index(("convergence", dataset)) < order.index(
            ("stages_of_inference", dataset)
        )


def test_notebook_dataset_override_replaces_global_selection(monkeypatch, tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"convergence": {"dataset": "hellaswag", "limit": 7}})
    )
    runs = []

    def record(name, parameters, **kwargs):
        runs.append((name, parameters))

    monkeypatch.setattr(run_cached, "run_notebook", record)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cached",
            "--notebooks",
            "convergence",
            "stages_of_inference",
            "--datasets",
            "gsm8k",
            "hellaswag",
            "--parameters",
            str(settings),
        ],
    )
    run_cached.main()
    assert [(n, p["dataset"]) for n, p in runs] == [
        ("convergence", "hellaswag"),
        ("stages_of_inference", "gsm8k"),
        ("stages_of_inference", "hellaswag"),
    ]
    assert runs[0][1]["limit"] == 7
    assert "limit" not in runs[1][1]
