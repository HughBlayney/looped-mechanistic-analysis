"""Disk reuse, stochastic restart equivalence, and shared scientific inputs."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from time import sleep

import numpy as np
import pytest
import torch
from test_experiments import (  # noqa: F401
    _generic_summary,
    _notebook_science,
    local_inference,
)
from test_notebooks import notebook_models  # noqa: F401

from experiments import cache as cache_module
from experiments import support
from experiments.cache import ExperimentCache, atomic_save, digest
from experiments.saved_runs import SavedRun


@pytest.fixture
def cached_inference(tmp_path, monkeypatch, request):
    counts = request.getfixturevalue("local_inference")
    cache = ExperimentCache(tmp_path / "cache", "reuse")
    monkeypatch.setattr(
        cache, "model", lambda name: (None, {"revision": "test", "name": name})
    )
    load = support._load_model
    monkeypatch.setattr(
        support, "_load_model", lambda name, config, **kwargs: load(name, config)
    )
    config = support.configure_experiment(
        models=["ouro_1_4b"],
        limit=3,
        num_steps=3,
        device="cpu",
        output_dir=str(tmp_path / "results"),
    )
    return cache, config, counts


def test_default_location_and_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv("LOOPED_INTERNALS_CACHE_DIR")
    monkeypatch.delenv("LOOPED_INTERNALS_CACHE_MODE")
    assert ExperimentCache().directory == Path(
        "/data-gauss/hugbla/looped-internals-cache"
    )
    monkeypatch.setenv("LOOPED_INTERNALS_CACHE_DIR", str(tmp_path / "shared"))
    assert ExperimentCache().directory == tmp_path / "shared"
    assert ExperimentCache(tmp_path / "explicit").directory == tmp_path / "explicit"
    with pytest.raises(ValueError, match="cache_mode"):
        ExperimentCache(mode="typo")


def test_cache_hit_and_new_model_do_not_repeat_inference(cached_inference):
    cache, config, counts = cached_inference
    first, _ = support.collect_metrics(config, _generic_summary, cache=cache)
    assert counts == {"loads": 1, "forwards": 3}
    full = replace(config, mode="full", output_dir=config.output_dir / "new")
    again, info = support.collect_metrics(full, _generic_summary, cache=cache)
    assert again == first
    assert counts == {"loads": 1, "forwards": 3}
    assert info["mode"] == "full"
    expanded = replace(config, models=("ouro_1_4b", "retrofitted_tinyllama"))
    support.collect_metrics(expanded, _generic_summary, cache=cache)
    assert counts == {"loads": 2, "forwards": 6}
    support.collect_metrics(config, _generic_summary, cache=cache, analysis_version=2)
    assert counts == {"loads": 3, "forwards": 9}


def test_interrupted_stochastic_collection_matches_uninterrupted(cached_inference):
    cache, config, counts = cached_inference
    expected, _ = support.collect_metrics(config, _generic_summary)
    counts.update(loads=0, forwards=0)
    calls = 0
    reduce = _generic_summary

    def interrupted(run):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return reduce(run)

    with pytest.raises(RuntimeError, match="interrupted"):
        support.collect_metrics(config, interrupted, cache=cache)
    assert counts == {"loads": 1, "forwards": 2}
    # Perturb every random stream and model-loading state before resuming.
    torch.rand(31)
    np.random.rand(17)
    resumed, _ = support.collect_metrics(config, reduce, cache=cache)
    assert resumed == expected
    assert counts == {"loads": 2, "forwards": 4}
    assert not list(cache.directory.rglob("progress.pt"))


@pytest.mark.parametrize("dataset", ["gsm8k", "hellaswag"])
def test_convergence_profiles_replace_stages_forwards(cached_inference, dataset):
    cache, config, counts = cached_inference
    config = replace(config, dataset=dataset, task=support.DATASETS[dataset])
    science = _notebook_science()
    support.collect_metrics(
        config,
        None,
        cache=cache,
        cache_reducer=science["reduce_convergence_cache"],
        names_filter_factory=science["recurrent_metric_names"],
        save_profiles=True,
    )
    stages = replace(
        config, experiment="stages_of_inference", compute_residual_entropy=False
    )

    def reduce(run):
        return {
            name: run.get_metric(name).tolist()
            for name in ("sink_rates", "mixing_scores", "residual_norms_mean")
        }

    cache.mode = "require"
    shared, _ = support.collect_metrics(stages, reduce, cache=cache)
    assert counts == {"loads": 1, "forwards": 3}
    assert not (cache.directory / "stages_of_inference").exists()
    cache.mode = "reuse"
    assert support.collect_metrics(stages, reduce, cache=cache)[0] == shared
    assert counts == {"loads": 1, "forwards": 3}
    fresh, _ = support.collect_metrics(stages, reduce)
    assert shared == fresh


def test_targeted_collection_skips_generic_metrics_and_scoring(
    cached_inference, monkeypatch
):
    from looped_internals.capture import extraction
    from looped_internals.evaluation.lm_eval import BridgeLM

    cache, config, counts = cached_inference
    expected, _ = support.collect_metrics(config, _generic_summary)
    counts.update(loads=0, forwards=0)
    science = _notebook_science()
    monkeypatch.setattr(
        extraction,
        "compute_model_run_data",
        lambda *a, **k: pytest.fail("unused metrics"),
    )
    monkeypatch.setattr(
        BridgeLM, "loglikelihood", lambda *a, **k: pytest.fail("unused scoring")
    )
    actual, _ = support.collect_metrics(
        config,
        None,
        cache=cache,
        cache_reducer=science["reduce_convergence_cache"],
        names_filter_factory=science["recurrent_metric_names"],
    )
    assert actual == expected
    assert counts == {"loads": 1, "forwards": 3}
    assert not (cache.directory / "layer_profiles").exists()
    # The output contract is unchanged: older complete summaries remain valid.
    cache.mode = "require"
    assert support.collect_metrics(config, _generic_summary, cache=cache)[0] == expected
    assert counts == {"loads": 1, "forwards": 3}


@pytest.mark.parametrize("initial_profiles", [False, True])
def test_targeted_resume_never_publishes_partial_profiles(
    cached_inference, initial_profiles
):
    cache, config, counts = cached_inference
    science = _notebook_science()
    reducer = science["reduce_convergence_cache"]
    expected, _ = support.collect_metrics(config, _generic_summary)
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return reducer(*args)

    with pytest.raises(RuntimeError, match="interrupted"):
        support.collect_metrics(
            config,
            None,
            cache=cache,
            cache_reducer=interrupted,
            names_filter_factory=science["recurrent_metric_names"],
            save_profiles=initial_profiles,
        )
    torch.rand(17)
    actual, _ = support.collect_metrics(
        config,
        None,
        cache=cache,
        cache_reducer=reducer,
        names_filter_factory=science["recurrent_metric_names"],
        save_profiles=True,
    )
    assert actual == expected
    profile_files = list((cache.directory / "layer_profiles").glob("*/result.pt"))
    assert len(profile_files) == int(initial_profiles)
    if initial_profiles:
        saved = torch.load(profile_files[0], weights_only=False)
        assert len(saved["profiles"]) == config.limit
    before = dict(counts)
    support.collect_metrics(
        config, None, cache=cache, cache_reducer=reducer, save_profiles=True
    )
    assert counts == before


def test_datasets_keep_separate_reusable_results(cached_inference):
    cache, config, counts = cached_inference
    expected = {}
    for dataset in ("gsm8k", "hellaswag"):
        selected = replace(config, dataset=dataset, task=support.DATASETS[dataset])
        expected[dataset] = support.collect_metrics(
            selected, _generic_summary, cache=cache
        )[0]
    assert counts == {"loads": 2, "forwards": 6}
    cache.mode = "require"
    for dataset in ("gsm8k", "hellaswag"):
        selected = replace(config, dataset=dataset, task=support.DATASETS[dataset])
        result, info = support.collect_metrics(selected, _generic_summary, cache=cache)
        assert result == expected[dataset]
        assert info["dataset"] == dataset
    assert counts == {"loads": 2, "forwards": 6}
    requests = [
        json.loads(p.read_text())
        for p in (cache.directory / "convergence").glob("*/request.json")
    ]
    assert {r["task"] for r in requests} == {"gsm8k_likelihood", "hellaswag_likelihood"}


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 99},
        {"num_steps": 4},
        {"dtype": "bfloat16"},
        {"bos_policy": "no_bos"},
    ],
)
def test_scientific_settings_change_the_key(cached_inference, change):
    cache, config, _ = cached_inference
    assert digest(cache.key(config, "ouro_1_4b")) != digest(
        cache.key(replace(config, **change), "ouro_1_4b")
    )


def test_require_and_off_never_create_cache_files(tmp_path):
    cache = ExperimentCache(tmp_path / "absent", "require")
    with pytest.raises(FileNotFoundError, match="Missing complete"):
        cache.run("test", {}, lambda: pytest.fail("inference"))
    assert not cache.directory.exists()
    cache = ExperimentCache(tmp_path / "off", "off")
    assert cache.run("test", {}, lambda: 7) == 7
    assert not cache.directory.exists()


def test_atomic_failure_preserves_the_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "progress.pt"
    atomic_save({"finished": 1}, path)

    def fail(value, stream):
        stream.write(b"partial file")
        raise OSError("disk write interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail)
        with pytest.raises(OSError):
            atomic_save({"finished": 2}, path)
    assert torch.load(path, weights_only=True) == {"finished": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_two_workers_compute_a_missing_entry_once(tmp_path):
    barrier = Barrier(2)
    calls = []

    def compute():
        calls.append(1)
        sleep(0.03)
        return {"value": 42}

    def worker():
        barrier.wait()
        return ExperimentCache(tmp_path, "reuse").run("shared", {}, compute)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: worker(), range(2)))
    assert results == [{"value": 42}] * 2
    assert calls == [1]


def test_local_checkpoint_content_changes_identity(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    weights = tmp_path / "pytorch_model.bin"
    weights.write_bytes(b"first")
    first, _ = cache_module._source_identity(str(tmp_path), "config.json")
    weights.write_bytes(b"other")
    second, _ = cache_module._source_identity(str(tmp_path), "config.json")
    assert first != second


def test_explicit_saved_run_matches_settings_and_keeps_original_files(tmp_path):
    settings = {
        "experiment": "convergence",
        "seed": 42,
        "dtype": "bfloat16",
        "mode": "full",
        "model_recurrences": [["old", 128]],
    }
    (tmp_path / "run.json").write_text(json.dumps(settings))
    torch.save(
        {"summaries": {("old", 128): [1]}, "run": {("old", 128): {"indices": [0]}}},
        tmp_path / "overview.pt",
    )
    before = (tmp_path / "overview.pt").read_bytes()
    saved = SavedRun(
        tmp_path,
        {
            **settings,
            "mode": "quick",
            "model_recurrences": [["old", 128], ["new", 128]],
        },
    )
    assert ("old", 128) in saved and ("new", 128) not in saved
    assert saved.info(("old", 128))["source_run"] == str(tmp_path)
    with pytest.raises(ValueError, match="seed"):
        SavedRun(tmp_path, {**settings, "seed": 9})
    with pytest.raises(ValueError, match="analysis_version"):
        SavedRun(tmp_path, settings, analysis_version=2)
    assert (tmp_path / "overview.pt").read_bytes() == before


def test_probe_training_and_evaluation_resume_exactly(tmp_path):
    from test_notebooks import TinyTokenizer

    from looped_internals.evaluation.datasets import MCQExample
    from looped_internals.evaluation.multiple_choice import (
        evaluate_multiple_choice_by_depth,
    )
    from looped_internals.evaluation.probes import train_probes
    from looped_internals.models import MinimalRecurrentConfig, ModelSpec, load_model

    torch.set_num_threads(1)
    bridge, structure = load_model(
        ModelSpec(
            MinimalRecurrentConfig(
                d_model=8, n_heads=2, n_layers=1, num_cycles=2, d_vocab=64
            )
        ),
        device="cpu",
    )
    tokenizer = TinyTokenizer()
    examples = [MCQExample(f"question {i}", ["yes", "no"], i % 2) for i in range(3)]
    args = (bridge, tokenizer, examples, structure, 2, 2, torch.device("cpu"))
    torch.manual_seed(42)
    expected = train_probes(*args, batch_size=1)
    state_path = tmp_path / "training.pt"

    def interrupt(state):
        atomic_save(state, state_path)
        raise RuntimeError("preempted")

    torch.manual_seed(42)
    with pytest.raises(RuntimeError, match="preempted"):
        train_probes(*args, batch_size=1, checkpoint=interrupt)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert state["next_batch"] == 1
    resumed = train_probes(*args, batch_size=1, resume=state)
    for a, b in zip(expected, resumed, strict=True):
        for name in a.state_dict():
            torch.testing.assert_close(
                a.state_dict()[name], b.state_dict()[name], rtol=0, atol=0
            )

    eval_args = (
        bridge,
        tokenizer,
        structure,
        examples,
        examples,
        2,
        2,
        torch.device("cpu"),
        "test",
        "test",
    )
    torch.manual_seed(10)
    expected_result, _ = evaluate_multiple_choice_by_depth(*eval_args, probes=expected)
    torch.manual_seed(10)
    with pytest.raises(RuntimeError, match="preempted"):
        evaluate_multiple_choice_by_depth(
            *eval_args, probes=resumed, checkpoint=interrupt
        )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert state["next_example"] == 1
    result, _ = evaluate_multiple_choice_by_depth(
        *eval_args, probes=resumed, resume=state
    )
    for metric in ("probe_accuracy", "logitlens_accuracy", "logitlens_norm_accuracy"):
        torch.testing.assert_close(
            result.get_metric(metric),
            expected_result.get_metric(metric),
            rtol=0,
            atol=0,
        )


def test_collect_only_executes_notebook_science_without_paper_exports(
    cached_inference, monkeypatch, tmp_path
):
    from experiments.run_cached import run_notebook

    cache, _config, counts = cached_inference
    monkeypatch.setattr(
        ExperimentCache,
        "model",
        lambda self, name: (None, {"revision": "test", "name": name}),
    )
    parameters = dict(
        mode="full",
        device="cpu",
        limit=1,
        model_recurrences=[["ouro_1_4b", 3]],
        output_dir=str(tmp_path / "results"),
        cache_dir=str(cache.directory),
        cache_mode="reuse",
    )
    directory = run_notebook("convergence", parameters, collect_only=True)
    assert (directory / "overview.pt").exists()
    assert not list(directory.glob("*.pdf"))
    before = dict(counts)
    run_notebook(
        "convergence", {**parameters, "cache_mode": "require"}, collect_only=True
    )
    assert counts == before


def test_population_shares_inference_and_then_runs_entirely_from_cache(
    cached_inference, monkeypatch, tmp_path
):
    import sys

    from experiments import paper_models, run_cached

    cache, _config, counts = cached_inference
    monkeypatch.setattr(
        ExperimentCache,
        "model",
        lambda self, name: (None, {"revision": "test", "name": name}),
    )
    monkeypatch.setattr(
        paper_models,
        "PAPER_MODEL_GROUPS",
        {"main": ("ouro_1_4b",), "appendix": (), "all": ("ouro_1_4b",)},
    )
    settings = tmp_path / "parameters.json"
    settings.write_text(
        json.dumps(
            {
                "convergence": {
                    "recurrences": 3,
                    "limit": 1,
                    "output_dir": str(tmp_path / "convergence"),
                },
                "stages_of_inference": {
                    "ouro_recurrences": 2,
                    "long_recurrences": 3,
                    "limit": 1,
                    "mixing_pair": ["ouro_1_4b", 2],
                    "ablation_pair": ["ouro_1_4b", 2],
                    "ablation_hook": "blocks.0.mlp.hook_out",
                    "output_dir": str(tmp_path / "stages"),
                },
            }
        )
    )
    arguments = [
        "run_cached",
        "--notebooks",
        "stages_of_inference",
        "convergence",
        "--collect-only",
        "--parameters",
        str(settings),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--cache-dir",
        str(cache.directory),
    ]
    monkeypatch.setattr(sys, "argv", [*arguments, "--cache-mode", "reuse"])
    run_cached.main()
    # Convergence + short stages + baseline + intervention. The matching long
    # stages request must not add a fifth model load or forward pass.
    assert counts == {"loads": 4, "forwards": 4}
    assert not list(tmp_path.rglob("*.pdf"))
    before = {path: path.read_bytes() for path in cache.directory.rglob("*.pt")}
    monkeypatch.setattr(sys, "argv", [*arguments, "--cache-mode", "require"])
    run_cached.main()
    assert counts == {"loads": 4, "forwards": 4}
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("dataset", ["gsm8k", "hellaswag"])
def test_layer_readouts_resume_and_load_without_a_model(
    cached_inference, monkeypatch, dataset
):
    from experiments import readout_support

    cache, config, counts = cached_inference
    config = replace(config, dataset=dataset, task=support.DATASETS[dataset])
    monkeypatch.setattr(readout_support, "get_task_manager", support.get_task_manager)

    def model(name, config, **kwargs):
        lm = support._load_model(name, config)
        return lm.bridge, lm.structure

    monkeypatch.setattr(readout_support, "load_experiment_model", model)

    def reduce(bridge, residual, layer, structure, **kwargs):
        return residual.float().mean().item()

    expected, _ = readout_support.collect_residual_readouts(config, reduce)
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 7:  # Six layers per example: the first example is checkpointed.
            raise RuntimeError("interrupted readout")
        return reduce(*args, **kwargs)

    with pytest.raises(RuntimeError, match="interrupted readout"):
        readout_support.collect_residual_readouts(config, interrupted, cache=cache)
    actual, _ = readout_support.collect_residual_readouts(config, reduce, cache=cache)
    assert actual == expected
    before = dict(counts)
    again, _ = readout_support.collect_residual_readouts(config, reduce, cache=cache)
    assert again == expected and counts == before


def test_parcae_cache_pins_weights_and_separate_tokenizer(tmp_path, monkeypatch):
    def identify(source, filename, **kwargs):
        revision = "weights-commit" if filename == "config.json" else "tokenizer-commit"
        return {"repo": source, "revision": revision}, revision

    monkeypatch.setattr(cache_module, "_source_identity", identify)
    spec, identity = ExperimentCache(tmp_path, "reuse").model("parcae_770m")
    assert spec.source.revision == "weights-commit"
    assert spec.source.tokenizer_revision == "tokenizer-commit"
    assert identity["source"]["weights"]["revision"] == "weights-commit"


@pytest.mark.parametrize("dataset", ["gsm8k", "hellaswag"])
def test_bos_conditions_have_separate_reusable_entries(
    tmp_path, monkeypatch, request, dataset
):
    request.getfixturevalue("notebook_models")
    from experiments.run_cached import run_notebook

    loads = []
    load = support._load_model

    def tracked(name, config, **kwargs):
        loads.append((name, config.bos_policy))
        return load(name, config)

    monkeypatch.setattr(support, "_load_model", tracked)
    monkeypatch.setattr(
        ExperimentCache,
        "model",
        lambda self, name: (None, {"name": name, "revision": "test"}),
    )
    parameters = dict(
        mode="quick",
        dataset=dataset,
        device="cpu",
        limit=1,
        model_recurrences=[["ouro_1_4b", 2]],
        output_dir=str(tmp_path / "results"),
        cache_dir=str(tmp_path / "cache"),
        cache_mode="reuse",
    )
    run_notebook("bos_comparison", parameters, collect_only=True)
    assert set(loads) == {("ouro_1_4b", "bos"), ("ouro_1_4b", "no_bos")}
    run_notebook(
        "bos_comparison", {**parameters, "cache_mode": "require"}, collect_only=True
    )
    assert len(loads) == 2


def test_resume_remaps_the_models_cuda_random_stream(monkeypatch):
    from looped_internals._random import restore_rng, rng_state

    calls = []
    expected = torch.tensor([4, 2], dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    def get_state(device):
        calls.append(("read", str(device)))
        return expected

    def set_state(state, device):
        torch.testing.assert_close(state, expected)
        calls.append(("restore", str(device)))

    monkeypatch.setattr(torch.cuda, "get_rng_state", get_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", set_state)
    state = rng_state("cuda:3")
    restore_rng(state, "cuda:0")
    assert calls == [("read", "cuda:3"), ("restore", "cuda:0")]
    assert rng_state("cpu")["cuda"] is None
