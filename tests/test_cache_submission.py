"""Check the actual submission script without contacting SLURM."""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def submission(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    script = tmp_path / "slurm_jobs" / "submit_cache.sh"
    script.parent.mkdir()
    shutil.copyfile(root / "slurm_jobs" / script.name, script)
    binary = tmp_path / "bin" / "sbatch"
    binary.parent.mkdir()
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "log = Path(os.environ['SUBMISSION_LOG'])\n"
        "lines = log.read_text().splitlines() if log.exists() else []\n"
        "if len(lines) == int(os.environ.get('FAIL_SUBMISSION', '-1')):\n"
        "    sys.exit('submission failed')\n"
        "with log.open('a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "print(1000 + len(lines))\n"
    )
    binary.chmod(0o755)
    log = tmp_path / "submissions.jsonl"
    monkeypatch.setenv("SUBMISSION_LOG", str(log))
    monkeypatch.setenv("PATH", f"{binary.parent}:{os.environ['PATH']}")
    monkeypatch.setenv("LOOPED_INTERNALS_CACHE_MODE", "off")
    monkeypatch.delenv("LOOPED_INTERNALS_CACHE_DIR", raising=False)
    monkeypatch.delenv("LOOPED_INTERNALS_MCQ_AFTER_JOB", raising=False)
    return script, log


@pytest.mark.parametrize("custom_directory", [False, True])
def test_both_datasets_are_partitioned_and_reruns_keep_job_records(
    submission, monkeypatch, tmp_path, custom_directory
):
    script, log = submission
    cache_directory = "/data-gauss/hugbla/looped-internals-cache"
    if custom_directory:
        cache_directory = str(tmp_path / "shared cache")
        monkeypatch.setenv("LOOPED_INTERNALS_CACHE_DIR", cache_directory)
    expected = {
        (dataset, notebooks)
        for dataset in ("gsm8k", "hellaswag")
        for notebooks in (
            ("convergence", "stages_of_inference"),
            ("isolating_stability",),
            ("entropy_lens",),
            ("cyclic_similarity",),
            ("bos_comparison",),
        )
    } | {
        ("shared", (name,))
        for name in (
            "limiting_behaviour",
            "multiple_choice",
            "initialisation",
            "trajectories",
            "neuron_detection",
        )
    }
    # Run from outside the repository; the script establishes the submission cwd.
    for _ in range(2):
        subprocess.run(["bash", str(script)], cwd=tmp_path.parent, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 30 and calls[:15] == calls[15:]
    selected = []
    for args in calls[:15]:
        assert args[0] == "--parsable"
        assert "slurm_jobs/populate_cache.job" in args
        assert args[args.index("--cache-dir") + 1] == cache_directory
        assert args[args.index("--cache-mode") + 1] == "reuse"
        dataset = (
            args[args.index("--datasets") + 1] if "--datasets" in args else "shared"
        )
        selected.append((dataset, tuple(args[args.index("--notebooks") + 1 :])))
    assert set(selected) == expected and len(selected) == len(expected)
    manifests = list(
        (tmp_path / "experiments" / "runs").glob("cache-submissions-*.tsv")
    )
    assert len(manifests) == 2
    recorded_ids = []
    for path in manifests:
        with path.open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert {
            (row["dataset"], tuple(row["notebooks"].split())) for row in rows
        } == expected
        recorded_ids.extend(int(row["job_id"]) for row in rows)
    assert sorted(recorded_ids) == list(range(1000, 1030))


def test_submission_failure_preserves_ids_already_submitted(
    submission, monkeypatch, tmp_path
):
    script, log = submission
    monkeypatch.setenv("FAIL_SUBMISSION", "2")
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert result.returncode != 0 and "submission failed" in result.stderr
    assert len(log.read_text().splitlines()) == 2
    (manifest,) = (tmp_path / "experiments" / "runs").glob("cache-submissions-*.tsv")
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["job_id"] for row in rows] == ["1000", "1001"]


def test_figures_wait_for_every_cache_job(submission, monkeypatch, tmp_path):
    script, log = submission
    directory = str(tmp_path / "shared cache")
    monkeypatch.setenv("LOOPED_INTERNALS_CACHE_DIR", directory)
    subprocess.run(["bash", str(script), "--figures"], check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 16
    render = calls[-1]
    assert "slurm_jobs/render_figures.job" in render
    assert "--dependency=afterok:" + ":".join(map(str, range(1000, 1015))) in render
    assert "--kill-on-invalid-dep=yes" in render
    assert render[render.index("--cache-dir") + 1] == directory
    (manifest,) = (tmp_path / "experiments" / "runs").glob("cache-submissions-*.tsv")
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert rows[-1] == {"dataset": "both", "notebooks": "figures", "job_id": "1015"}


def test_only_multiple_choice_waits_for_an_existing_job(submission, monkeypatch):
    script, log = submission
    monkeypatch.setenv("LOOPED_INTERNALS_MCQ_AFTER_JOB", "12345")
    subprocess.run(["bash", str(script), "--figures"], check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    for args in calls[:15]:
        dependencies = [a for a in args if a.startswith("--dependency=")]
        expected = (
            ["--dependency=afterany:12345"] if args[-1] == "multiple_choice" else []
        )
        assert dependencies == expected


def test_render_job_requires_cached_gpu_results_without_requesting_a_gpu(
    tmp_path, monkeypatch
):
    job = Path(__file__).parents[1] / "slurm_jobs" / "render_figures.job"
    assert not any(
        line.startswith("#SBATCH") and ("--gres" in line or "--gpu" in line)
        for line in job.read_text().splitlines()
    )
    binary = tmp_path / ".venv" / "bin" / "python"
    binary.parent.mkdir(parents=True)
    record = tmp_path / "arguments.json"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['RENDER_ARGUMENTS']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("SLURM_SUBMIT_DIR", str(tmp_path))
    monkeypatch.setenv("RENDER_ARGUMENTS", str(record))
    subprocess.run(
        ["bash", str(job), "--cache-dir", str(tmp_path / "cache")], check=True
    )
    args = json.loads(record.read_text())
    assert args[:2] == ["-m", "experiments.run_cached"]
    assert args[args.index("--datasets") + 1 : args.index("--datasets") + 3] == [
        "gsm8k",
        "hellaswag",
    ]
    assert args[args.index("--device") + 1] == "cuda"
    assert args[args.index("--dtype") + 1] == "bfloat16"
    assert args[args.index("--cache-mode") + 1] == "require"
    assert "--collect-only" not in args
