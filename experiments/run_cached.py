"""Populate experiment caches using the scientific code in the notebooks."""

import argparse
import gc
import json
from pathlib import Path

from experiments.cache import ExperimentCache

NOTEBOOKS = (
    "convergence",
    "stages_of_inference",
    "mixing_correlation",
    "cyclic_similarity",
    "entropy_lens",
    "multiple_choice",
    "trajectories",
    "neuron_detection",
    "initialisation",
    "isolating_stability",
    "bos_comparison",
    "limiting_behaviour",
)
DATASET_NOTEBOOKS = {
    "convergence",
    "stages_of_inference",
    "mixing_correlation",
    "cyclic_similarity",
    "entropy_lens",
    "isolating_stability",
    "bos_comparison",
}


def _execute_notebook(name, parameters, *, stop_tag=None):
    """Apply overrides after parameters and stop at an explicit notebook tag."""
    path = Path(__file__).with_name(f"{name}.ipynb")
    notebook = json.loads(path.read_text())
    namespace = {"__name__": "__main__"}
    tagged = any(
        stop_tag in cell.get("metadata", {}).get("tags", [])
        for cell in notebook["cells"]
    )
    if stop_tag and not tagged:
        raise ValueError(f"Notebook has no {stop_tag} tag: {name}")
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tags = cell.get("metadata", {}).get("tags", [])
        exec(compile(source, f"{path}:{cell['id']}", "exec"), namespace)
        if "parameters" in tags:
            namespace.update(parameters)
        if stop_tag in tags:
            break
    return namespace


def run_notebook(name, parameters, *, collect_only=False):
    namespace = _execute_notebook(
        name, parameters, stop_tag="collection-end" if collect_only else None
    )
    return namespace.get("run_dir")


def share_stage_profiles(jobs):
    """Request profiles only for matching convergence/stages inputs.

    Execute configuration cells, never collection, so the notebooks remain the
    source of truth for model pairs, entropy requirements and resolved defaults.
    An explicit ``cache_profile_pairs`` override takes precedence over planning.
    """
    stages = []
    for name, parameters in jobs:
        if name != "stages_of_inference" or parameters.get("source_run"):
            continue
        cache = ExperimentCache(parameters["cache_dir"], parameters["cache_mode"])
        if cache.enabled:
            setup = _execute_notebook(name, parameters, stop_tag="configuration-end")
            stages.append((cache.directory.resolve(), setup["configs"]))
    for name, parameters in jobs:
        if name != "convergence" or "cache_profile_pairs" in parameters or not stages:
            continue
        cache = ExperimentCache(parameters["cache_dir"], parameters["cache_mode"])
        if not cache.enabled:
            continue
        setup = _execute_notebook(name, parameters, stop_tag="configuration-end")
        pairs = set()
        for directory, stage_configs in stages:
            if directory != cache.directory.resolve():
                continue
            for pair in setup["configs"].keys() & stage_configs.keys():
                convergence, stage = setup["configs"][pair], stage_configs[pair]
                if stage.compute_residual_entropy:
                    continue
                if (
                    all(
                        getattr(convergence, field) == getattr(stage, field)
                        for field in ("task", "limit", "seed", "dtype", "bos_policy")
                    )
                    and convergence.device.split(":")[0] == stage.device.split(":")[0]
                ):
                    pairs.add(pair)
        parameters["cache_profile_pairs"] = sorted(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", nargs="+", choices=NOTEBOOKS, default=NOTEBOOKS)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("gsm8k", "hellaswag"),
        default=["gsm8k"],
        help="Datasets for likelihood analyses (default: gsm8k); other notebooks run once",
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-mode", choices=("reuse", "require", "off"))
    parser.add_argument(
        "--parameters",
        type=Path,
        help="JSON mapping notebook names to parameter overrides",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Stop after collection; do not export paper figures",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="List saved entries without resolving models or datasets",
    )
    args = parser.parse_args()
    cache = ExperimentCache(args.cache_dir, args.cache_mode)
    if args.status:
        for path in sorted(cache.directory.glob("**/request.json")):
            request = json.loads(path.read_text())
            state = (
                "complete"
                if (path.parent / "result.pt").exists()
                else "partial"
                if (path.parent / "progress.pt").exists()
                or (path.parent / "parts").exists()
                else "pending"
            )
            print(state, request.get("model"), request.get("num_steps"), path.parent)
        return
    overrides = json.loads(args.parameters.read_text()) if args.parameters else {}
    notebooks = list(dict.fromkeys(args.notebooks))
    if "convergence" in notebooks and "stages_of_inference" in notebooks:
        notebooks.remove("convergence")
        notebooks.insert(notebooks.index("stages_of_inference"), "convergence")
    jobs = []
    for name in notebooks:
        parameters = {
            "mode": "full",
            "device": args.device,
            "dtype": args.dtype,
            "cache_dir": str(cache.directory),
            "cache_mode": cache.mode,
            **overrides.get(name, {}),
        }
        datasets = [None]
        if name in DATASET_NOTEBOOKS:
            # A notebook-specific dataset override replaces the global selection.
            datasets = list(
                dict.fromkeys(
                    [parameters["dataset"]]
                    if "dataset" in parameters
                    else args.datasets
                )
            )
        for dataset in datasets:
            selected = {**parameters, "dataset": dataset} if dataset else parameters
            jobs.append((name, selected))
    if any(name == "convergence" for name, _ in jobs):
        share_stage_profiles(jobs)
    for name, parameters in jobs:
        dataset = parameters.get("dataset")
        label = f"{name}/{dataset}" if dataset else name
        print(f"Running {label}: cache_mode={parameters['cache_mode']}", flush=True)
        if name == "convergence" and parameters.get("cache_profile_pairs"):
            print(
                f"Sharing stages profiles for: {parameters['cache_profile_pairs']}",
                flush=True,
            )
        run_dir = run_notebook(name, parameters, collect_only=args.collect_only)
        print(f"Saved {label}: {run_dir}", flush=True)
        import matplotlib.pyplot as plt

        plt.close("all")
        gc.collect()


if __name__ == "__main__":
    main()
