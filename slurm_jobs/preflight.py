"""Short GPU check of each paper checkpoint and packaged dataset before full jobs."""

import gc
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import torch

from experiments.support import (
    BASELINES,
    MODELS,
    STABILITY,
    configure_experiment,
    load_experiment_model,
)
from looped_internals.capture import compute_model_run_data, metrics_names_filter
from looped_internals.evaluation import load_mcq_dataset
from looped_internals.evaluation.lm_eval import get_task_manager


def expected_virtual_layers(name, structure, recurrences: int) -> int:
    """Count applications at the depth this checkpoint actually runs."""
    if name in BASELINES:
        recurrences = 1
    return sum(1 for _ in structure.virtual_layers(recurrences))


def main() -> None:
    config = configure_experiment(mode="quick", device="cuda", num_steps=2)
    run_id = os.environ.get("SLURM_JOB_ID") or datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    report_path = Path("experiments/runs") / f"preflight-{run_id}.json"
    report = {}
    for task in ["gsm8k_likelihood", "hellaswag_likelihood"]:
        loaded = get_task_manager().load([task])["tasks"][task]
        report[task] = {"examples": len(loaded.eval_docs)}
    for name in ["arc_easy", "arc_challenge", "hellaswag", "winogrande"]:
        examples = load_mcq_dataset(
            name, "test" if name.startswith("arc_") else "validation", max_examples=1
        )
        assert len(examples) == 1
        report[name] = {"choices": len(examples[0].choices)}
    for name in [*MODELS, *BASELINES, *STABILITY, "gpt2"]:
        print(f"PREFLIGHT {name}", flush=True)
        torch.manual_seed(42)
        bridge, structure = load_experiment_model(name, config)
        tokens = bridge.tokenizer.encode(
            "A short example sequence.", return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            logits, cache = bridge.run_with_cache(
                tokens,
                accumulate=True,
                names_filter=metrics_names_filter,
                device="cpu",
            )
        run = compute_model_run_data(
            cache, tokens, structure, compute_pairwise=False, compute_differences=True
        )[0]
        assert torch.isfinite(logits).all()
        assert torch.isfinite(run.get_metric("sink_scores")).all()
        expected = expected_virtual_layers(name, structure, config.num_steps)
        assert len(run.per_virtual_layer_metrics) == expected, (
            name,
            len(run.per_virtual_layer_metrics),
            expected,
        )
        report[name] = {"virtual_layers": expected, "tokens": tokens.shape[1]}
        report_path.write_text(json.dumps(report, indent=2))
        del bridge, logits, cache, run
        gc.collect()
        torch.cuda.empty_cache()
    print("PREFLIGHT PASSED", flush=True)


if __name__ == "__main__":
    main()
