"""Checkpoint probe training and evaluation independently for each benchmark."""

from dataclasses import asdict

import torch

from experiments.support import _set_seed, record_tokenization, write_tokenization
from looped_internals._random import restore_rng, rng_state
from looped_internals.evaluation.probes import MultipleChoiceProbe
from looped_internals.models.tokenization import tokenizer_metadata


def restore_probes(states, width):
    probes = [MultipleChoiceProbe(width) for _ in states]
    for probe, state in zip(probes, states, strict=True):
        probe.load_state_dict(state)
        probe.eval()
    return probes


def collect_benchmark(
    session,
    config,
    cache,
    dataset,
    train,
    test,
    *,
    batch_size,
    learning_rate,
    compute_coda,
    project_pre_coda,
    analysis_version,
    run_dir,
):
    from looped_internals.evaluation import evaluate_multiple_choice_by_depth
    from looped_internals.evaluation.probes import train_probes

    name = session.name
    n_classes = 2 if dataset == "winogrande" else 4
    train_key = cache.key(
        config,
        name,
        analysis_version=analysis_version,
        dataset=dataset,
        train=train,
        n_classes=n_classes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=1,
    )
    key = {
        **train_key,
        "test": [asdict(example) for example in test],
        "compute_coda": compute_coda,
        # Keep earlier keys unchanged for runs without the projected readout.
        **({"project_pre_coda": True} if project_pre_coda else {}),
    }
    with cache.entry("multiple_choice", key, device=config.device) as entry:
        if entry.complete:
            saved = entry.load()
            write_tokenization(run_dir, name, saved["tokenization"])
            return (
                saved["result"],
                restore_probes(saved["probes"], saved["width"]),
                saved["width"],
            )
        bridge, structure = session.get()
        record_tokenization(run_dir, name, bridge)
        with cache.entry("mcq_probes", train_key, device=config.device) as training:
            if training.complete:
                trained = training.load()
                probes = restore_probes(trained["probes"], bridge.cfg.d_model)
                restore_rng(trained["rng"], config.device)
            else:
                _set_seed(config.seed)
                progress = training.progress
                probes = train_probes(
                    bridge,
                    bridge.tokenizer,
                    train,
                    structure,
                    config.num_steps,
                    n_classes,
                    torch.device(config.device),
                    batch_size=batch_size,
                    lr=learning_rate,
                    resume=progress["value"] if progress else None,
                    checkpoint=training.checkpoint if cache.enabled else None,
                )
                training.save(
                    {
                        "probes": [p.state_dict() for p in probes],
                        "rng": rng_state(config.device),
                    }
                )
        progress = entry.progress
        result, probes = evaluate_multiple_choice_by_depth(
            bridge,
            bridge.tokenizer,
            structure,
            train,
            test,
            config.num_steps,
            n_classes,
            torch.device(config.device),
            name,
            dataset,
            probes=probes,
            compute_coda=compute_coda,
            project_pre_coda=project_pre_coda,
            resume=progress["value"] if progress else None,
            checkpoint=entry.checkpoint if cache.enabled else None,
        )
        width = bridge.cfg.d_model
        entry.save(
            {
                "result": result,
                "probes": [p.state_dict() for p in probes],
                "width": width,
                "tokenization": {
                    "bos_policy": "native",
                    **tokenizer_metadata(bridge.tokenizer),
                },
            }
        )
        return result, probes, width
