"""Stream layer readouts over the notebooks' seeded likelihood-task inputs."""

import gc
import hashlib
from dataclasses import asdict

import torch
from tqdm.auto import tqdm

from experiments.support import _set_seed, load_experiment_model, sample_indices
from looped_internals.evaluation.lm_eval import BridgeLM, get_task_manager
from looped_internals.models.tokenization import tokenizer_metadata


def collect_residual_readouts(
    config, reduce_layer, *, cache=None, analysis_version=1, readout_options=None
):
    from experiments.cache import ExperimentCache

    cache = cache if cache is not None else ExperimentCache(mode="off")
    (name,) = config.models
    task = get_task_manager().load([config.task])["tasks"][config.task]
    indices = sorted(sample_indices(len(task.eval_docs), config.limit, config.seed))
    texts = [
        (task.doc_to_text(task.eval_docs[i]), task.doc_to_target(task.eval_docs[i]))
        for i in indices
    ]
    key = cache.key(
        config,
        name,
        analysis_version=analysis_version,
        task=config.task,
        indices=indices,
        texts=texts,
        readout_options=readout_options,
    )
    with cache.entry("entropy_lens", key, device=config.device) as entry:
        if entry.complete:
            examples, info = entry.load()
        else:
            examples, info = _collect_readouts(
                config, reduce_layer, cache, entry, task, indices
            )
            entry.save((examples, info))
    info.update(
        mode=config.mode, output_dir=str(config.output_dir), cache=cache.settings
    )
    return examples, info


def _collect_readouts(config, reduce_layer, cache, entry, task, indices):
    """Reduce each block output immediately; retain no activation history.

    ``reduce_layer(bridge, residual, layer, structure, input_ids=...)`` returns
    a compact value. Input IDs retain the complete causal context for readouts
    that need token-dependent components. Inputs match the zero-shot likelihood
    tasks, including their reference continuation and native tokenization.
    Each config selects one model.
    """
    (name,) = config.models
    _set_seed(config.seed)
    bridge, structure = (
        load_experiment_model(name, config, cache=cache)
        if cache.enabled
        else load_experiment_model(name, config)
    )
    lm = BridgeLM(
        bridge, structure, compute_metrics=False, bos_policy=config.bos_policy
    )
    bridge.eval()
    layout = list(structure.virtual_layers(config.num_steps))
    info = {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "indices": indices,
        "sample_count": len(indices),
        "split_size": len(task.eval_docs),
        "layer_groups": [asdict(group) for group in structure.groups],
        "virtual_layers": [asdict(layer) for layer in layout],
        "tokenization": {
            **tokenizer_metadata(bridge.tokenizer),
            "bos_policy": config.bos_policy,
            "model_revision": getattr(
                bridge.original_model.config, "_commit_hash", None
            ),
            "inputs": [],
        },
    }
    examples = []
    values = []
    reducing = False

    def read_layer(residual, hook):
        nonlocal reducing
        # A readout may itself run model layers (e.g. the coda lens).
        if reducing:
            return
        layer = layout[len(values)]
        if hook.name != f"{layer.hook_prefix}.hook_out":
            raise ValueError("Observed block order differs from the model structure")
        reducing = True
        try:
            values.append(
                reduce_layer(bridge, residual, layer, structure, input_ids=input_ids)
            )
        finally:
            reducing = False

    records = entry.records()
    if records:
        from looped_internals._random import restore_rng

        examples = [record["value"]["values"] for record in records]
        info["tokenization"]["inputs"] = [
            record["value"]["input"] for record in records
        ]
        restore_rng(records[-1]["rng"], config.device)
    del records
    try:
        with torch.inference_mode():
            for index in tqdm(indices[len(examples) :], desc=f"{name}: layer readouts"):
                doc = task.eval_docs[index]
                context, continuation = lm._encode_pair(
                    task.doc_to_text(doc), task.doc_to_target(doc)
                )
                input_ids = torch.tensor([context + continuation], device=lm.device)
                values = []
                bridge.run_with_hooks(
                    input_ids,
                    fwd_hooks=[(hook, read_layer) for hook in structure.hook_names()],
                    return_type=None,
                )
                if len(values) != len(layout):
                    raise ValueError("Forward pass did not visit every virtual layer")
                examples.append(values)
                ids = input_ids.cpu()
                info["tokenization"]["inputs"].append(
                    {
                        "dataset_index": index,
                        "length": ids.numel(),
                        "context_length": len(context),
                        "first_ids": ids.flatten()[:8].tolist(),
                        "sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                    }
                )
                entry.append(
                    {"values": values, "input": info["tokenization"]["inputs"][-1]}
                )
    finally:
        del lm
        bridge = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return examples, info
