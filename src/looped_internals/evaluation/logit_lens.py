"""Intermediate vocabulary readouts and multiple-choice likelihood scoring."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from looped_internals.structure import ModelLayerStructure

from .datasets import MCQExample


def _tokenize_mcq(
    tokenizer,
    example: MCQExample,
) -> tuple[list[int], list[list[int]]]:
    """Tokenise an MCQ example into context and per-choice continuation ids.

    Returns:
        ``(ctx_ids, cont_ids_list)`` where ``ctx_ids`` is the tokenised context
        and ``cont_ids_list[i]`` are the continuation tokens for choice ``i``.
        For full-sentence choices (WinoGrande), use the checkpoint's native
        encoding. Only an automatically inserted BOS becomes shared context.
    """
    if example.full_sentence_choices:
        from looped_internals.models.tokenization import tokenizer_metadata

        info = tokenizer_metadata(tokenizer)
        ctx_ids = [info["bos_token_id"]] if info["prepends_bos"] else []
        cont_ids_list = [
            tokenizer.encode(c, add_special_tokens=True)[len(ctx_ids) :]
            for c in example.choices
        ]
    else:
        ctx_ids = tokenizer.encode(example.context, add_special_tokens=True)
        cont_ids_list = [
            tokenizer.encode(c, add_special_tokens=False) for c in example.choices
        ]

    return ctx_ids, cont_ids_list


def _build_right_padded_batch(
    ctx_ids: list[int],
    cont_ids_list: list[list[int]],
    pad_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Return right-padded choice sequences with shape ``(n_choices, max_len)``."""
    max_len = len(ctx_ids) + max(len(c) for c in cont_ids_list)
    n_choices = len(cont_ids_list)

    input_ids = torch.full(
        (n_choices, max_len), pad_id, dtype=torch.long, device=device
    )
    for i, cont in enumerate(cont_ids_list):
        row = ctx_ids + cont
        input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)

    return input_ids


@torch.no_grad()
def logit_lens(bridge, hidden_states: torch.Tensor) -> torch.Tensor:
    """Apply the model's final layer norm + unembedding to intermediate hidden states.

    Args:
        bridge: ``TransformerBridge`` instance.
        hidden_states: ``(B, T, d_model)`` tensor.

    Returns:
        Logits of shape ``(B, T, vocab_size)`` in float32.
    """
    h = bridge.ln_final(hidden_states)
    return bridge.unembed(h).float()


def _continuation_score(
    logits: torch.Tensor, context_length: int, continuation: list[int]
) -> tuple[float, float]:
    """Sum/average next-token log-probabilities, excluding any unpredicted prefix."""
    positions = torch.arange(
        context_length - 1,
        context_length + len(continuation) - 1,
        device=logits.device,
    )
    valid = (positions >= 0) & (positions < logits.shape[0])
    targets = torch.tensor(continuation, dtype=torch.long, device=logits.device)[valid]
    log_probs = F.log_softmax(logits[positions[valid]].float(), dim=-1)
    total = log_probs.gather(1, targets[:, None]).sum().item()
    return total, total / max(len(targets), 1)


def _choose_options(
    logits: torch.Tensor, context_length: int, continuations: list[list[int]]
) -> tuple[int, int]:
    """Return the options with the highest summed and mean log-probabilities."""
    scores = np.array(
        [
            _continuation_score(logits[i], context_length, continuation)
            for i, continuation in enumerate(continuations)
        ]
    )
    return int(np.argmax(scores[:, 0])), int(np.argmax(scores[:, 1]))


@torch.no_grad()
def score_logitlens(
    bridge,
    tokenizer,
    example: MCQExample,
    structure: ModelLayerStructure,
    n_recurrences: int,
    device: torch.device,
    forward_kwargs: dict | None = None,
    *,
    compute_coda: bool = False,
    project_pre_coda: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Score MCQ choices with LogitLens at every virtual layer.

    Runs a single batched forward pass (one row per choice, right-padded) and
    applies ``ln_final + unembed`` to each virtual layer's hidden states to
    compute continuation log-probabilities.

    Args:
        bridge: TransformerBridge.
        tokenizer: Model tokenizer.
        example: MCQExample to score.
        structure: Model layer groups and hook names.
        n_recurrences: Number of applications of each recurrent layer.
        device: Device for inference.
        forward_kwargs: Extra kwargs for bridge.forward (e.g. ``num_steps``).
        compute_coda: Also apply the native coda lens to prelude/core states,
            reusing the same residual cache. Each option's complete unpadded
            sequence is read independently. Supports Huginn, retrofits, and Parcae.
        project_pre_coda: Also score a Parcae readout in which prelude and core
            states first pass through the learned core-output projection C.
            Observed coda states, which C already precedes, are read directly.

    Returns:
        ``(predicted, predicted_norm, predicted_coda, predicted_projected)``
        arrays of shape ``(n_virtual_layers,)``. The first two use summed and
        mean continuation log-probabilities; the optional readouts use summed
        likelihood. Observed coda states are NaN in the coda prediction because
        the suffix must not be applied twice.
    """
    forward_kwargs = forward_kwargs or {}
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0

    ctx_ids, cont_ids_list = _tokenize_mcq(tokenizer, example)
    ctx_len = len(ctx_ids)
    input_ids = _build_right_padded_batch(ctx_ids, cont_ids_list, pad_id, device)

    virtual_layers = list(structure.virtual_layers(n_recurrences))
    last_core_layer = max(
        (layer.layer_index for layer in virtual_layers if layer.iteration is not None),
        default=-1,
    )
    if compute_coda:
        from .coda_lens import coda_lens

    _, cache = bridge.run_with_cache(
        input_ids,
        accumulate=True,
        names_filter=structure.hook_names(),
        device="cpu",
        **forward_kwargs,
    )

    n_virtual_layers = len(virtual_layers)
    predicted = np.empty(n_virtual_layers, dtype=np.int32)
    predicted_norm = np.empty(n_virtual_layers, dtype=np.int32)
    predicted_coda = np.full(n_virtual_layers, np.nan) if compute_coda else None
    predicted_projected = (
        np.empty(n_virtual_layers, dtype=np.int32) if project_pre_coda else None
    )

    for k, layer in enumerate(virtual_layers):
        h = cache[layer.residual_key].to(device)  # (n_choices, T, d_model)
        before_coda = layer.layer_index <= last_core_layer
        logits = logit_lens(bridge, h)  # (n_choices, T, vocab) float32
        predicted[k], predicted_norm[k] = _choose_options(
            logits, ctx_len, cont_ids_list
        )
        del logits

        if predicted_projected is not None:
            if before_coda:
                logits = logit_lens(bridge, bridge.core_projection(h))
                predicted_projected[k], _ = _choose_options(
                    logits, ctx_len, cont_ids_list
                )
                del logits
            else:
                predicted_projected[k] = predicted[k]

        if predicted_coda is not None and before_coda:
            coda_scores = np.empty(len(cont_ids_list))
            for i, cont in enumerate(cont_ids_list):
                length = ctx_len + len(cont)
                # Preserve causal positions and all context; remove right-padding.
                logits = coda_lens(
                    bridge,
                    h[i : i + 1, :length],
                    input_ids=input_ids[i : i + 1, :length],
                    from_core=layer.iteration is not None,
                )
                coda_scores[i], _ = _continuation_score(logits[0], ctx_len, cont)
                del logits
            predicted_coda[k] = int(np.argmax(coda_scores))

    return predicted, predicted_norm, predicted_coda, predicted_projected
