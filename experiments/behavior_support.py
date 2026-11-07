"""Shared utilities for GSM8K per-token limiting-behavior analysis.

Extracted from gsm8k_behaviors.py so the same model-loading, tokenisation,
and forward-pass logic can be reused in notebooks and other scripts without
duplication.

Typical notebook usage::

    from experiments.behavior_support import BehaviorRunConfig, load_model, run_example
    from datasets import load_dataset

    behavior_dir = Path("data/huginn_huginn_behaviors")
    config = BehaviorRunConfig.load(behavior_dir)
    bridge, structure, forward_kwargs, system_token_count = load_model(config,
    device="cuda")

    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    run_data = run_example(
        bridge, structure, forward_kwargs,
        question=gsm8k[42]["question"],
        config=config,
        system_token_count=system_token_count,
        device="cuda",
    )
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import looped_internals  # noqa: F401  — integration initialized by model loading
from looped_internals import models
from looped_internals.capture import compute_model_run_data, metrics_names_filter
from looped_internals.results import ModelRunData
from looped_internals.structure import ModelLayerStructure

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MODEL_SLUGS: frozenset[str] = frozenset(
    ["huginn", "retrofitted-llama", "retrofitted-olmo", "retrofitted-tinyllama"]
)

_MODEL_NAMES = {
    "huginn": "raven",
    "retrofitted-llama": "retrofitted_llama_32",
    "retrofitted-olmo": "retrofitted_olmo_32",
    "retrofitted-tinyllama": "retrofitted_tinyllama_32",
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BehaviorRunConfig:
    """Complete specification of one gsm8k_behaviors.py run.

    Serialised to ``config.json`` inside each behavior directory so that
    downstream consumers (notebooks, analysis scripts) can reconstruct the
    exact run parameters from just the directory path.

    Parameters
    ----------
    model_slug:
        One of the VALID_MODEL_SLUGS values.
    system_prompt:
        Full text of the system prompt, or ``""`` for no system prompt.
    prompt_stem:
        Filename stem of the system-prompt file, or ``"noprompt"``.
    n_recurrences:
        Number of recurrence steps (passed to the model at load time).
    add_special_tokens:
        Explicit override for ``tokenizer.encode``. None follows native special
        tokens for plain text and avoids adding them again to chat templates.
    replace_prompt_with_pad:
        Whether system-prompt tokens were replaced by pad tokens.
    lookback:
        ``recurrence_lookback_distance`` used when classifying behavior.
    threshold:
        ``closeness_threshold`` used when classifying behavior.
    """

    model_slug: str
    system_prompt: str  # "" for no system prompt
    prompt_stem: str  # "noprompt" when system_prompt is ""
    n_recurrences: int = 128
    add_special_tokens: bool | None = None
    replace_prompt_with_pad: bool = False
    lookback: int = 32
    threshold: float = 1e-2

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def behavior_dir_name(self) -> str:
        """Canonical directory name for this run's output."""
        pad_suffix = "_padded" if self.replace_prompt_with_pad else ""
        return f"{self.model_slug}_{self.prompt_stem}{pad_suffix}_behaviors"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, behavior_dir: Path) -> None:
        """Write ``config.json`` into *behavior_dir*."""
        (behavior_dir / "config.json").write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, behavior_dir: Path) -> BehaviorRunConfig:
        """Read ``config.json`` from *behavior_dir* and return a config instance."""
        meta = json.loads((behavior_dir / "config.json").read_text())
        return cls(**meta)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    config: BehaviorRunConfig,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Any, ModelLayerStructure, dict, int]:
    """Load the bridge + block structure for the model described by *config*.

    Returns
    -------
    bridge
        The TransformerLens bridge (already ``.eval()``).
    structure
        :class:`~looped_internals.structure.ModelLayerStructure`.
    forward_kwargs
        Empty: the loader sets the inference recurrence count in the model config.
    system_token_count
        Number of tokens consumed by the system prompt under this model's
        tokenizer (0 when ``config.system_prompt`` is empty).
    """
    bridge, structure = models.load_model(
        _MODEL_NAMES[config.model_slug],
        num_steps=config.n_recurrences,
        device=device,
        dtype=dtype,
    )
    forward_kwargs: dict = {}

    tokenizer = bridge.tokenizer
    assert tokenizer is not None, "Bridge must have a tokenizer"
    system_token_count = count_system_tokens(
        tokenizer, config.system_prompt, config.add_special_tokens
    )
    return bridge, structure, forward_kwargs, system_token_count


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------


def count_system_tokens(
    tokenizer, system_prompt: str, add_special_tokens: bool | None
) -> int:
    """Return the number of tokens that precede the user question content.

    This includes the system section (if any) AND the user-turn header emitted
    by the chat template (e.g. ``<|start_header_id|>user<|end_header_id|>\\n\\n``
    in Llama-style templates).  A null-byte placeholder is inserted as the user
    content so we can locate the exact text boundary before the question.

    Falls back to encoding the raw system prompt text when ``apply_chat_template``
    is unavailable. With no system prompt, count only a native automatic BOS.
    """
    _PLACEHOLDER = "\x00"  # null byte — won't appear in prompts or template boilerplate
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": _PLACEHOLDER})
    try:
        full_text: str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if _PLACEHOLDER not in full_text:
            raise ValueError("placeholder not found in template output")
        prefix_text = full_text[: full_text.index(_PLACEHOLDER)]
        return len(
            tokenizer.encode(prefix_text, add_special_tokens=bool(add_special_tokens))
        )
    except Exception:
        # No chat template: only the raw system prompt precedes the question.
        effective = True if add_special_tokens is None else add_special_tokens
        if not system_prompt:
            from looped_internals.models.tokenization import tokenizer_metadata

            return int(effective and tokenizer_metadata(tokenizer)["prepends_bos"])
        return len(tokenizer.encode(system_prompt, add_special_tokens=effective))


def build_input_ids(
    tokenizer,
    question: str,
    config: BehaviorRunConfig,
    system_token_count: int,
    device: str,
) -> Tensor:
    """Tokenise *question* (with optional system prompt) into ``input_ids``.

    Respects ``config.system_prompt``, ``config.add_special_tokens``, and
    ``config.replace_prompt_with_pad``.  Falls back gracefully when the
    tokenizer has no chat template.
    """
    templated = True
    if config.system_prompt:
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": question},
        ]
        try:
            input_text: str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            templated = False
            input_text = config.system_prompt + "\n\n" + question
    else:
        try:
            input_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            templated = False
            input_text = question

    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt",
        add_special_tokens=(not templated)
        if config.add_special_tokens is None
        else config.add_special_tokens,
    ).to(device)

    if config.replace_prompt_with_pad and system_token_count > 0:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids = input_ids.clone()
        input_ids[0, :system_token_count] = pad_id

    return input_ids


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def capture_example(bridge, structure, question, config, device, *, attention=False):
    """Tokenize a question and capture residuals, optionally with attention.

    Return input IDs, the question-region boundary, and a CPU activation cache.
    No metrics are computed; the notebook reduces and releases the cache.
    """
    start = count_system_tokens(
        bridge.tokenizer, config.system_prompt, config.add_special_tokens
    )
    input_ids = build_input_ids(bridge.tokenizer, question, config, start, device)
    with torch.no_grad():
        _, cache = bridge.run_with_cache(
            input_ids,
            names_filter=metrics_names_filter if attention else structure.hook_names(),
            accumulate=True,
            device="cpu",
        )
    return input_ids, start, cache


def run_example(
    bridge: Any,
    structure: ModelLayerStructure,
    forward_kwargs: dict,
    question: str,
    config: BehaviorRunConfig,
    system_token_count: int,
    device: str,
    store_raw_values: bool = False,
) -> ModelRunData:
    """Run one GSM8K question through the model and return its :class:`ModelRunData`.

    Tokenisation uses :func:`build_input_ids`, so the prompt construction
    is identical to the batch script.

    Parameters
    ----------
    bridge:
        The loaded TransformerLens bridge (``bridge.tokenizer`` must be set).
    structure:
        Block structure returned by :func:`load_model`.
    forward_kwargs:
        Extra kwargs for ``run_with_cache`` returned by :func:`load_model`.
    question:
        Raw question string (no system prompt — that is read from *config*).
    config:
        The run configuration (used for tokenisation options and prompt text).
    system_token_count:
        Precomputed token count for the system prompt (from :func:`load_model`).
    device:
        Torch device string.
    store_raw_values:
        Whether to store raw activation values in the returned ``ModelRunData``.
    """
    tokenizer = bridge.tokenizer
    assert tokenizer is not None

    input_ids = build_input_ids(tokenizer, question, config, system_token_count, device)

    with torch.no_grad():
        _, act_cache = bridge.run_with_cache(
            input_ids,
            names_filter=metrics_names_filter
            if store_raw_values
            else lambda name: (
                name.endswith("hook_out") and name.split(".")[-2].isdigit()
            ),
            accumulate=True,
            device="cpu",
            **forward_kwargs,
        )

    run_data = compute_model_run_data(
        cache=act_cache,
        input_ids=input_ids,
        structure=structure,
        compute_pairwise=False,
        compute_differences=True,
        compute_residual_entropy=store_raw_values,
        store_raw_values=store_raw_values,
        verbose=False,
    )[0]
    del act_cache
    return run_data
