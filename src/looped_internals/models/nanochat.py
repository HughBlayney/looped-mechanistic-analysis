"""nanochat TrellisGPT2 bridge support.

The nanochat recurrent GPT used for the training-stage figures is not a
HuggingFace model, but its module structure is close enough to wrap with the
same TransformerLens ``TransformerBridge`` machinery used elsewhere in this
repo. This module provides an architecture adapter plus a checkpoint loader
that returns the standard ``(bridge, structure)`` pair consumed by
``compute_model_run_data``.
"""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge

from looped_internals.structure import LayerGroup, ModelLayerStructure

from .adapters.nanochat import NanochatArchitectureAdapter, NanochatLogitsWrapper

DEFAULT_NANOCHAT_REPO = os.environ.get("NANOCHAT_REPO", "../nanochat")
DEFAULT_NANOCHAT_BASE_DIR = os.environ.get(
    "NANOCHAT_BASE_DIR", str(Path.home() / ".cache" / "nanochat")
)


class NanochatTokenizerAdapter:
    """Small HuggingFace-style adapter around nanochat's tokenizer."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.bos_token_id = self._get_bos_token_id()
        # nanochat models used here do not need real padding semantics because
        # B3 forwards one example at a time for scoring. A concrete integer keeps
        # HF-style batching helpers happy when they inspect the tokenizer.
        self.pad_token_id = 0

    def _get_bos_token_id(self) -> int | None:
        if hasattr(self.tokenizer, "bos_token_id"):
            return int(self.tokenizer.bos_token_id)
        if hasattr(self.tokenizer, "get_bos_token_id"):
            return int(self.tokenizer.get_bos_token_id())
        return None

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **_kwargs: Any,
    ) -> list[int]:
        if add_special_tokens and self.bos_token_id is not None:
            try:
                return list(self.tokenizer.encode(text, prepend=self.bos_token_id))
            except TypeError:
                tokens = list(self.tokenizer.encode(text))
                return [self.bos_token_id, *tokens]
        try:
            return list(self.tokenizer.encode(text))
        except TypeError:
            return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int] | torch.Tensor, **kwargs: Any) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        try:
            return self.tokenizer.decode(token_ids, **kwargs)
        except TypeError:
            return self.tokenizer.decode(token_ids)

    def __call__(self, text: str, **kwargs: Any) -> list[int]:
        return self.encode(text, **kwargs)


def ensure_nanochat_importable(
    nanochat_repo: str | Path = DEFAULT_NANOCHAT_REPO,
) -> Path:
    """Add the external nanochat repo to ``sys.path`` if needed."""

    repo_path = Path(nanochat_repo).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"nanochat repo not found: {repo_path}")
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_path


def make_nanochat_structure(config: Any) -> ModelLayerStructure:
    """Return the current repo's block-structure descriptor for nanochat GPT."""

    return ModelLayerStructure(
        groups=[
            LayerGroup("prelude_blocks", config.n_prelude, is_recurrent=False),
            LayerGroup("core_blocks", config.n_recur_block, is_recurrent=True),
            LayerGroup("coda_blocks", config.n_coda, is_recurrent=False),
        ]
    )


def make_nanochat_bridge(
    model: nn.Module,
    tokenizer: Any = None,
) -> tuple[TransformerBridge, ModelLayerStructure]:
    """Wrap an already-created nanochat model in ``TransformerBridge``."""

    config = cast(Any, model).config
    tl_cfg = TransformerBridgeConfig(
        d_model=config.n_embd,
        d_head=config.n_embd // config.n_head,
        n_layers=config.n_prelude + config.n_recur_block + config.n_coda,
        n_ctx=config.sequence_len,
        n_heads=config.n_head,
        d_vocab=config.vocab_size,
        architecture="NanochatGPT",
    )
    tl_cfg.n_key_value_heads = config.n_kv_head

    wrapper = NanochatLogitsWrapper(model)
    adapter = NanochatArchitectureAdapter(tl_cfg)
    if tokenizer is not None and not isinstance(tokenizer, NanochatTokenizerAdapter):
        tokenizer = NanochatTokenizerAdapter(tokenizer)
    bridge = TransformerBridge(wrapper, adapter, tokenizer=tokenizer)
    bridge.eval()
    return bridge, make_nanochat_structure(config)


def load_nanochat_trellis(
    model_tag: str,
    step: int = 7125,
    source: str = "base",
    num_recur: int = 4,
    nanochat_repo: str | Path = DEFAULT_NANOCHAT_REPO,
    nanochat_base_dir: str | Path = DEFAULT_NANOCHAT_BASE_DIR,
    device: str | torch.device = "cuda",
) -> tuple[TransformerBridge, ModelLayerStructure, dict[str, int]]:
    """Load a nanochat checkpoint and return ``(bridge, structure, forward_kwargs)``."""

    ensure_nanochat_importable(nanochat_repo)
    os.environ.setdefault(
        "NANOCHAT_BASE_DIR", str(Path(nanochat_base_dir).expanduser())
    )

    load_model = import_module("nanochat.checkpoint_manager").load_model

    torch_device = torch.device(device)
    model, tokenizer, _metadata = load_model(
        source,
        torch_device,
        phase="eval",
        model_tag=model_tag,
        step=step,
    )
    model.eval()
    bridge, structure = make_nanochat_bridge(model, tokenizer)
    forward_kwargs = {} if model_tag == "control" else {"num_recur": num_recur}
    return bridge, structure, forward_kwargs
