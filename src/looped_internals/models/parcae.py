"""Load Parcae checkpoints as ``(bridge, structure)`` pairs.

The released Parcae models (``SandyResearch/parcae-140m`` and larger) and the
recurrence-ablated controls trained for this repository share one checkpoint
layout: a ``config.json`` describing a ``parcae_lm.ParcaeConfig`` and a
``pytorch_model.bin`` state dict. Controls are exported from their Lightning
training checkpoints into that layout by ``scripts/export_hf.py`` in the
parcae training repository, so this module only depends on the ``parcae-lm``
package and never on training code.

Tokenization: every Parcae checkpoint uses ``SandyResearch/parcae-tokenizer``,
a 32768-token BPE vocabulary whose ``tokenizer.json`` declares ``<|bos|>``,
``<|eos|>``, and ``<|pad|>`` but no post-processing. Pretraining tokenized each
FineWeb-Edu document with plain ``encode`` and packed documents back to back,
so no BOS or EOS token ever appeared in training inputs. The loader keeps that
native behaviour: nothing is prepended, and position 0 is the first text token.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformers import PreTrainedTokenizerFast

from looped_internals.models.adapters.parcae import ParcaeArchitectureAdapter
from looped_internals.models.tokenization import tokenizer_metadata
from looped_internals.structure import LayerGroup, ModelLayerStructure

TOKENIZER_REPO = "SandyResearch/parcae-tokenizer"


@dataclass(frozen=True)
class ParcaeCheckpoint:
    """Hub repository ID or local directory with ``config.json`` and weights.

    ``subfolder`` selects one model within the source, so a sweep of related
    checkpoints can live in a single repository, one directory per arm. The
    tokenizer is always read from its own repository, never the subfolder.
    """

    weights: str
    tokenizer: str = TOKENIZER_REPO
    subfolder: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None


def _resolve(
    source: str,
    filename: str,
    subfolder: str | None = None,
    revision: str | None = None,
) -> Path:
    """Return a local path for ``filename`` in a directory or Hub repository."""
    directory = Path(source).expanduser()
    if directory.is_dir():
        path = directory / subfolder / filename if subfolder else directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"{source} does not contain {path.name}")
        return path
    return Path(
        hf_hub_download(source, filename, subfolder=subfolder, revision=revision)
    )


class ParcaeLogitsWrapper(nn.Module):
    """Run Parcae for an explicit recurrence count and return logits only.

    ``parcae_lm.Parcae.forward`` returns a dictionary and, unless a step pair
    is supplied, samples its recurrence depth per sequence. Every call here
    passes ``num_steps_pair`` so evaluation is deterministic, and ``num_steps``
    can override the loaded default per call, as Raven's native forward does.
    The original model stays available as ``.parcae`` for component mapping.
    """

    def __init__(self, model: nn.Module, num_steps: int) -> None:
        super().__init__()
        self.parcae = model
        self.num_steps = num_steps

    @property
    def config(self) -> Any:
        return self.parcae.config

    def forward(
        self, input_ids: torch.Tensor, num_steps: int | None = None, **kwargs: Any
    ) -> torch.Tensor:
        # Attention patterns come from the adapter's SDPA capture; the bridge's
        # Hugging Face style arguments have no meaning for Parcae.
        kwargs.pop("return_type", None)
        kwargs.pop("output_attentions", None)
        if kwargs:
            raise TypeError(f"Unsupported Parcae forward arguments: {sorted(kwargs)}")
        steps = self.num_steps if num_steps is None else num_steps
        pair = torch.tensor([steps, 0], device=input_ids.device)
        output = self.parcae(input_ids, return_logits=True, num_steps_pair=pair)
        return output["logits"]


def load_parcae_tokenizer(
    source: str = TOKENIZER_REPO, *, revision: str | None = None
) -> PreTrainedTokenizerFast:
    """Build the checkpoint tokenizer without adding BOS, matching pretraining."""
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(_resolve(source, "tokenizer.json", revision=revision)),
        eos_token="<|eos|>",
        pad_token="<|pad|>",
    )
    if tokenizer_metadata(tokenizer)["prepends_bos"]:
        raise ValueError(f"{source} prepends BOS; Parcae was trained without it")
    return tokenizer


def load_parcae(
    checkpoint: ParcaeCheckpoint,
    *,
    device: str,
    dtype: torch.dtype,
    num_steps: int | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> tuple[TransformerBridge, ModelLayerStructure]:
    """Load a Parcae checkpoint into a hooked ``TransformerBridge``.

    ``num_steps`` fixes the inference recurrence count; ``None`` keeps the
    checkpoint's ``mean_recurrence``. Overrides are applied to the saved
    configuration before construction, so architectural changes would fail the
    strict state-dict load rather than silently drop weights.
    """
    from parcae_lm.attention_backends import flash_attention
    from parcae_lm.models.parcae.config import ParcaeConfig

    config_dict = json.loads(
        _resolve(
            checkpoint.weights, "config.json", checkpoint.subfolder, checkpoint.revision
        ).read_text()
    )
    class_name = config_dict.pop("_class_name", "ParcaeConfig")
    if class_name != "ParcaeConfig":
        raise ValueError(f"{checkpoint.weights} is a {class_name}, not a Parcae model")
    config_dict.update(config_overrides or {})
    config = ParcaeConfig(**config_dict)

    # Parcae selects FlashAttention 3 on Hopper GPUs, which bypasses the SDPA
    # interception that exposes attention patterns. SDPA is exact and cheap at
    # analysis batch sizes, so use it on every device.
    flash_attention._override_impl = "sdpa"  # ty: ignore[invalid-assignment]
    model = config.construct_model()
    state = torch.load(
        _resolve(
            checkpoint.weights,
            "pytorch_model.bin",
            checkpoint.subfolder,
            checkpoint.revision,
        ),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    # Cast parameters only: the persistent RoPE table stays float32, as in
    # mixed-precision training, and every norm already computes in float32.
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype)
    model.to(device)
    model.eval()

    steps = config.mean_recurrence if num_steps is None else num_steps
    wrapper = ParcaeLogitsWrapper(model, steps)
    tl_config = TransformerBridgeConfig(
        d_model=config.n_embd,
        d_head=config.head_size,
        n_layers=config.n_layers_in_prelude
        + config.n_layers_in_recurrent_block
        + config.n_layers_in_coda,
        n_ctx=config.block_size,
        n_heads=config.num_attention_heads,
        d_vocab=config.padded_vocab_size or config.vocab_size,
        architecture="Parcae",
    )
    tl_config.n_key_value_heads = config.num_key_value_heads
    adapter = ParcaeArchitectureAdapter(tl_config, prelude_norm=config.prelude_norm)
    tokenizer = load_parcae_tokenizer(
        checkpoint.tokenizer, revision=checkpoint.tokenizer_revision
    )
    bridge = TransformerBridge(wrapper, adapter, tokenizer=tokenizer)
    tokenization = tokenizer_metadata(tokenizer)
    bridge.cfg.default_prepend_bos = tokenization["prepends_bos"]
    bridge.cfg.tokenizer_prepends_bos = tokenization["prepends_bos"]
    bridge.cfg.tokenizer_appends_eos = tokenization["appends_eos"]
    structure = ModelLayerStructure(
        groups=[
            LayerGroup("prelude_blocks", config.n_layers_in_prelude, False),
            LayerGroup("core_blocks", config.n_layers_in_recurrent_block, True),
            LayerGroup("coda_blocks", config.n_layers_in_coda, False),
        ]
    )
    return bridge, structure
