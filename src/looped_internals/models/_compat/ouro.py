"""Load Ouro's Transformers 4 remote code with Transformers 5.

Runtime changes are confined to the loaded Ouro module and its config. During
import, a scoped docstring decorator supplies Ouro's missing cache-position
documentation. The checkpoint's own model and UniversalTransformerCache remain
in use; cached source files are never edited.
"""

from __future__ import annotations

import sys
from functools import partial
from inspect import cleandoc
from textwrap import indent
from threading import RLock
from typing import Any, cast

import torch
from transformers import PreTrainedConfig, utils
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.utils import auto_docstring

_IMPORT_LOCK = RLock()
_CACHE_POSITION_DOC = """
cache_position (`torch.LongTensor` of shape `(sequence_length,)`, *optional*):
    Positions of the input tokens in the full sequence, used for position IDs
    and key/value cache updates. Defaults to consecutive positions after the
    tokens already cached.
"""


def _validate_rope(config):
    """Use the current validation API at Ouro's existing validation point."""
    config.standardize_rope_params()
    config.validate_rope()


def prepare_ouro_source(model_name: str, *, revision: str | None = None) -> None:
    """Adapt Ouro's config module before AutoConfig invokes its constructor."""
    config_dict, _ = PreTrainedConfig.get_config_dict(model_name, revision=revision)
    if config_dict.get("model_type") != "ouro":
        return
    config_class = get_class_from_dynamic_module(
        config_dict["auto_map"]["AutoConfig"],
        model_name,
        revision=config_dict.get("_commit_hash"),
    )
    module = cast(Any, sys.modules[config_class.__module__])
    module.rope_config_validation = _validate_rope


def _ouro_auto_docstring(obj=None, **kwargs):
    """Document Ouro's retained argument before Transformers checks signatures."""
    if obj is not None and obj.__qualname__ in {
        "OuroModel.forward",
        "OuroForCausalLM.forward",
    }:
        doc = cleandoc(obj.__doc__ or "")
        argument = indent(_CACHE_POSITION_DOC.strip(), "    ") + "\n"
        if "Args:\n" in doc:
            doc = doc.replace("Args:\n", "Args:\n" + argument, 1)
        else:
            doc = "Args:\n" + argument + doc
        obj.__doc__ = doc
    return auto_docstring(obj, **kwargs)


def _default_rope(config, device=None, seq_len=None):
    """The unscaled RoPE calculation used by Ouro under Transformers 4.56."""
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
    frequencies = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()
    return 1.0 / (config.rope_theta ** (frequencies / dim)), 1.0


def _legacy_mask(
    mask_function,
    *,
    input_embeds,
    cache_position=None,
    **kwargs,
):
    # Transformers 5 derives cache positions from the cache and position_ids.
    return mask_function(inputs_embeds=input_embeds, **kwargs)


def prepare_ouro_config(config: Any) -> type:
    """Return the checkpoint's model class after applying Ouro-local fixes.

    The caller must opt into executing the checkpoint's remote code. Use the
    config's resolved revision so its Python source and weights agree.
    """
    if config.model_type != "ouro":
        raise ValueError("Expected an Ouro configuration")
    if not hasattr(config, "pad_token_id"):
        config.pad_token_id = None
    # The remote module imports this decorator and runs it immediately. Supply
    # only the missing Ouro argument, preserving all other documentation checks.
    # Serialize our imports and restore the decorator even if loading fails.
    with _IMPORT_LOCK:
        utils_module = cast(Any, utils)
        original_docstring = utils_module.auto_docstring
        utils_module.auto_docstring = _ouro_auto_docstring
        try:
            model_class = cast(
                Any,
                get_class_from_dynamic_module(
                    config.auto_map["AutoModelForCausalLM"],
                    config.name_or_path,
                    revision=getattr(config, "_commit_hash", None),
                ),
            )
        finally:
            utils_module.auto_docstring = original_docstring
    module = cast(Any, sys.modules[model_class.__module__])
    # Copy the mapping before adding the removed default initializer. Other
    # models still see Transformers' original ROPE_INIT_FUNCTIONS.
    module.ROPE_INIT_FUNCTIONS = {
        **module.ROPE_INIT_FUNCTIONS,
        "default": _default_rope,
    }
    # Transformers 5 also reinitializes rotary buffers through this class API.
    module.OuroRotaryEmbedding.compute_default_rope_parameters = staticmethod(
        _default_rope
    )
    module.create_causal_mask = partial(_legacy_mask, create_causal_mask)
    module.create_sliding_window_causal_mask = partial(
        _legacy_mask, create_sliding_window_causal_mask
    )
    model_class.register_for_auto_class("AutoModelForCausalLM")
    # The old list only named aliases; Transformers 5 expects alias -> source.
    model_class._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    return model_class
