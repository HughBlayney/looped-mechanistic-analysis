"""Select the correct Raven remote classes and adapt their Transformers 4 APIs."""

from copy import deepcopy
from functools import wraps
from typing import Any, cast

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedConfig,
    PreTrainedModel,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def prepare_raven_source(model_name: str, *, revision: str | None = None) -> None:
    """Prepare a trusted Raven/retrofitted source before AutoConfig constructs it.

    Every variant calls AutoConfig.register("huginn_raven", ...) when imported.
    Transformers 5 prefers that registration even with trust_remote_code=True.
    Resolve and register this checkpoint's own classes on each load so loading
    Huginn first cannot accidentally select its architecture for a retrofit.
    """
    config_dict, _ = PreTrainedConfig.get_config_dict(model_name, revision=revision)
    if config_dict.get("model_type") != "huginn_raven":
        return
    revision = config_dict.get("_commit_hash")
    config_class = cast(
        Any,
        get_class_from_dynamic_module(
            config_dict["auto_map"]["AutoConfig"], model_name, revision=revision
        ),
    )
    original = config_class.__init__
    if not getattr(original, "_looped_config", False):

        @wraps(original)
        def init(self, *args, **kwargs):
            # Base-config RoPE validation now runs before it assigns unknown
            # kwargs. Make the checkpoint's actual context length visible then.
            self.max_position_embeddings = kwargs.get(
                "max_position_embeddings", kwargs.get("block_size", 4096)
            )
            original(self, *args, **kwargs)
            # Current Llama/OLMo rotary modules read this normalized dictionary.
            self.rope_parameters = deepcopy(
                kwargs.get("rope_parameters") or kwargs.get("rope_scaling") or {}
            )
            self.rope_parameters.setdefault(
                "rope_theta", kwargs.get("rope_theta", self.rope_base)
            )
            self.standardize_rope_params()

        cast(Any, init)._looped_config = True
        config_class.__init__ = init
    # The remote modeling modules disable math SDPA at import time. Loading a
    # checkpoint must not change the caller's kernel settings for other models.
    math_sdp_enabled = torch.backends.cuda.math_sdp_enabled()
    try:
        model_class = cast(
            Any,
            get_class_from_dynamic_module(
                config_dict["auto_map"]["AutoModelForCausalLM"],
                model_name,
                revision=revision,
            ),
        )
    finally:
        torch.backends.cuda.enable_math_sdp(math_sdp_enabled)
    model_class._tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}
    original_init_weights = model_class._init_weights
    if not getattr(original_init_weights, "_looped_rotary_init", False):

        @wraps(original_init_weights)
        def init_weights(self, module):
            # Transformers 5 constructs pretrained models on meta, then restores
            # non-persistent buffers through _init_weights. The retrofit's no-op
            # implementation otherwise leaves its Llama/OLMo RoPE uninitialized.
            if "RotaryEmbedding" in type(module).__name__:
                PreTrainedModel._init_weights(self, module)
            else:
                original_init_weights(self, module)

        cast(Any, init_weights)._looped_rotary_init = True
        model_class._init_weights = init_weights
    original_model_init = model_class.__init__
    if not getattr(original_model_init, "_looped_post_init", False):

        @wraps(original_model_init)
        def model_init(self, *args, **kwargs):
            original_model_init(self, *args, **kwargs)
            # Retrofitted implementations omit post_init; Transformers 5 needs
            # its weight-tying and load metadata even for pretrained models.
            if not hasattr(self, "all_tied_weights_keys"):
                self.post_init()

        cast(Any, model_init)._looped_post_init = True
        model_class.__init__ = model_init
    AutoConfig.register("huginn_raven", config_class, exist_ok=True)
    AutoModelForCausalLM.register(config_class, model_class, exist_ok=True)
