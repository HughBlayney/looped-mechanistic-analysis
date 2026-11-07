"""Read intermediate recurrent states through the model's trained coda."""

import torch

from looped_internals.models.adapters.parcae import ParcaeArchitectureAdapter
from looped_internals.models.adapters.raven import RavenArchitectureAdapter

from .logit_lens import logit_lens


@torch.no_grad()
def coda_hidden_states(
    bridge,
    hidden_states: torch.Tensor,
    *,
    input_ids: torch.Tensor | None = None,
    from_core: bool = True,
) -> torch.Tensor:
    """Run the native coda suffix on raw block outputs, before the final norm.

    Supports Huginn, the Raven-based retrofits, and Parcae. Inputs are unpadded
    ``(batch, sequence, width)`` states with positions starting at zero. Causal
    attention runs over the entire sequence without a KV cache; token chunks
    would change the coda computation. Huginn normalizes the raw recurrent state
    before the coda; the retrofits pass it directly to the coda.

    Parcae requires matching input IDs for its coda value embeddings. Core
    states pass through its learned output projection C; prelude states already
    use the coda's residual space and enter directly (``from_core=False``).

    This invokes the bridge's coda hooks. Collectors running this inside an
    observation hook must ignore these auxiliary applications.
    """
    if not isinstance(
        bridge.adapter, (RavenArchitectureAdapter, ParcaeArchitectureAdapter)
    ):
        raise ValueError("Coda lens requires a Huginn, retrofitted, or Parcae coda")
    if not len(bridge.coda_blocks):
        raise ValueError("Coda lens requires at least one coda layer")

    if isinstance(bridge.adapter, ParcaeArchitectureAdapter):
        if input_ids is None or input_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                "Parcae coda requires input_ids matching the hidden states"
            )
        model = bridge.original_model.parcae
        freqs = model.freqs_cis[:, : hidden_states.shape[1]]
        h = bridge.core_projection(hidden_states) if from_core else hidden_states
        offset = (
            model.config.n_layers_in_prelude + model.config.n_layers_in_recurrent_block
        )
        for index, block in enumerate(bridge.coda_blocks, start=offset):
            key = str(index)
            ve = (
                model.value_embeds[key](input_ids)
                if key in model.value_embeds
                else None
            )
            h = block(h, freqs, None, ve=ve)
        return h

    model = bridge.original_model
    length = hidden_states.shape[1]
    # The retrofits use HF rotary embeddings; Huginn stores complex RoPE phases.
    if hasattr(model, "rotary_emb"):
        positions = torch.arange(length, device=hidden_states.device).unsqueeze(0)
        freqs = model.rotary_emb(hidden_states, positions)
        h = hidden_states
    else:
        freqs = model.freqs_cis[:, :length]
        h = bridge.ln_final(hidden_states)

    for index, block in enumerate(bridge.coda_blocks, start=1):
        step = torch.tensor(-index, dtype=torch.long, device="cpu")
        h = block(h, freqs, step, None, None)
    return h


@torch.no_grad()
def coda_lens(
    bridge,
    hidden_states: torch.Tensor,
    *,
    input_ids: torch.Tensor | None = None,
    from_core: bool = True,
) -> torch.Tensor:
    """Return float32 vocabulary logits after the coda, final norm, and head.

    For memory-bounded readouts, call ``coda_hidden_states`` on the full sequence,
    then apply ``logit_lens`` to token chunks of its output.
    """
    return logit_lens(
        bridge,
        coda_hidden_states(
            bridge, hidden_states, input_ids=input_ids, from_core=from_core
        ),
    )
