"""Architecture-specific TransformerLens bridges and registration."""


def register_adapters() -> None:
    """Register architecture adapters in the factory."""
    from transformer_lens.factories.architecture_adapter_factory import (
        SUPPORTED_ARCHITECTURES,
        ArchitectureAdapterFactory,
    )

    from looped_internals.models.adapters.minimal import MinimalRecurrentAdapter
    from looped_internals.models.adapters.nanochat import NanochatArchitectureAdapter
    from looped_internals.models.adapters.ouro import OuroArchitectureAdapter
    from looped_internals.models.adapters.raven import RavenArchitectureAdapter

    SUPPORTED_ARCHITECTURES["RavenForCausalLM"] = RavenArchitectureAdapter
    SUPPORTED_ARCHITECTURES["OuroForCausalLM"] = OuroArchitectureAdapter
    SUPPORTED_ARCHITECTURES["MinimalRecurrentLM"] = MinimalRecurrentAdapter
    SUPPORTED_ARCHITECTURES["NanochatGPT"] = NanochatArchitectureAdapter
    # boot_transformers checks SUPPORTED_ARCHITECTURES; adapter selection has a
    # separate registry. Use its public API without replacing other registrations.
    for name in (
        "RavenForCausalLM",
        "OuroForCausalLM",
        "MinimalRecurrentLM",
        "NanochatGPT",
    ):
        ArchitectureAdapterFactory.register_adapter(name, SUPPORTED_ARCHITECTURES[name])
