"""Small, idempotent TransformerLens integration entry point."""


def initialize() -> None:
    from .capture.cache import patch_activation_cache, patch_bridge
    from .models.adapters import register_adapters

    patch_activation_cache()
    patch_bridge()
    register_adapters()
