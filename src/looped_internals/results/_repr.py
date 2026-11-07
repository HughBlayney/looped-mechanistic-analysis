from __future__ import annotations

import dataclasses
from typing import Any

from torch import Tensor


def _short_value_repr(value: Any) -> str:
    """Return a bounded repr that never formats full tensor contents."""
    if isinstance(value, Tensor):
        shape = tuple(value.shape)
        device = "" if value.device.type == "cpu" else f", device={value.device}"
        return f"Tensor(shape={shape}, dtype={value.dtype}{device})"

    if isinstance(value, list):
        if not value:
            return "[]"
        if all(isinstance(item, Tensor) for item in value):
            items = ", ".join(_short_value_repr(item) for item in value[:3])
            suffix = ", ..." if len(value) > 3 else ""
            return f"list(len={len(value)}, items=[{items}{suffix}])"
        return f"list(len={len(value)})"

    if isinstance(value, dict):
        if not value:
            return "{}"
        keys = list(value.keys())
        key_preview = ", ".join(repr(key) for key in keys[:5])
        suffix = ", ..." if len(keys) > 5 else ""
        if all(isinstance(item, Tensor) for item in value.values()):
            return (
                f"dict(len={len(value)}, keys=[{key_preview}{suffix}], values=Tensor)"
            )
        return f"dict(len={len(value)}, keys=[{key_preview}{suffix}])"

    return repr(value)


def _fast_dataclass_repr(instance: Any) -> str:
    fields = (
        f"{field.name}={_short_value_repr(getattr(instance, field.name))}"
        for field in dataclasses.fields(instance)
    )
    return f"{type(instance).__name__}({', '.join(fields)})"


class _FastReprMixin:
    """Mixin for tensor-heavy dataclasses whose default repr is too expensive."""

    def __repr__(self) -> str:
        return _fast_dataclass_repr(self)
