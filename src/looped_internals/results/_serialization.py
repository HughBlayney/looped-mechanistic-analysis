"""Dataclass serialization that preserves tensor storage instead of deepcopying it."""

from dataclasses import fields, is_dataclass


def state_dict(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: state_dict(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {key: state_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(state_dict(item) for item in value)
    return value
