"""Regression tests for preflight model-depth checks."""

from types import SimpleNamespace

import torch

from experiments import support
from looped_internals.structure import LayerGroup, ModelLayerStructure
from slurm_jobs.preflight import expected_virtual_layers


def test_expected_virtual_layers_respects_one_pass_control_depth():
    structure = ModelLayerStructure(
        groups=[
            LayerGroup("prelude", 2, False),
            LayerGroup("core", 2, True),
            LayerGroup("coda", 2, False),
        ]
    )

    assert expected_virtual_layers("parcae_140m_rec1", structure, 2) == 6
    assert expected_virtual_layers("parcae_140m", structure, 2) == 8


def test_stability_models_receive_the_preflight_recurrence_override(monkeypatch):
    calls = []

    def load_model(name, **kwargs):
        calls.append((name, kwargs))
        return object(), object()

    monkeypatch.setattr("looped_internals.models.load_model", load_model)
    config = SimpleNamespace(device="cpu", dtype="float32", num_steps=2)

    support.load_experiment_model("stab_fixed_0_25", config)

    assert calls == [
        (
            "stab_fixed_0_25",
            {"device": "cpu", "dtype": torch.float32, "num_steps": 2},
        )
    ]
