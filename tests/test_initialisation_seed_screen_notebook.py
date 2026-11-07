"""Regression test for the lightweight initialisation seed screen."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch

NOTEBOOK = (
    Path(__file__).parents[1] / "experiments" / "initialisation_seed_screen.ipynb"
)


def test_seed_screen_uses_fixed_seeds_and_minimal_measurement(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {}

    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            assert namespace["seeds"] == list(range(42, 52))
            namespace.update(
                seeds=[3, 5],
                recurrences=4,
                layers=2,
                width=8,
                num_heads=2,
                sequence_length=1,
                vocab_size=16,
                device="cpu",
            )

    config = namespace["model_config"]
    assert config.input_injection == "projection"
    assert config.norm_pre_attn and config.norm_pre_mlp
    assert not any(
        (
            config.norm_post_attn,
            config.norm_post_mlp,
            config.norm_post_attn_residual,
            config.norm_post_mlp_residual,
            config.norm_post_cycle,
        )
    )
    assert set(namespace["curves"]) == {3, 5}
    for curve in namespace["curves"].values():
        assert curve.shape == (4,)
        assert curve[-1].item() == pytest.approx(1, abs=1e-6)

    repeated = namespace["final_state_cosine_curve"](3)
    torch.testing.assert_close(repeated, namespace["curves"][3])
    assert [ax.get_title() for ax in namespace["fig"].axes] == ["Seed 3", "Seed 5"]
