"""Run the smoke-test notebook end to end with tiny local models."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch

from experiments import support
from experiments.paper_models import PAPER_MODEL_GROUPS
from looped_internals.models import MinimalRecurrentConfig, ModelSpec, load_model

NOTEBOOK = Path(__file__).parents[1] / "experiments" / "smoke_test.ipynb"
CODA_MODELS = {"raven", "retrofitted_llama", "parcae_770m", "parcae_770m_rec1"}


def execute(namespace=None, *, stop_after=None, **parameters):
    """Execute code cells, applying overrides after the tagged parameter cell."""
    namespace = {} if namespace is None else namespace
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(parameters)
        if stop_after is not None and cell["id"] == stop_after:
            break
    return namespace


def test_defaults_follow_stages_of_inference_and_papermill(tmp_path, caplog):
    namespace = execute(stop_after="configuration")
    assert namespace["model_recurrences"] == [
        (name, 4 if name == "ouro_1_4b" else 32) for name in PAPER_MODEL_GROUPS["main"]
    ]
    assert namespace["control_pairs"] == [("llama_1b", 1), ("parcae_770m_rec1", 1)]
    assert {c.experiment for c in namespace["configs"].values()} == {"smoke_test"}
    assert namespace["configs"][("parcae_770m_rec1", 1)].num_steps == 1

    papermill = pytest.importorskip("papermill")
    papermill.execute_notebook(
        NOTEBOOK,
        tmp_path / "prepared.ipynb",
        parameters={"device": "cuda", "prompt": "Paris is the capital of"},
        prepare_only=True,
    )
    assert "unknown parameter" not in caplog.text


class WordTokenizer:
    """Whitespace tokens with fixed IDs, for models without a tokenizer."""

    def encode(self, text, add_special_tokens=True, return_tensors=None):
        ids = [3 + sum(map(ord, word)) % 60 for word in text.split()]
        return torch.tensor([ids]) if return_tensors else ids

    def decode(self, ids):
        return " ".join(f"w{index}" for index in ids)


@pytest.fixture
def parcae_checkpoint(tmp_path_factory):
    """A tiny randomly weighted Parcae export with prelude, core and coda."""
    pytest.importorskip("parcae_lm")
    from test_match_parcae import checkpoint

    return checkpoint.__wrapped__(tmp_path_factory)


def test_notebook_runs_all_cells_with_coda_and_coda_free_models(
    tmp_path, monkeypatch, parcae_checkpoint
):
    torch.set_num_threads(1)
    loads, figures = [], []

    def load_experiment_model(name, config):
        loads.append((name, config.num_steps))
        if name in CODA_MODELS:
            # Prelude, core and coda: exercises the coda-lens readout.
            return load_model(
                ModelSpec(parcae_checkpoint), device="cpu", num_steps=config.num_steps
            )
        bridge, structure = load_model(
            ModelSpec(
                MinimalRecurrentConfig(
                    d_model=16, n_heads=4, n_layers=2, d_vocab=64, input_injection=None
                )
            ),
            device="cpu",
            num_steps=config.num_steps,
        )
        bridge.tokenizer = WordTokenizer()
        return bridge, structure

    def save(fig, directory, name, **kwargs):
        figures.append((name, fig.get_size_inches()))
        original_save(fig, directory, name, **kwargs)

    original_save = support.save_figure
    monkeypatch.setattr(support, "load_experiment_model", load_experiment_model)
    monkeypatch.setattr(support, "record_tokenization", lambda *args: None)
    monkeypatch.setattr(support, "save_figure", save)
    monkeypatch.setattr(plt, "show", lambda: None)

    namespace = execute(
        prompt="w3 w4 w5 w6",
        expected=" w7",
        recurrences=3,
        ouro_recurrences=2,
        new_tokens=2,
        device="cpu",
        output_dir=str(tmp_path / "results"),
    )
    pairs = [
        ("ouro_1_4b", 2),
        ("raven", 3),
        ("retrofitted_llama", 3),
        ("parcae_770m", 3),
        ("llama_1b", 1),
        ("parcae_770m_rec1", 1),
    ]
    assert loads == pairs
    run_dir = namespace["run_dir"]
    assert run_dir.parent.name == "smoke_test"
    results = torch.load(run_dir / "overview.pt", weights_only=False)
    assert list(results) == pairs
    for (name, steps), result in results.items():
        assert len(result["readout"]) == steps
        assert result["hooked_vs_plain"] == 0
        # The last readout is the model's own prediction, with or without a coda.
        assert result["readout_vs_output"] < 1e-5
        assert result["readout"][-1]["p_expected"] == pytest.approx(
            result["prediction"]["p_expected"], abs=1e-5
        )
        assert len(result["prediction"]["top"]) == 5
        assert len(result["continuation"].split()) == 2
        n_virtual = len(result["layers"])
        assert all(len(curve) == n_virtual for curve in result["curves"].values())
    coda_result = results[("parcae_770m", 3)]
    assert coda_result["input_tokens"] == ["w3", "w4", "w5", "w6"]
    assert coda_result["layers"] == [0, 1, 2, 1, 2, 1, 2, 3]
    assert results[("ouro_1_4b", 2)]["layers"] == [0, 1, 0, 1]

    assert [name for name, _ in figures] == ["smoke_test"]
    assert tuple(figures[0][1]) == pytest.approx((6.75, 5.2))
    assert (run_dir / "smoke_test.pdf").exists()
    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["experiment"] == "smoke_test"
    assert metadata["prompt"] == "w3 w4 w5 w6" and metadata["expected"] == " w7"
    assert not {"dataset", "task", "limit"} & metadata.keys()
