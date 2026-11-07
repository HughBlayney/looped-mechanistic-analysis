"""Parcae bridge: output matching, hook layout, tokenization, and registry."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

import looped_internals  # noqa: F401 - registers local TransformerBridge patches
from looped_internals.capture import compute_model_run_data, metrics_names_filter
from looped_internals.models import ModelSpec, ParcaeCheckpoint, load_model
from looped_internals.models.parcae import load_parcae_tokenizer
from looped_internals.structure import LayerGroup

parcae_lm = pytest.importorskip("parcae_lm")
from parcae_lm.models.parcae.config import ParcaeConfig

RECURRENCES = 3
SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>"]


def _tiny_config() -> ParcaeConfig:
    """One prelude, two core, and one coda layer at width 32 (released settings)."""
    return ParcaeConfig(
        name="tiny-parcae",
        block_size=32,
        n_embd=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=64,
        padding_multiple=64,
        norm_class_name="RMSNorm",
        nonlin_name="ReLU2",
        init_strategy="scaled-zero",
        init_orthogonal=True,
        qk_norm=True,
        injection_type="diagonal",
        n_layers_in_recurrent_block=2,
        n_layers_in_prelude=1,
        n_layers_in_coda=1,
        recurrent_embedding_dimension=32,
        recurrent_intermediation_embedding_dimension=64,
        prelude_norm=True,
        sampling_scheme="fixed",
        mean_recurrence=RECURRENCES,
        mean_backprop_depth=1,
        recurrent_iteration_method="per-sequence",
        tie_embeddings=True,
    )


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> ParcaeCheckpoint:
    """Export a randomly weighted tiny model in the released two-file layout."""
    directory = tmp_path_factory.mktemp("parcae")
    config = _tiny_config()
    torch.manual_seed(0)
    model = config.construct_model()
    # Scaled-zero initialization zeroes output projections; use dense random
    # weights so every path contributes to the compared logits.
    for parameter in model.parameters():
        parameter.data.normal_(0.0, 0.1)
    torch.save(model.state_dict(), directory / "pytorch_model.bin")
    with open(directory / "config.json", "w") as f:
        json.dump({**asdict(config), "_class_name": "ParcaeConfig"}, f)

    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    vocab.update({f"w{index}": index for index in range(len(vocab), 64)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<|pad|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.save(str(directory / "tokenizer.json"))
    return ParcaeCheckpoint(str(directory), tokenizer=str(directory))


@pytest.fixture(scope="module")
def loaded(checkpoint):
    return load_model(
        ModelSpec(checkpoint, default_steps=RECURRENCES),
        device="cpu",
        dtype=torch.float32,
    )


@pytest.fixture
def sample_input():
    torch.manual_seed(42)
    return torch.randint(3, 64, (2, 12))


def _raw_logits(model, input_ids, steps):
    torch.manual_seed(7)  # The recurrent state is initialized randomly.
    pair = torch.tensor([steps, 0])
    return model(input_ids, return_logits=True, num_steps_pair=pair)["logits"]


def test_bridge_matches_parcae_forward(checkpoint, loaded, sample_input):
    bridge, structure = loaded
    raw = _tiny_config().construct_model()
    raw.load_state_dict(torch.load(f"{checkpoint.weights}/pytorch_model.bin"))
    raw.eval()
    for steps in (RECURRENCES, 1, 5):
        with torch.no_grad():
            expected = _raw_logits(raw, sample_input, steps)
            torch.manual_seed(7)
            actual = bridge(sample_input, return_type="logits", num_steps=steps)
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-5)
    with torch.no_grad():
        torch.manual_seed(7)
        default = bridge(sample_input, return_type="logits")
    assert torch.allclose(default, _raw_logits(raw, sample_input, RECURRENCES))
    assert structure.groups == [
        LayerGroup("prelude_blocks", 1, False),
        LayerGroup("core_blocks", 2, True),
        LayerGroup("coda_blocks", 1, False),
    ]


def test_mlp_weights_use_neuron_by_residual_orientation(loaded):
    bridge, structure = loaded
    raw = bridge.original_model.parcae
    layers = [
        block
        for group in (
            raw.transformer.prelude,
            raw.transformer.core_block,
            raw.transformer.coda,
        )
        for block in group
    ]
    for (group, index), block in zip(structure.iter_layers(), layers, strict=True):
        mlp = getattr(bridge, group.hook_prefix)[index].mlp
        torch.testing.assert_close(mlp.W_out, block.mlp.proj.weight.T)
        torch.testing.assert_close(mlp.W_in, block.mlp.fc.weight.T)
        assert mlp.W_out.shape == (64, 32)


def test_coda_lens_matches_native_suffix_and_preserves_value_embeddings(
    loaded, sample_input
):
    from looped_internals.evaluation.coda_lens import coda_hidden_states, coda_lens

    torch.set_num_threads(1)
    bridge, structure = loaded
    tokens = sample_input[:, :6]
    with torch.no_grad():
        output, cache = bridge.run_with_cache(
            tokens, accumulate=True, names_filter=structure.hook_names()
        )
    state = cache[f"core_blocks.1.hook_out.iter_{RECURRENCES - 1}"]
    original = state.clone()
    readout = coda_lens(bridge, state, input_ids=tokens)
    torch.testing.assert_close(readout, output)
    torch.testing.assert_close(state, original)
    assert not readout.requires_grad and readout.dtype == torch.float32

    # Prelude states skip C; they already live in the coda's residual space.
    prelude = cache["prelude_blocks.0.hook_out"]
    model = bridge.original_model.parcae
    with torch.no_grad():
        expected = model.transformer.coda[0](
            prelude, model.freqs_cis[:, :6], None, ve=model.value_embeds["3"](tokens)
        )
    torch.testing.assert_close(
        coda_hidden_states(bridge, prelude, input_ids=tokens, from_core=False), expected
    )

    changed_tokens = tokens.clone()
    changed_tokens[:, -1] = (changed_tokens[:, -1] + 1) % 64
    changed_state = state.clone()
    changed_state[:, -1] += 100
    changed = coda_lens(bridge, changed_state, input_ids=changed_tokens)
    torch.testing.assert_close(changed[:, :-1], readout[:, :-1])
    # Reusing the last backbone input IDs would ignore this intervention.
    assert not torch.allclose(
        coda_lens(bridge, state, input_ids=changed_tokens)[:, -1], readout[:, -1]
    )
    with pytest.raises(ValueError, match="input_ids matching"):
        coda_lens(bridge, state)


def test_multiple_choice_coda_readout_uses_each_unpadded_option(loaded):
    from looped_internals.evaluation.datasets import MCQExample
    from looped_internals.evaluation.logit_lens import score_logitlens

    bridge, structure = loaded
    torch.set_num_threads(1)
    predictions = score_logitlens(
        bridge,
        bridge.tokenizer,
        MCQExample("w3 w4", ["w5", "w6 w7 w8"], 0),
        structure,
        RECURRENCES,
        torch.device("cpu"),
        compute_coda=True,
    )
    # The final core state plus the native suffix must choose the same option
    # as the observed final coda state, even with unequal option lengths.
    assert predictions[2][-2] == predictions[0][-1]
    assert torch.isnan(torch.tensor(predictions[2][-1]))


def test_strict_loading_and_parameter_dtype(checkpoint):
    bridge, _ = load_model(ModelSpec(checkpoint), device="cpu", dtype=torch.bfloat16)
    model = bridge.original_model.parcae
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert model.freqs_cis.dtype == torch.float32
    assert model.lm_head.weight is model.transformer.wte.weight
    assert not model.training
    assert bridge.original_model.num_steps == RECURRENCES


def test_recurrent_hooks_and_metrics(loaded, sample_input):
    bridge, structure = loaded
    with torch.no_grad():
        torch.manual_seed(7)
        _, cache = bridge.run_with_cache(
            sample_input[:1],
            accumulate=True,
            names_filter=metrics_names_filter,
        )
    assert "prelude_blocks.0.hook_out" in cache
    assert "coda_blocks.0.hook_out" in cache
    assert "prelude_blocks.0.hook_out.iter_1" not in cache
    for block in range(2):
        for suffix in ("hook_out", "attn.hook_pattern"):
            base = f"core_blocks.{block}.{suffix}"
            assert all(f"{base}.iter_{i}" in cache for i in range(RECURRENCES))
            assert f"{base}.iter_{RECURRENCES}" not in cache
    pattern = cache["core_blocks.0.attn.hook_pattern.iter_0"]
    assert pattern.shape == (1, 4, 12, 12)
    assert torch.allclose(pattern.sum(-1), torch.ones(1, 4, 12), atol=1e-5)
    assert torch.equal(pattern.triu(1), torch.zeros_like(pattern))

    data = compute_model_run_data(
        cache=cache,
        input_ids=sample_input[:1],
        structure=structure,
        compute_pairwise=False,
        compute_differences=False,
    )
    metrics = data[0].per_virtual_layer_metrics
    assert len(metrics) == 1 + 2 * RECURRENCES + 1
    assert all(
        m.sink_scores is not None and m.mixing_scores is not None for m in metrics
    )


@pytest.mark.parametrize("save_profiles", [False, True])
def test_convergence_reduction_uses_recurrent_layers(
    loaded, sample_input, save_profiles
):
    """Shared stages profiles must not add prelude/coda to convergence curves."""
    from test_experiments import _generic_summary, _notebook_science

    bridge, structure = loaded
    science = _notebook_science()
    names_filter = (
        metrics_names_filter
        if save_profiles
        else science["recurrent_metric_names"](structure)
    )
    with torch.no_grad():
        _, cache = bridge.run_with_cache(
            sample_input, accumulate=True, names_filter=names_filter
        )
    for prefix in ("prelude_blocks.0", "coda_blocks.0"):
        assert (f"{prefix}.hook_out.iter_0" in cache) == save_profiles
    expected = compute_model_run_data(
        cache, sample_input, structure, compute_pairwise=False, compute_differences=True
    )
    actual = science["reduce_convergence_cache"](cache, sample_input, structure)
    assert [record["summary"] for record in actual] == [
        _generic_summary(run) for run in expected
    ]
    assert all(record["summary"]["layers"] == [1, 2] for record in actual)


def test_first_token_mlp_ablation_hook_is_addressable(loaded, sample_input):
    bridge, _ = loaded
    seen = []

    def record(value, hook):
        seen.append(value.shape)
        return value

    with torch.no_grad():
        bridge.run_with_hooks(
            sample_input[:1], fwd_hooks=[("prelude_blocks.0.mlp.hook_out", record)]
        )
    assert seen == [(1, 12, 32)]


def test_tokenizer_adds_no_special_tokens(checkpoint):
    tokenizer = load_parcae_tokenizer(checkpoint.tokenizer)
    assert tokenizer.encode("w3 w4") == [3, 4]
    assert tokenizer.bos_token_id is None
    assert tokenizer.eos_token_id == 1
    assert tokenizer.pad_token_id == 2


def test_bridge_tokenization_flags(loaded):
    bridge, _ = loaded
    assert not bridge.cfg.default_prepend_bos
    assert not bridge.cfg.tokenizer_prepends_bos
    assert not bridge.cfg.tokenizer_appends_eos
    assert bridge.tokenizer.encode("w5") == [5]


def test_unknown_forward_arguments_are_rejected(loaded, sample_input):
    bridge, _ = loaded
    with pytest.raises(TypeError, match="attention_mask"):
        bridge.original_model(
            sample_input, attention_mask=torch.ones_like(sample_input)
        )
