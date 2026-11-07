"""BOS interventions preserve native content, EOS, and explicit special tokens."""

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from looped_internals.evaluation.datasets import MCQExample
from looped_internals.evaluation.lm_eval import BridgeLM
from looped_internals.evaluation.logit_lens import _tokenize_mcq
from looped_internals.models.tokenization import (
    encode_with_bos_policy,
    tokenizer_metadata,
)
from looped_internals.structure import LayerGroup, ModelLayerStructure


def tokenizer(bos=False, eos=False):
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "Question": 3, ":": 4, "Hello": 5},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single=("[BOS] " if bos else "") + "$A" + (" [EOS]" if eos else ""),
        special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        name_or_path="local-checkpoint",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )


@pytest.mark.parametrize("native_bos", [False, True])
@pytest.mark.parametrize("native_eos", [False, True])
def test_explicit_policy_changes_only_automatic_bos(native_bos, native_eos):
    tok = tokenizer(native_bos, native_eos)
    info = tokenizer_metadata(tok)
    assert info["prepends_bos"] == native_bos
    assert info["appends_eos"] == native_eos
    for text in ["Question: Hello", "[BOS] Hello"]:
        original = tok.encode(text)
        plain_condition = encode_with_bos_policy(tok, text, "no_bos")
        bos_condition = encode_with_bos_policy(tok, text, "bos")
        assert bos_condition == [1, *plain_condition]
        assert plain_condition == original[int(native_bos) :]
        assert encode_with_bos_policy(tok, text) == original
        assert tok.encode(text) == original  # Never mutate checkpoint tokenizer.
        assert encode_with_bos_policy(
            tok, text, "bos", add_special_tokens=False
        ) == tok.encode(text, add_special_tokens=False)


@pytest.mark.parametrize("native_bos", [False, True])
def test_lm_eval_pair_encoding_has_one_controlled_prefix(native_bos):
    tok = tokenizer(native_bos)
    bridge = SimpleNamespace(tokenizer=tok, original_model=torch.nn.Linear(1, 1))
    native = BridgeLM(bridge, compute_metrics=False)
    no_bos = BridgeLM(bridge, compute_metrics=False, bos_policy="no_bos")
    bos = BridgeLM(bridge, compute_metrics=False, bos_policy="bos")
    context, continuation = "Question:", " Hello"

    def encoded(lm):
        prefix, suffix = lm._encode_pair(context, continuation)
        return prefix + suffix

    assert encoded(native) == tok.encode(context + continuation)
    assert encoded(bos) == [1, *encoded(no_bos)]


@pytest.mark.parametrize("native_bos", [False, True])
def test_full_sentence_mcq_respects_native_bos(native_bos):
    tok = tokenizer(native_bos)
    example = MCQExample("", ["Hello", "Question"], 0, full_sentence_choices=True)
    ctx, choices = _tokenize_mcq(tok, example)
    assert ctx == ([1] if native_bos else [])
    for text, choice in zip(example.choices, choices, strict=True):
        assert ctx + choice == tok.encode(text)


def test_missing_bos_is_an_error_not_a_silent_duplicate_condition():
    tok = tokenizer()
    tok.bos_token = None
    with pytest.raises(ValueError, match="no BOS token"):
        encode_with_bos_policy(tok, "Hello", "bos")


@pytest.mark.parametrize("native_bos", [False, True])
def test_shared_loader_restores_checkpoint_default(monkeypatch, native_bos):
    from looped_internals.models import loading
    from looped_internals.models.minimal import MinimalRecurrentConfig

    native = tokenizer(native_bos)
    forced = tokenizer(True)
    # Keep the real TransformerBridge tokenizer property: reassignment through
    # its setter calls setup_tokenizer and can silently force BOS back on.
    bridge, _ = loading.load_model(
        loading.ModelSpec(
            MinimalRecurrentConfig(d_model=8, n_heads=2, n_layers=1, d_vocab=6)
        ),
        device="cpu",
    )
    bridge.original_model.config._commit_hash = "pinned"
    bridge.tokenizer = forced
    monkeypatch.setattr(
        loading.TransformerBridge, "boot_transformers", lambda *a, **kw: bridge
    )
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return forced if kwargs.get("add_bos_token") else native

    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", load)
    result, _ = loading.load_model("gpt2", device="cpu")
    assert result.tokenizer is native
    assert result.cfg.default_prepend_bos == native_bos
    assert result.cfg.tokenizer_prepends_bos == native_bos
    expected = native.encode("Question: Hello")
    assert BridgeLM(result).tok_encode("Question: Hello") == expected
    assert result.to_tokens("Question: Hello").flatten().tolist() == expected
    assert len(calls) == 1
    assert calls[0]["revision"] == "pinned"
    assert "add_bos_token" not in calls[0]


def test_no_bos_full_sentence_scoring_uses_previous_position(monkeypatch):
    import importlib

    module = importlib.import_module("looped_internals.evaluation.logit_lens")
    logits = torch.zeros(2, 3, 6)
    # Choice 0 predicts its next tokens correctly. Choice 1 instead predicts
    # its current tokens: the former off-by-one would incorrectly prefer it.
    logits[0, 0, 5] = logits[0, 1, 4] = 20
    logits[1, 0, 3] = logits[1, 1, 4] = logits[1, 2, 5] = 20
    monkeypatch.setattr(module, "logit_lens", lambda bridge, hidden: logits)
    bridge = SimpleNamespace(
        run_with_cache=lambda *a, **kw: (
            None,
            {"blocks.0.hook_out": torch.zeros(2, 3, 1)},
        )
    )
    example = MCQExample(
        "", ["Question Hello :", "Question : Hello"], 0, full_sentence_choices=True
    )
    predictions = module.score_logitlens(
        bridge,
        tokenizer(),
        example,
        ModelLayerStructure([LayerGroup("blocks", 1, False)]),
        1,
        torch.device("cpu"),
    )
    assert all(result.tolist() == [0] for result in predictions[:2])
    assert predictions[2] is None


@pytest.mark.parametrize("native_bos", [False, True])
def test_behavior_plain_text_follows_native_and_chat_does_not_double_bos(native_bos):
    from experiments.behavior_support import (
        BehaviorRunConfig,
        build_input_ids,
        count_system_tokens,
    )

    tok = tokenizer(native_bos)
    config = BehaviorRunConfig("huginn", "", "no_prompt")
    actual = build_input_ids(tok, "Hello", config, 0, "cpu")
    assert actual[0].tolist() == tok.encode("Hello")
    assert count_system_tokens(tok, "", None) == int(native_bos)
    tok.chat_template = "[BOS] {{ messages[-1]['content'] }}"
    actual = build_input_ids(tok, "Hello", config, 0, "cpu")
    assert actual[0].tolist() == [1, 5]
    assert count_system_tokens(tok, "", None) == 1
