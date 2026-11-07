"""Checkpoint tokenization and explicit, paired BOS interventions."""

from typing import Literal

BOSPolicy = Literal["native", "bos", "no_bos"]


def tokenizer_metadata(tokenizer) -> dict:
    """Record observed special-token behavior, not just configuration flags."""
    probe = "Question: Hello"
    plain = tokenizer.encode(probe, add_special_tokens=False)
    native = tokenizer.encode(probe, add_special_tokens=True)
    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "bos_token_id": bos,
        "eos_token_id": eos,
        "prepends_bos": bool(native and native[0] == bos and plain[0] != bos),
        "appends_eos": bool(native and native[-1] == eos and plain[-1] != eos),
        "probe_text": probe,
        "probe_native_ids": native,
        "probe_content_ids": plain,
    }


def encode_with_bos_policy(
    tokenizer, text: str, policy: BOSPolicy = "native", *, add_special_tokens=True
) -> list[int]:
    """Change only the automatic leading BOS; retain native content and EOS.

    Both forced conditions start from the same checkpoint encoding. This avoids
    differences between automatic and manual special-token processing changing
    the content tokens. Literal special tokens present in the text are retained.
    Continuation fragments (``add_special_tokens=False``) are never prefixed.
    """
    if policy not in {"native", "bos", "no_bos"}:
        raise ValueError("bos_policy must be native, bos, or no_bos")
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if policy == "native" or not add_special_tokens:
        return ids
    info = tokenizer_metadata(tokenizer)
    if info["prepends_bos"]:
        if not ids or ids[0] != info["bos_token_id"]:
            raise ValueError("Tokenizer's observed BOS prefix changed for this input")
        ids = ids[1:]
    if policy == "bos":
        if info["bos_token_id"] is None:
            raise ValueError("Cannot force BOS: checkpoint tokenizer has no BOS token")
        ids = [info["bos_token_id"], *ids]
    return ids
