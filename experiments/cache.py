"""Small, versioned experiment artifacts; no model or dataset work at import."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from looped_internals._random import rng_state

DEFAULT_CACHE_DIR = Path("/data-gauss/hugbla/looped-internals-cache")
# Bump when collection/tokenization semantics change. Notebook reductions have
# separate versions, so a plotting edit or a new paper model preserves entries.
COLLECTION_VERSION = 1


def _json(value):
    if is_dataclass(value):
        return _json(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, (Path, torch.dtype)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def digest(value):
    encoded = json.dumps(_json(value), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_save(value, path):
    """Readers see either the previous complete file or the new complete file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _source_identity(source, filename, *, subfolder=None, revision=None):
    """Resolve Hub commits without loading weights; hash local checkpoint files."""
    from huggingface_hub import hf_hub_download

    local = Path(source).expanduser()
    if local.is_dir():
        directory = local / subfolder if subfolder else local
        files = {}
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {
                ".json",
                ".bin",
                ".safetensors",
                ".py",
                ".model",
                ".txt",
            }:
                with path.open("rb") as stream:
                    files[str(path.relative_to(directory))] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
        if filename not in files:
            raise FileNotFoundError(directory / filename)
        return {"directory": str(directory.resolve()), "files": files}, None
    path = Path(
        hf_hub_download(source, filename, subfolder=subfolder, revision=revision)
    )
    snapshot = next((p for p in path.parents if p.parent.name == "snapshots"), None)
    if snapshot is None:
        raise ValueError(f"Cannot determine the checkpoint revision for {source}")
    return {
        "repo": source,
        "revision": snapshot.name,
        "subfolder": subfolder,
    }, snapshot.name


class ExperimentCache:
    def __init__(self, directory=None, mode=None):
        self.directory = Path(
            directory
            or os.environ.get("LOOPED_INTERNALS_CACHE_DIR")
            or DEFAULT_CACHE_DIR
        ).expanduser()
        self.mode = mode or os.environ.get("LOOPED_INTERNALS_CACHE_MODE", "reuse")
        if self.mode not in {"reuse", "require", "off"}:
            raise ValueError("cache_mode must be reuse, require, or off")
        self._models = {}
        self._used = set()

    @property
    def enabled(self):
        return self.mode != "off"

    @property
    def settings(self):
        return {
            "directory": str(self.directory),
            "mode": self.mode,
            "collection_version": COLLECTION_VERSION,
        }

    def model(self, name):
        """Return a spec pinned to the exact checkpoint/tokenizer in the key."""
        from experiments.support import MODEL_CHECKPOINTS
        from looped_internals.models import MODEL_SPECS
        from looped_internals.models.parcae import ParcaeCheckpoint

        spec = MODEL_SPECS[MODEL_CHECKPOINTS.get(name, name)]
        if not self.enabled:
            return spec, {}
        if name not in self._models:
            source = spec.source
            if isinstance(source, str):
                identity, revision = _source_identity(
                    source, "config.json", revision=spec.revision
                )
                spec = replace(spec, revision=revision)
            elif isinstance(source, ParcaeCheckpoint):
                weights, revision = _source_identity(
                    source.weights,
                    "config.json",
                    subfolder=source.subfolder,
                    revision=source.revision,
                )
                tokenizer, tokenizer_revision = _source_identity(
                    source.tokenizer,
                    "tokenizer.json",
                    revision=source.tokenizer_revision,
                )
                identity = {"weights": weights, "tokenizer": tokenizer}
                spec = replace(
                    spec,
                    source=replace(
                        source, revision=revision, tokenizer_revision=tokenizer_revision
                    ),
                )
            else:
                identity = asdict(source)
            self._models[name] = spec, {"source": identity, "spec": _json(spec)}
        return self._models[name]

    def key(self, config=None, name=None, *, analysis_version=1, **parameters):
        settings = {}
        if config is not None:
            settings = {
                field: getattr(config, field)
                for field in ("dtype", "seed", "num_steps", "bos_policy")
                if hasattr(config, field)
            }
            settings["device_type"] = config.device.split(":")[0]
        if self.enabled and name is not None:
            settings["checkpoint"] = self.model(name)[1]
        return _json(
            {
                "collection_version": COLLECTION_VERSION,
                "analysis_version": analysis_version,
                "versions": {
                    p: version(p)
                    for p in (
                        "torch",
                        "transformer-lens",
                        "transformers",
                        "parcae-lm",
                        "lm-eval",
                        "scikit-learn",
                    )
                },
                "model": name,
                **settings,
                **parameters,
            }
        )

    @contextmanager
    def entry(self, namespace, key, *, device=None):
        path = self.directory / namespace / digest(key)
        entry = CacheEntry(path if self.enabled else None, device=device)
        if self.enabled:
            self._used.add(str(path.resolve()))
        if not self.enabled:
            yield entry
            return
        if self.mode == "require":
            if not entry.complete:
                raise FileNotFoundError(
                    f"Missing complete cache entry: {path}. Run with cache_mode='reuse' to collect it."
                )
            yield entry
            return
        path.mkdir(parents=True, exist_ok=True)
        with (path / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                manifest = path / "request.json"
                if not manifest.exists():
                    manifest.write_text(json.dumps(key, indent=2) + "\n")
                print(
                    f"Cache {'hit' if entry.complete else 'resume' if entry.pending else 'miss'}: {namespace}/{path.name[:12]}",
                    flush=True,
                )
                yield entry
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def run(self, namespace, key, compute, *args):
        with self.entry(namespace, key) as entry:
            if entry.complete:
                return entry.load()
            value = compute(*args)
            entry.save(value)
            return value

    def read(self, namespace, key):
        """Optional reuse of a completed artifact from another collector."""
        entry = CacheEntry(self.directory / namespace / digest(key))
        if self.enabled and entry.complete:
            self._used.add(str(entry.path.resolve()))
            return entry.load()
        return None

    def write_manifest(self, run_dir):
        """Link a run to the exact requests and checkpoint identities it reused."""
        (Path(run_dir) / "cache.json").write_text(
            json.dumps({**self.settings, "entries": sorted(self._used)}, indent=2)
            + "\n"
        )


class CacheEntry:
    """One completed result and, while unfinished, one replaceable checkpoint."""

    def __init__(self, path, *, device=None):
        self.path = path
        self.device = device
        self._next_part = None

    @property
    def complete(self):
        return self.path is not None and (self.path / "result.pt").exists()

    @property
    def pending(self):
        return self.path is not None and (
            (self.path / "progress.pt").exists() or (self.path / "parts").exists()
        )

    def records(self):
        if self.path is None:
            return []
        paths = sorted((self.path / "parts").glob("*.pt"))
        if [p.stem for p in paths] != [f"{i:06d}" for i in range(len(paths))]:
            raise ValueError(f"Non-contiguous checkpoint records: {self.path}")
        self._next_part = len(paths)
        return [torch.load(p, map_location="cpu", weights_only=False) for p in paths]

    def append(self, value):
        """Commit one example without rewriting previously completed examples."""
        if self.path is not None:
            if self._next_part is None:
                self._next_part = len(list((self.path / "parts").glob("*.pt")))
            atomic_save(
                {"value": value, "rng": rng_state(self.device)},
                self.path / "parts" / f"{self._next_part:06d}.pt",
            )
            self._next_part += 1

    @property
    def progress(self):
        if self.path is not None and (self.path / "progress.pt").exists():
            return torch.load(
                self.path / "progress.pt", map_location="cpu", weights_only=False
            )
        return None

    def load(self):
        return torch.load(
            self.path / "result.pt", map_location="cpu", weights_only=False
        )

    def checkpoint(self, value):
        if self.path is not None:
            atomic_save(
                {"value": value, "rng": rng_state(self.device)},
                self.path / "progress.pt",
            )

    def save(self, value):
        if self.path is not None:
            atomic_save(value, self.path / "result.pt")
            (self.path / "progress.pt").unlink(missing_ok=True)
            if (self.path / "parts").exists():
                shutil.rmtree(self.path / "parts")

    def export(self, value, path):
        """New run directories reference immutable completed cache artifacts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.complete:
            path.symlink_to((self.path / "result.pt").resolve())
        else:
            atomic_save(value, path)
