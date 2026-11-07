"""Explicit, read-only reuse of completed notebook summaries."""

import json
from pathlib import Path

import torch

from experiments.cache import _json


class SavedRun:
    """Match recorded scientific settings; never relabel historical data as fresh.

    Selecting a source explicitly asserts that its scientific implementation is
    suitable. Old runs do not record reduction versions or immutable identities
    for every checkpoint, so they are not imported into the fresh-result cache.
    """

    def __init__(self, directory, settings, *, analysis_version=1):
        self.directory = Path(directory).expanduser().resolve() if directory else None
        self.data = {}
        self.settings = {}
        if self.directory is None:
            return
        self.settings = json.loads((self.directory / "run.json").read_text())
        ignored = {
            "mode",
            "output_dir",
            "cache",
            "models",
            "model_recurrences",
            "figure_pairs",
            "feedforward_models",
            "looped_models",
            "long_pairs",
            "coda_pairs",
            "entropy_pairs",
            "source_run",
        }
        mismatches = [
            key
            for key, value in settings.items()
            if key not in ignored and _json(value) != self.settings.get(key)
        ]
        if analysis_version != self.settings.get("analysis_version", 1):
            mismatches.append("analysis_version")
        if mismatches:
            raise ValueError(
                f"Saved run differs in {sorted(set(mismatches))}: {self.directory}"
            )
        self.data = torch.load(
            self.directory / "overview.pt", map_location="cpu", weights_only=False
        )
        print(
            f"Reusing explicitly selected historical summaries: {self.directory}",
            flush=True,
        )

    def __contains__(self, pair):
        return pair in self.data.get("summaries", {}) and pair not in getattr(
            self, "incompatible_pairs", set()
        )

    def check_pairs(self, settings):
        self.incompatible_pairs = set()
        for field in ("entropy_pairs", "coda_pairs"):
            old = {tuple(p) for p in self.settings.get(field, [])}
            new = {tuple(p) for p in settings.get(field, [])}
            self.incompatible_pairs.update(old ^ new)

    def info(self, pair):
        info = self.data.get("run", {}).get(pair, {})
        return {**info, "source_run": str(self.directory)}

    def tokenization(self, name):
        path = self.directory / "tokenization.json"
        return json.loads(path.read_text())[name]
