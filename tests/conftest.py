"""Tests never use the user's persistent experiment cache."""

import pytest


@pytest.fixture(autouse=True)
def isolated_experiment_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("LOOPED_INTERNALS_CACHE_MODE", "off")
    monkeypatch.setenv("LOOPED_INTERNALS_CACHE_DIR", str(tmp_path / "cache"))
