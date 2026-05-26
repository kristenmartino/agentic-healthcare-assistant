"""Tests for the search-backend introspection helpers.

`configured_backend()` answers "what would we try first?" — used in the
sidebar to set expectations before the user even types a query.

`effective_backend()` answers "what actually produced these results?" —
used in the per-query state panel + the eval summary to tell a real
MedlinePlus/Tavily citation from a stub-fallback.
"""
from __future__ import annotations

from tools.medical_search import configured_backend, effective_backend


def test_configured_backend_prefers_tavily_when_key_set(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert configured_backend(tavily_api_key="tvly-xxx") == "tavily"


def test_configured_backend_reads_env_when_arg_missing(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    assert configured_backend() == "tavily"


def test_configured_backend_defaults_to_ddg(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert configured_backend() == "duckduckgo"


def test_effective_backend_empty_input():
    assert effective_backend([]) == "unknown"


def test_effective_backend_all_stub():
    results = [
        {"source": "stub", "title": "x"},
        {"source": "stub", "title": "y"},
    ]
    assert effective_backend(results) == "stub"


def test_effective_backend_real_results_win_over_stub():
    """If even one result has a real source, that's what we report."""
    results = [
        {"source": "stub", "title": "fallback"},
        {"source": "duckduckgo", "title": "real"},
    ]
    assert effective_backend(results) == "duckduckgo"


def test_effective_backend_tavily_label():
    assert effective_backend([{"source": "tavily", "title": "x"}]) == "tavily"
