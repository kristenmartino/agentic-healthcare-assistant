"""Tests for tools/medical_search — multi-backend fallback chain.

The fallback chain (Tavily → DDG → stub) must guarantee the caller never
sees an empty list. Real-world failure modes we have to handle:

- DDG raises an exception (rate-limit, TLS, ImportError)  → caught, stub fires
- DDG returns [] silently (rate-limit shape on some days) → also fires stub
  (this was the bug behind the 4/14 gpt-4o-mini eval failures)
- Tavily returns [] when the query filters everything out → fall through
- No internet at all                                       → stub always works
"""
from __future__ import annotations

from tools import medical_search as ms


def test_stub_is_floor_when_ddg_raises(monkeypatch):
    monkeypatch.setattr(ms, "_ddg_search",
                        lambda q, k: (_ for _ in ()).throw(RuntimeError("no net")))
    out = ms.medical_search("pneumonia", top_k=2)
    assert out
    assert out[0]["source"] == "stub"


def test_stub_is_floor_when_ddg_returns_empty(monkeypatch):
    """Regression test for the gpt-4o-mini eval failures: DDG rate-limited
    silently returns [] without raising. We must still hand back a non-empty
    list so the medical_search_node populates `medical_info`."""
    monkeypatch.setattr(ms, "_ddg_search", lambda q, k: [])
    out = ms.medical_search("pneumonia", top_k=2)
    assert out
    assert out[0]["source"] == "stub"


def test_passes_through_ddg_real_results(monkeypatch):
    """If DDG returns something, we use it — stub does not override."""
    fake = [{"title": "real", "snippet": "x", "url": "https://medlineplus.gov", "source": "duckduckgo"}]
    monkeypatch.setattr(ms, "_ddg_search", lambda q, k: fake)
    out = ms.medical_search("pneumonia", top_k=2)
    assert out == fake


def test_tavily_empty_falls_through_to_ddg(monkeypatch):
    """A Tavily 0-hit shouldn't stop us; DDG still gets a turn."""
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(ms, "_tavily_search", lambda q, k, tk: [])
    fake_ddg = [{"title": "ddg result", "snippet": "y", "url": "https://who.int", "source": "duckduckgo"}]
    monkeypatch.setattr(ms, "_ddg_search", lambda q, k: fake_ddg)
    out = ms.medical_search("flu", top_k=2, tavily_api_key="fake-key")
    assert out == fake_ddg


def test_tavily_raises_falls_through_to_ddg(monkeypatch):
    monkeypatch.setattr(ms, "_tavily_search",
                        lambda q, k, tk: (_ for _ in ()).throw(RuntimeError("403")))
    fake_ddg = [{"title": "ddg", "snippet": "z", "url": "https://cdc.gov", "source": "duckduckgo"}]
    monkeypatch.setattr(ms, "_ddg_search", lambda q, k: fake_ddg)
    out = ms.medical_search("covid", top_k=2, tavily_api_key="fake-key")
    assert out == fake_ddg


def test_all_real_backends_fail_returns_stub(monkeypatch):
    """The worst case — Tavily AND DDG both fail (empty or exception).
    Stub still produces at least one entry so the graph completes."""
    monkeypatch.setattr(ms, "_tavily_search",
                        lambda q, k, tk: (_ for _ in ()).throw(RuntimeError("503")))
    monkeypatch.setattr(ms, "_ddg_search", lambda q, k: [])
    out = ms.medical_search("dengue", top_k=3, tavily_api_key="fake-key")
    assert out
    assert all(r["source"] == "stub" for r in out)
