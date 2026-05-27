"""Tests for the workflow trace logger (tools/tracing.py).

The JSONL writer must survive disk failures, the LangSmith detection must
key off both env vars together, and `tail_runs` must read newest-first.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools import tracing
from tools.tracing import langsmith_enabled, run_summary, tail_runs, trace_run


@pytest.fixture
def trace_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "runs.jsonl"
    monkeypatch.setattr(tracing, "_trace_log_path", lambda: p)
    return p


def test_trace_run_writes_one_row_per_invocation(trace_path):
    with trace_run("t1", "hello") as evt:
        evt["intents"] = ["general"]
    rows = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert "general" in rows[0]
    assert "t1" in rows[0]


def test_trace_run_captures_latency(trace_path):
    with trace_run("t1", "hello") as evt:
        evt["x"] = 1
    events = tail_runs(path=trace_path)
    assert events[0]["latency_seconds"] >= 0
    assert events[0]["x"] == 1


def test_trace_run_logs_errors_and_reraises(trace_path):
    with pytest.raises(RuntimeError):
        with trace_run("t1", "broken") as evt:
            evt["intents"] = ["general"]
            raise RuntimeError("boom")
    events = tail_runs(path=trace_path)
    assert events[0]["error"] == "boom"
    assert events[0]["error_type"] == "RuntimeError"


def test_trace_run_swallows_write_failures(monkeypatch, caplog):
    """A JSONL write failure must never raise."""
    def _broken(_event):
        raise OSError("disk full")
    monkeypatch.setattr(tracing, "_append_jsonl", _broken)
    # Should not raise
    with trace_run("t1", "hello"):
        pass
    assert "Trace JSONL write failed" in caplog.text


def test_tail_runs_returns_newest_first(trace_path):
    for i in range(3):
        with trace_run("t1", f"q{i}") as evt:
            evt["i"] = i
    events = tail_runs(limit=2, path=trace_path)
    assert len(events) == 2
    assert events[0]["i"] == 2  # newest first


def test_tail_runs_skips_malformed_lines(trace_path):
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text('{"ok": 1}\nNOT JSON\n{"ok": 2}\n')
    events = tail_runs(path=trace_path)
    assert len(events) == 2
    assert all("ok" in e for e in events)


def test_tail_runs_returns_empty_for_missing_log(tmp_path):
    assert tail_runs(path=tmp_path / "never.jsonl") == []


def test_run_summary_empty():
    assert run_summary([])["total"] == 0


def test_run_summary_percentiles():
    events = [{"latency_seconds": v} for v in [0.1, 0.2, 0.3, 0.4, 0.5]]
    s = run_summary(events)
    assert s["total"] == 5
    assert s["p50_latency_seconds"] == 0.3
    assert s["p95_latency_seconds"] == 0.5


def test_run_summary_counts_errors_and_emergencies():
    events = [
        {"latency_seconds": 0.1, "error": "x"},
        {"latency_seconds": 0.2, "is_emergency": True},
        {"latency_seconds": 0.3},
    ]
    s = run_summary(events)
    assert s["error_rate"] == round(1 / 3, 3)
    assert s["emergency_count"] == 1


# ---------- LangSmith detection ----------

def test_langsmith_off_by_default(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert langsmith_enabled() is False


def test_langsmith_requires_both_key_and_flag(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls-xxx")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    # Key without the tracing flag → not enabled
    assert langsmith_enabled() is False
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    assert langsmith_enabled() is True


def test_langsmith_accepts_either_alias(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls-xxx")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "1")
    assert langsmith_enabled() is True
