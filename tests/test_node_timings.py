"""Tests for the per-node latency instrumentation (issue #12).

The trace log used to capture only total latency, which can't tell a cold
boot apart from slow in-request LLM hops. The timing wrapper in graph.py
records each node's wall-clock duration into `node_timings`, and the
state reducer unions entries so parallel fan-out branches don't clobber
each other.
"""
from __future__ import annotations

from graph import _timed, build_workflow
from state import _merge_timings

# ---------- reducer ----------

def test_merge_timings_unions_entries():
    assert _merge_timings({"safety": 1.0}, {"classify_intent": 2.0}) == {
        "safety": 1.0, "classify_intent": 2.0,
    }


def test_merge_timings_handles_empty_sides():
    assert _merge_timings(None, {"a": 1.0}) == {"a": 1.0}
    assert _merge_timings({"a": 1.0}, None) == {"a": 1.0}
    assert _merge_timings(None, None) == {}


# ---------- wrapper ----------

def test_timed_wrapper_adds_node_timing():
    def node(state):
        return {"foo": "bar"}

    wrapped = _timed("mynode", node)
    out = wrapped({})
    assert out["foo"] == "bar"  # passthrough preserved
    assert "mynode" in out["node_timings"]
    assert isinstance(out["node_timings"]["mynode"], float)
    assert out["node_timings"]["mynode"] >= 0


def test_timed_wrapper_passes_non_dict_through():
    wrapped = _timed("n", lambda state: "not-a-dict")
    assert wrapped({}) == "not-a-dict"


# ---------- end-to-end through the compiled graph ----------

def test_workflow_records_node_timings(monkeypatch):
    """A simple (general) query in stub mode should route safety →
    classify_intent → compose_response and surface a timing for each."""
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_MODE", raising=False)

    graph = build_workflow(with_checkpoint=False)
    result = graph.invoke({"user_input": "hello there", "history": []},
                          config={"configurable": {"thread_id": "timing-test"}})

    timings = result.get("node_timings") or {}
    assert "safety" in timings
    assert "classify_intent" in timings
    assert "compose_response" in timings
    assert all(v >= 0 for v in timings.values())
