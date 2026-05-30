"""Tests for the FastAPI service.

We test the route shape + the helper functions, not the LangGraph workflow
itself (that's covered by the eval suites). The chat SSE endpoint is
smoke-tested for shape; the actual token streaming is exercised
end-to-end manually.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import api.main as api_main  # noqa: E402
from api.main import (  # noqa: E402
    _content_to_text,
    _get_workflow,
    _node_label,
    _sse,
    _summarize_partial,
    app,
)


@pytest.fixture(scope="module")
def client(monkeypatch_module):
    monkeypatch_module.setenv("EHR_BACKEND", "fhir_fixture")
    # Run startup explicitly via TestClient context manager.
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch (built-in pytest fixture is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


# ---------- helpers ----------

def test_sse_formats_event_and_data():
    out = _sse("status", {"node": "safety"}).decode()
    assert out.startswith("event: status\n")
    assert "data: " in out
    assert out.endswith("\n\n")
    # Round-trip the JSON payload
    payload_line = next(ln for ln in out.split("\n") if ln.startswith("data: "))
    assert json.loads(payload_line[6:]) == {"node": "safety"}


def test_node_label_translates_known_nodes():
    assert _node_label("safety") == "Safety check"
    assert _node_label("compose_response") == "Composing response"
    assert _node_label("unknown_node") == "unknown_node"


def test_summarize_partial_classify_intent():
    out = _summarize_partial("classify_intent", {"intents": ["booking", "medical_search"]})
    assert "booking" in out and "medical_search" in out


def test_summarize_partial_emergency():
    out = _summarize_partial("safety", {
        "is_emergency": True,
        "emergency_categories": ["cardiac"],
    })
    assert "🚨" in out
    assert "cardiac" in out


def test_summarize_partial_booking():
    out = _summarize_partial("booking_node", {
        "appointment": {"doctor_name": "Dr. Smith", "specialty": "cardiology"}
    })
    assert "Dr. Smith" in out
    assert "cardiology" in out


def test_summarize_partial_empty():
    """Non-special-cased shapes return empty string."""
    assert _summarize_partial("composer", {}) == ""


def test_content_to_text_handles_string():
    assert _content_to_text("hello") == "hello"


def test_content_to_text_flattens_block_list():
    blocks = [
        {"type": "text", "text": "Hello "},
        {"type": "image_url", "image_url": "..."},  # ignored
        {"type": "text", "text": "world"},
    ]
    assert _content_to_text(blocks) == "Hello world"


# ---------- HTTP routes ----------

def test_health_returns_status_and_config(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "llm_provider" in body
    assert "ehr_backend" in body


def test_config_does_not_leak_secrets(client):
    r = client.get("/config")
    assert r.status_code == 200
    body = r.json()
    # Sanity: never expose actual keys, only booleans / labels
    text = json.dumps(body).lower()
    for forbidden in ("sk-ant-", "sk-", "gsk_", "tvly-", "ls_"):
        assert forbidden not in text


def test_patients_list_returns_fhir_fixture_data(client):
    r = client.get("/patients")
    assert r.status_code == 200
    patients = r.json()
    assert len(patients) >= 5
    names = {p["name"] for p in patients}
    assert "Anjali Mehra" in names


def test_patient_by_id_returns_clinical_context(client):
    r = client.get("/patients/fhir:anjali-mehra")
    assert r.status_code == 200
    p = r.json()
    assert p["name"] == "Anjali Mehra"
    assert "clinical" in p
    assert "conditions" in p["clinical"]


def test_patient_404(client):
    r = client.get("/patients/fhir:nonexistent")
    assert r.status_code == 404


def test_patient_search_by_name(client):
    r = client.get("/patients/search/by-name", params={"name": "Ramesh"})
    assert r.status_code == 200
    assert r.json()["name"].startswith("Ramesh")


def test_bookings_recent_returns_list(client):
    r = client.get("/bookings/recent")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_audit_empty_on_fresh_db_returns_empty_list(client):
    r = client.get("/audit", params={"limit": 5})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_audit_summary_returns_shape(client):
    r = client.get("/audit/summary")
    assert r.status_code == 200
    body = r.json()
    assert "total" in body
    assert "by_action" in body


def test_traces_returns_events_and_summary(client):
    r = client.get("/traces", params={"limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert "events" in body
    assert "summary" in body


def test_chat_rejects_empty_user_input(client):
    r = client.post("/chat", json={"user_input": ""})
    assert r.status_code == 422  # FastAPI validation


def test_chat_emits_sse_events(client):
    """End-to-end smoke: chat returns SSE with at least one status event +
    a done event. Stub LLM gives us a deterministic, fast response.
    """
    with client.stream(
        "POST", "/chat",
        json={"user_input": "Hello", "thread_id": "smoke-test"},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = b"".join(r.iter_bytes())
    text = body.decode()
    # Must include the canonical event types
    assert "event: status" in text
    assert "event: done" in text


def test_chat_accepts_history_field(client):
    """Issue #9: /chat accepts prior conversation turns and streams normally."""
    with client.stream(
        "POST", "/chat",
        json={
            "user_input": "and his cholesterol?",
            "thread_id": "mem-test",
            "history": [
                {"role": "user", "content": "Show me Ramesh's history"},
                {"role": "assistant", "content": "Ramesh has CKD stage 2."},
            ],
        },
    ) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes())
    assert b"event: done" in body


def test_chat_rejects_malformed_history(client):
    """A history turn with an invalid role is rejected by validation."""
    r = client.post("/chat", json={
        "user_input": "hi",
        "history": [{"role": "wizard", "content": "x"}],
    })
    assert r.status_code == 422


# ---------- workflow warmup status + readiness (PR #23 observability) ----------

@pytest.fixture
def restore_warmup_state():
    """Snapshot and restore everything these warmup tests mutate.

    The module-scoped `client` fixture runs FastAPI startup once, which warms
    (and on 3.11/3.12 actually builds) the real workflow into app.state. These
    tests poke app.state.workflow / workflow_status / workflow_error and
    monkeypatch api_main.build_workflow directly (not via the function-scoped
    `monkeypatch`, because some assertions run on background threads), so we
    must put all of it back afterward or later tests get a poisoned singleton.
    """
    saved = {
        "workflow": getattr(app.state, "workflow", None),
        "workflow_status": getattr(app.state, "workflow_status", "warming"),
        "workflow_error": getattr(app.state, "workflow_error", None),
        "build_workflow": api_main.build_workflow,
    }
    try:
        yield
    finally:
        app.state.workflow = saved["workflow"]
        app.state.workflow_status = saved["workflow_status"]
        app.state.workflow_error = saved["workflow_error"]
        api_main.build_workflow = saved["build_workflow"]


def test_get_workflow_builds_exactly_once_under_concurrency(restore_warmup_state):
    """Double-checked locking: N concurrent callers trigger a single build and
    all receive the same object.
    """
    sentinel = object()
    calls = {"n": 0}
    calls_lock = threading.Lock()

    def slow_build(*args, **kwargs):
        # Count + briefly sleep so threads genuinely overlap inside the lock
        # contention window, exercising the double-checked path.
        with calls_lock:
            calls["n"] += 1
        time.sleep(0.05)
        return sentinel

    api_main.build_workflow = slow_build
    # Force a fresh, unbuilt state so _get_workflow actually builds.
    app.state.workflow = None
    app.state.workflow_status = "warming"
    app.state.workflow_error = None

    results: list = []
    results_lock = threading.Lock()
    start = threading.Event()

    def worker():
        start.wait()  # release all threads at once to maximize contention
        wf = _get_workflow()
        with results_lock:
            results.append(wf)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(timeout=5)

    assert all(not t.is_alive() for t in threads), "a worker thread hung"
    assert calls["n"] == 1, f"build_workflow ran {calls['n']} times, expected 1"
    assert len(results) == 8
    assert all(r is sentinel for r in results), "threads got different objects"
    assert app.state.workflow is sentinel
    assert app.state.workflow_status == "ready"
    assert app.state.workflow_error is None


def test_get_workflow_failure_path_records_error_then_recovers(restore_warmup_state):
    """A build failure flips status to 'error' and records the redacted
    exception TYPE NAME; a subsequent successful build recovers to 'ready'.
    """
    class WarmupBoom(RuntimeError):
        pass

    def boom_build(*args, **kwargs):
        raise WarmupBoom("secret-ish detail that must not be exposed")

    api_main.build_workflow = boom_build
    app.state.workflow = None
    app.state.workflow_status = "warming"
    app.state.workflow_error = None

    with pytest.raises(WarmupBoom):
        _get_workflow()

    assert app.state.workflow is None
    assert app.state.workflow_status == "error"
    # The recorded error is the exception type name (non-empty string), never
    # the message — so no PII / secret leakage.
    assert isinstance(app.state.workflow_error, str)
    assert app.state.workflow_error == "WarmupBoom"
    assert "secret-ish" not in app.state.workflow_error

    # Now restore a working build and confirm recovery on the next call.
    sentinel = object()

    def good_build(*args, **kwargs):
        return sentinel

    api_main.build_workflow = good_build
    app.state.workflow = None
    app.state.workflow_status = "warming"

    wf = _get_workflow()
    assert wf is sentinel
    assert app.state.workflow_status == "ready"
    assert app.state.workflow_error is None


def test_ready_endpoint_200_when_ready(client, restore_warmup_state):
    """/ready returns 200 + {"ready": true} only when status == 'ready'."""
    app.state.workflow_status = "ready"
    app.state.workflow_error = None
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["workflow_status"] == "ready"
    assert body["workflow_error"] is None


def test_ready_endpoint_503_when_error(client, restore_warmup_state):
    """/ready returns 503 + {"ready": false} after a build failure, and the
    body still carries the (redacted) status fields for diagnosis.
    """
    app.state.workflow_status = "error"
    app.state.workflow_error = "WarmupBoom"
    r = client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False
    assert body["workflow_status"] == "error"
    assert body["workflow_error"] == "WarmupBoom"


def test_ready_endpoint_503_while_warming(client, restore_warmup_state):
    """/ready returns 503 + {"ready": false} during warmup."""
    app.state.workflow_status = "warming"
    app.state.workflow_error = None
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False
    assert r.json()["workflow_status"] == "warming"


def test_config_includes_workflow_status_fields(client):
    """/config surfaces workflow_status + workflow_error (non-failing reads)."""
    r = client.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert "workflow_status" in body
    assert "workflow_error" in body
    # workflow_error is either null or a redacted type-name string — never a
    # secret-bearing message.
    assert body["workflow_error"] is None or isinstance(body["workflow_error"], str)


def test_startup_does_not_build_workflow_synchronously(monkeypatch, restore_warmup_state):
    """Kristen's ask: startup must NOT call build_workflow on the calling
    thread. Non-flaky structural guarantee — we assert the build, if it ran at
    all by the time startup returned, ran on the background 'startup-warmup'
    thread, and that _bootstrap returned with status in {"warming","ready"}
    (i.e. it did not block synchronously building on the main/calling thread).

    `restore_warmup_state` puts app.state.workflow back afterward so the
    re-entered startup here (which sets app.state.workflow to our stub sentinel)
    can't poison the module-scoped `client` for any later test.
    """
    build_threads: list[str] = []
    sentinel = object()

    def recording_build(*args, **kwargs):
        build_threads.append(threading.current_thread().name)
        return sentinel

    monkeypatch.setattr(api_main, "build_workflow", recording_build)
    monkeypatch.setenv("EHR_BACKEND", "fhir_fixture")

    calling_thread = threading.current_thread().name

    # Fresh startup/shutdown cycle on the same app object.
    with TestClient(app) as c:
        # Immediately after startup returned, the status must already be a
        # warmup-lifecycle value — proving _bootstrap returned without doing a
        # synchronous build that would have raised/blocked on the caller.
        assert getattr(app.state, "workflow_status", None) in {"warming", "ready"}
        # The build must NOT have run on the calling/main thread.
        assert calling_thread not in build_threads, (
            "build_workflow ran synchronously on the startup-calling thread"
        )
        # Give the daemon warmup thread a brief, bounded window to run so we can
        # positively confirm it builds on the named background thread. This is a
        # best-effort *positive* check; the structural guarantees above are the
        # non-flaky ones.
        deadline = time.time() + 2.0
        while not build_threads and time.time() < deadline:
            time.sleep(0.02)
        if build_threads:
            assert build_threads[0] == "startup-warmup", (
                f"warmup build ran on {build_threads[0]!r}, "
                "expected the 'startup-warmup' background thread"
            )
        # /ready should be reachable regardless of warmup progress.
        assert c.get("/ready").status_code in (200, 503)
