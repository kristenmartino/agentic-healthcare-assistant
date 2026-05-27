"""Tests for the FastAPI service.

We test the route shape + the helper functions, not the LangGraph workflow
itself (that's covered by the eval suites). The chat SSE endpoint is
smoke-tested for shape; the actual token streaming is exercised
end-to-end manually.
"""
from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import _content_to_text, _node_label, _sse, _summarize_partial, app  # noqa: E402


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
