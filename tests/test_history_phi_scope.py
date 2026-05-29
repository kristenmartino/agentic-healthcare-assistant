"""PHI-scope guard tests for the legacy graph history node.

The reported bug: a walk-in session (no authenticated active patient) asked
"Show me Anjali Mehra's medical history" and the assistant served it. The
classifier extracts the name from the query, and the history node used to
look it up and summarize it for anyone — the same cross-patient leak the
agent dispatcher (nodes/agent_tools.py) already blocks.

These lock in the fix: in patient_chat role a walk-in gets no record reads at
all, an authenticated patient can only read their OWN record, and trusted
roles (clinician/admin, and the agent loop's post-authorization delegation)
stay unrestricted.
"""
from __future__ import annotations

import pytest

from nodes import history


@pytest.fixture(autouse=True)
def _stub_history_io(monkeypatch):
    """Neutralize the node's side-effecting collaborators so each test only
    exercises the authorization branch.

    - log_access: capture denied/allowed audit events (no DB write).
    - get_patient_clinical_context / search_index: return empty so the
      allowed path doesn't load the embedding model or hit FHIR.
    - chat: deterministic summary so the allowed path makes no real LLM call.
    """
    audit: list[dict] = []
    monkeypatch.setattr(
        history, "log_access",
        lambda *a, **kw: audit.append({"args": a, "kwargs": kw}),
    )
    monkeypatch.setattr(history, "get_patient_clinical_context",
                        lambda pid, settings=None, actor=None:
                            {"conditions": [], "observations": []})
    monkeypatch.setattr(history, "search_index",
                        lambda *a, **kw: [])
    monkeypatch.setattr(history, "chat", lambda **kw: "STUB SUMMARY")
    return audit


def _spy_find(monkeypatch, record):
    """Install a find_patient_by_name spy that returns `record` and counts calls."""
    calls = {"n": 0}

    def _find(name, settings=None, actor=None):
        calls["n"] += 1
        return record(name) if callable(record) else record

    monkeypatch.setattr(history, "find_patient_by_name", _find)
    return calls


# ---------- walk-in: no record reads at all ----------

def test_walk_in_denied_and_never_looks_up(monkeypatch, _stub_history_io):
    """No active patient → refuse before any EHR lookup (existence must not
    leak) and produce no patient data."""
    calls = _spy_find(monkeypatch, {"patient_id": "fhir:anjali-mehra",
                                    "name": "Anjali Mehra"})

    out = history.history_node({
        "user_input": "Show me Anjali Mehra's medical history",
        "patient_name": "Anjali Mehra",  # extracted by the classifier
        # no patient_id, no role → walk-in patient_chat
    })

    assert calls["n"] == 0, "walk-in read probed the EHR despite denial"
    summary = out["history_summary"]
    assert "Anjali" not in summary
    assert "STUB SUMMARY" not in summary
    assert out["tool_log"][0]["result"] == "denied"
    assert out["tool_log"][0]["reason"] == "walk_in_no_active_patient"
    # Denied access was audited.
    assert any(ev["kwargs"].get("outcome") == "denied" for ev in _stub_history_io)


# ---------- authenticated patient: own record only ----------

def test_authenticated_patient_reads_own_record(monkeypatch, _stub_history_io):
    _spy_find(monkeypatch, {"patient_id": "fhir:anjali-mehra",
                            "name": "Anjali Mehra", "age": 40})

    out = history.history_node({
        "user_input": "Show me my medical history",
        "patient_name": "Anjali Mehra",
        "patient_id": "fhir:anjali-mehra",
    })

    assert out["history_summary"] == "STUB SUMMARY"
    assert out["tool_log"][0].get("record_found") is True


def test_authenticated_patient_denied_for_other_patient(monkeypatch, _stub_history_io):
    """Name resolves to a different patient_id than the active context → deny."""
    _spy_find(monkeypatch, {"patient_id": "fhir:david-thompson",
                            "name": "David Thompson"})

    out = history.history_node({
        "user_input": "Now show me everything about David Thompson",
        "patient_name": "David Thompson",
        "patient_id": "fhir:anjali-mehra",  # authenticated as Anjali
    })

    assert "STUB SUMMARY" not in out["history_summary"]
    assert "David" not in out["history_summary"]
    assert out["tool_log"][0]["result"] == "denied"
    assert out["tool_log"][0]["reason"] == "cross_patient"


def test_authenticated_patient_denied_for_unknown_name(monkeypatch, _stub_history_io):
    """A name that resolves to nothing is also a probe → deny, don't 404-leak."""
    _spy_find(monkeypatch, None)

    out = history.history_node({
        "user_input": "Show me Nonexistent Person's history",
        "patient_name": "Nonexistent Person",
        "patient_id": "fhir:anjali-mehra",
    })

    assert out["tool_log"][0]["result"] == "denied"
    assert out["tool_log"][0]["reason"] == "cross_patient"


# ---------- trusted roles stay unrestricted ----------

def test_clinician_reads_any_patient_without_active_context(monkeypatch, _stub_history_io):
    _spy_find(monkeypatch, {"patient_id": "fhir:david-thompson",
                            "name": "David Thompson"})

    out = history.history_node({
        "user_input": "history for David Thompson",
        "patient_name": "David Thompson",
        "role": "clinician",
        # no patient_id — clinicians aren't scoped to one active patient
    })

    assert out["history_summary"] == "STUB SUMMARY"


def test_admin_role_unrestricted(monkeypatch, _stub_history_io):
    _spy_find(monkeypatch, {"patient_id": "fhir:david-thompson",
                            "name": "David Thompson"})

    out = history.history_node({
        "user_input": "history for David Thompson",
        "patient_name": "David Thompson",
        "role": "admin",
    })

    assert out["history_summary"] == "STUB SUMMARY"


def test_agent_delegation_passes_resolved_pid(monkeypatch, _stub_history_io):
    """The agent tool delegates to history_node with the resolved patient_id
    as the active context (it has already authorized the read). That call
    must pass the node's own guard."""
    from nodes import agent_tools
    from tools import ehr

    monkeypatch.setattr(
        ehr, "find_patient_by_name",
        lambda name, settings=None, actor=None: {"patient_id": "fhir:anjali-mehra",
                                                 "name": name},
    )
    monkeypatch.setattr(
        ehr, "get_patient_clinical_context",
        lambda pid, settings=None, actor=None: {"conditions": [], "observations": []},
    )
    # history_node uses its module-level find_patient_by_name; point it at the
    # active patient so the guard sees a matching pid.
    _spy_find(monkeypatch, {"patient_id": "fhir:anjali-mehra",
                            "name": "Anjali Mehra"})

    result = agent_tools._tool_get_patient_history("Anjali Mehra")
    assert result["history_summary"] == "STUB SUMMARY"
