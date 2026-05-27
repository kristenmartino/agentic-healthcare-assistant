"""Adversarial tests for the PHI scope guardrails on the agent dispatcher.

Direct response to the PR #6 review. The LLM picks the tool, but the
dispatcher decides whether the (tool, args, scope) combination is allowed
to execute. Fail-closed: a missing or unknown role is treated as
patient_chat (the most restrictive).

We test BOTH that legitimate calls still go through AND that the
patient-facing chat role can't:
  - enumerate the patient table
  - read other patients' bookings, audit, or doctor-schedule identifiers
  - read its own audit log unfiltered
"""
from __future__ import annotations

from nodes import agent_tools

# ---------- list_patients: blocked for patient_chat ----------

def test_list_patients_denied_for_patient_chat():
    result = agent_tools.dispatch(
        "list_patients", {}, scope={"role": "patient_chat"},
    )
    assert "Not authorized" in (result.get("error") or "")


def test_list_patients_allowed_for_clinician(monkeypatch):
    fake = [{"patient_id": "fhir:x", "name": "X"}]
    monkeypatch.setattr(agent_tools, "_tool_list_patients", lambda: fake)
    result = agent_tools.dispatch(
        "list_patients", {}, scope={"role": "clinician"},
    )
    assert result == fake


def test_list_patients_allowed_for_admin(monkeypatch):
    monkeypatch.setattr(agent_tools, "_tool_list_patients", lambda: [])
    result = agent_tools.dispatch(
        "list_patients", {}, scope={"role": "admin"},
    )
    assert isinstance(result, list)


def test_missing_scope_treated_as_patient_chat():
    """Fail-closed: dispatch(...) with no scope should refuse list_patients."""
    result = agent_tools.dispatch("list_patients", {}, scope=None)
    assert "Not authorized" in (result.get("error") or "")


# ---------- list_my_bookings: requires patient_id filter ----------

def test_unfiltered_list_my_bookings_denied_for_patient_chat():
    result = agent_tools.dispatch(
        "list_my_bookings", {},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert "Not authorized" in (result.get("error") or "")


def test_list_my_bookings_with_active_patient_id_allowed(monkeypatch):
    monkeypatch.setattr(agent_tools, "_tool_list_my_bookings",
                        lambda **kwargs: [])
    result = agent_tools.dispatch(
        "list_my_bookings", {"patient_id": "fhir:active"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert result == []


def test_list_my_bookings_for_other_patient_denied():
    """Even with a patient_id filter, can't query someone else's bookings."""
    result = agent_tools.dispatch(
        "list_my_bookings", {"patient_id": "fhir:someone-else"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    err = result.get("error") or ""
    assert "another patient" in err or "Not authorized" in err


# ---------- get_audit_log: requires patient_id, must match active ----------

def test_unfiltered_audit_log_denied_for_patient_chat():
    result = agent_tools.dispatch(
        "get_audit_log", {},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert "Not authorized" in (result.get("error") or "")


def test_audit_log_for_other_patient_denied():
    result = agent_tools.dispatch(
        "get_audit_log", {"patient_id": "fhir:someone-else"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    err = result.get("error") or ""
    assert "another patient" in err or "Not authorized" in err


def test_audit_log_for_self_allowed(monkeypatch):
    monkeypatch.setattr(agent_tools, "_tool_get_audit_log", lambda **kwargs: [])
    result = agent_tools.dispatch(
        "get_audit_log", {"patient_id": "fhir:active"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert result == []


def test_admin_can_read_unfiltered_audit(monkeypatch):
    monkeypatch.setattr(agent_tools, "_tool_get_audit_log",
                        lambda **kwargs: [{"id": 1}])
    result = agent_tools.dispatch(
        "get_audit_log", {},
        scope={"role": "admin"},
    )
    assert result == [{"id": 1}]


# ---------- get_doctor_schedule: mask other patients' identifiers ----------

def test_doctor_schedule_masks_other_patient_identifiers(monkeypatch):
    fake = {
        "doctor": {"doctor_id": "DOC1", "name": "Dr. X"},
        "schedule": [
            {"slot_id": 1, "start_time": "2026-06-01T09:00:00", "booked": 0,
             "booked_by_patient_id": None, "confirmation_no": None},
            {"slot_id": 2, "start_time": "2026-06-01T09:30:00", "booked": 1,
             "booked_by_patient_id": "fhir:other",
             "confirmation_no": "AGS-OTHER"},
            {"slot_id": 3, "start_time": "2026-06-01T10:00:00", "booked": 1,
             "booked_by_patient_id": "fhir:active",
             "confirmation_no": "AGS-ME"},
        ],
    }
    monkeypatch.setattr(agent_tools, "_tool_get_doctor_schedule",
                        lambda **kwargs: fake)

    result = agent_tools.dispatch(
        "get_doctor_schedule", {"doctor_name": "X"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    schedule = result["schedule"]
    # Open slot — unchanged.
    assert schedule[0]["booked_by_patient_id"] is None
    # Other patient's slot — masked.
    assert schedule[1]["booked_by_patient_id"] == "(other patient — masked)"
    assert schedule[1]["confirmation_no"] == "(masked)"
    # Active patient's own slot — visible.
    assert schedule[2]["booked_by_patient_id"] == "fhir:active"
    assert schedule[2]["confirmation_no"] == "AGS-ME"


def test_doctor_schedule_no_masking_for_clinician(monkeypatch):
    """Clinicians legitimately need the booker info to operate the calendar."""
    fake = {
        "doctor": {"doctor_id": "DOC1"},
        "schedule": [
            {"slot_id": 1, "booked": 1,
             "booked_by_patient_id": "fhir:other",
             "confirmation_no": "AGS-X"},
        ],
    }
    monkeypatch.setattr(agent_tools, "_tool_get_doctor_schedule",
                        lambda **kwargs: fake)

    result = agent_tools.dispatch(
        "get_doctor_schedule", {"doctor_name": "X"},
        scope={"role": "clinician"},
    )
    assert result["schedule"][0]["booked_by_patient_id"] == "fhir:other"
    assert result["schedule"][0]["confirmation_no"] == "AGS-X"


# ---------- booking flows still work in patient_chat ----------

def test_book_appointment_still_works_in_patient_chat(monkeypatch):
    monkeypatch.setattr(agent_tools, "_tool_book_appointment",
                        lambda **kwargs: {"confirmation_no": "AGS-OK"})
    result = agent_tools.dispatch(
        "book_appointment",
        {"patient_name": "Test", "specialty": "cardiology"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert result.get("confirmation_no") == "AGS-OK"


def test_medical_search_still_works_in_patient_chat(monkeypatch):
    monkeypatch.setattr(agent_tools, "_tool_medical_search",
                        lambda **kwargs: [{"title": "x"}])
    result = agent_tools.dispatch(
        "medical_search", {"query": "pneumonia"},
        scope={"role": "patient_chat"},
    )
    assert result == [{"title": "x"}]


# ---------- agent_loop integration: scope from state ----------

def test_agent_loop_passes_scope_into_dispatch(monkeypatch):
    """The loop must build scope from state.role + state.patient_id and
    pass it to dispatch — otherwise the policy guardrails are bypassed."""
    from nodes.agent_loop import agent_loop_node
    from tests.test_agent_loop import (
        _ai_text,
        _ai_tool_call,
        _patch_client_with,
        _ScriptedClient,
    )

    captured_scope: dict = {}
    real_dispatch = agent_tools.dispatch

    def _spy(name, args, scope=None):
        captured_scope.update(scope or {})
        return real_dispatch(name, args, scope=scope)

    monkeypatch.setattr("nodes.agent_loop.dispatch", _spy)
    monkeypatch.setattr(agent_tools, "_tool_find_patient",
                        lambda name: {"patient_id": "fhir:active", "name": name})

    client = _ScriptedClient([
        _ai_tool_call("find_patient", {"name": "Active Patient"}, "c1"),
        _ai_text("Found."),
    ])
    with _patch_client_with(client):
        agent_loop_node({
            "user_input": "find me",
            "patient_id": "fhir:active",
            "patient_name": "Active Patient",
        })
    assert captured_scope.get("role") == "patient_chat"
    assert captured_scope.get("patient_id") == "fhir:active"
    assert captured_scope.get("actor") == "agent"


def test_agent_loop_with_clinician_role_unlocks_list_patients(monkeypatch):
    """If state.role='clinician', list_patients should reach the function."""
    from nodes.agent_loop import agent_loop_node
    from tests.test_agent_loop import (
        _ai_text,
        _ai_tool_call,
        _patch_client_with,
        _ScriptedClient,
    )

    called = {"n": 0}

    def _impl():
        called["n"] += 1
        return []

    monkeypatch.setattr(agent_tools, "_tool_list_patients", _impl)
    client = _ScriptedClient([
        _ai_tool_call("list_patients", {}, "c1"),
        _ai_text("Here."),
    ])
    with _patch_client_with(client):
        agent_loop_node({
            "user_input": "list all patients",
            "role": "clinician",
        })
    assert called["n"] == 1
