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

def test_list_my_bookings_auto_scopes_to_active_patient(monkeypatch):
    """The agent prompt tells Claude not to pass patient_id (it's
    auto-scoped). The dispatcher must inject scope.patient_id into args
    when patient_chat has an active patient and the call came in empty —
    otherwise prompt and dispatcher contradict each other and 'show my
    bookings' fails."""
    seen: dict = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(agent_tools, "_tool_list_my_bookings", _capture)
    result = agent_tools.dispatch(
        "list_my_bookings", {},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert result == []
    assert seen.get("patient_id") == "fhir:active"


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


def test_list_my_bookings_by_patient_name_denied_for_other_patient(monkeypatch):
    """patient_name is an alternative to patient_id; it must be resolved
    in pre-auth and rejected when it points at a different patient."""
    monkeypatch.setattr(
        "tools.ehr.find_patient_by_name",
        lambda name, settings, actor=None: {"patient_id": "fhir:david-thompson",
                                            "name": "David Thompson"}
        if "david" in name.lower() else None,
    )
    result = agent_tools.dispatch(
        "list_my_bookings", {"patient_name": "David Thompson"},
        scope={"role": "patient_chat", "patient_id": "fhir:anjali-mehra"},
    )
    err = result.get("error") or ""
    assert "Not authorized" in err
    assert "fhir:david-thompson" in err


def test_list_my_bookings_by_patient_name_allowed_for_active_patient(monkeypatch):
    """patient_name pointing at the active patient is allowed."""
    monkeypatch.setattr(
        "tools.ehr.find_patient_by_name",
        lambda name, settings, actor=None: {"patient_id": "fhir:anjali-mehra",
                                            "name": "Anjali Mehra"},
    )
    monkeypatch.setattr(agent_tools, "_tool_list_my_bookings",
                        lambda **kwargs: [])
    result = agent_tools.dispatch(
        "list_my_bookings", {"patient_name": "Anjali Mehra"},
        scope={"role": "patient_chat", "patient_id": "fhir:anjali-mehra"},
    )
    assert result == []


def test_list_my_bookings_by_patient_name_denied_for_nonexistent_patient(monkeypatch):
    """Patient-chat can't probe for arbitrary names via list_my_bookings —
    nonexistent name returns a deny, not a silent empty list."""
    monkeypatch.setattr(
        "tools.ehr.find_patient_by_name",
        lambda name, settings, actor=None: None,
    )
    result = agent_tools.dispatch(
        "list_my_bookings", {"patient_name": "Nonexistent Person"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    err = result.get("error") or ""
    assert "Not authorized" in err
    assert "Nonexistent Person" in err


# ---------- get_audit_log: requires patient_id, must match active ----------

def test_get_audit_log_auto_scopes_to_active_patient(monkeypatch):
    """The agent prompt says patient-chat sees only their own events;
    'who accessed my records' should not require Claude to pass patient_id
    explicitly. Mirror the list_my_bookings auto-scope: when patient_chat
    has an active patient and the call came in empty, inject the
    active patient_id."""
    seen: dict = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return [{"id": 1}]

    monkeypatch.setattr(agent_tools, "_tool_get_audit_log", _capture)
    result = agent_tools.dispatch(
        "get_audit_log", {},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert result == [{"id": 1}]
    assert seen.get("patient_id") == "fhir:active"


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


# ---------- review-fix #11: post-resolution PHI checks ----------
#
# Direct response to the second review. Four real holes that the
# args-only authorization step left open:
#   - get_patient_history could read any patient by name
#   - cancel_booking could cancel any slot by id/confirmation
#   - upsert_patient could update arbitrary existing records
#   - no-active-patient scope was too permissive
#
# These tests stub the EHR lookups so the dispatcher's pre-resolution
# step (find_patient_by_name / list_all_bookings) sees deterministic
# data, and verify the denial happens BEFORE the tool function runs
# (no mutation can have leaked through).

def test_get_patient_history_denied_for_other_patient(monkeypatch):
    """patient_chat asking for a name that resolves to a different
    patient_id than the active one must be denied at dispatch."""
    from tools import ehr
    monkeypatch.setattr(
        ehr, "find_patient_by_name",
        lambda name, settings=None, actor=None: {"patient_id": "fhir:other", "name": name},
    )
    called = {"tool": 0}

    def _should_not_run(*args, **kwargs):
        called["tool"] += 1
        return {"oops": "tool ran despite denial"}

    monkeypatch.setattr(agent_tools, "_tool_get_patient_history", _should_not_run)
    result = agent_tools.dispatch(
        "get_patient_history", {"patient_name": "David Thompson"},
        scope={"role": "patient_chat", "patient_id": "fhir:anjali-mehra"},
    )
    assert "Not authorized" in (result.get("error") or "")
    assert called["tool"] == 0, "tool was invoked despite cross-patient denial"


def test_get_patient_history_allowed_for_active_patient(monkeypatch):
    """Name that resolves to the active patient_id IS allowed."""
    from tools import ehr
    monkeypatch.setattr(
        ehr, "find_patient_by_name",
        lambda name, settings=None, actor=None: {"patient_id": "fhir:anjali-mehra", "name": name},
    )
    monkeypatch.setattr(agent_tools, "_tool_get_patient_history",
                        lambda **kwargs: {"history_summary": "ok"})
    result = agent_tools.dispatch(
        "get_patient_history", {"patient_name": "Anjali Mehra"},
        scope={"role": "patient_chat", "patient_id": "fhir:anjali-mehra"},
    )
    assert result.get("history_summary") == "ok"


def test_get_patient_history_denied_when_no_active_patient(monkeypatch):
    monkeypatch.setattr(agent_tools, "_tool_get_patient_history",
                        lambda **kwargs: {"oops": True})
    result = agent_tools.dispatch(
        "get_patient_history", {"patient_name": "Anyone"},
        scope={"role": "patient_chat"},  # no patient_id
    )
    assert "Not authorized" in (result.get("error") or "")


def test_get_patient_history_denied_for_nonexistent_name_in_patient_chat(monkeypatch):
    """Patient-chat probing by name is a leak vector even on misses —
    the lack-of-match signal is information about who's a patient."""
    from tools import ehr
    monkeypatch.setattr(ehr, "find_patient_by_name",
                        lambda name, settings=None, actor=None: None)
    monkeypatch.setattr(agent_tools, "_tool_get_patient_history",
                        lambda **kwargs: {"oops": True})
    result = agent_tools.dispatch(
        "get_patient_history", {"patient_name": "FishingExpedition"},
        scope={"role": "patient_chat", "patient_id": "fhir:anjali-mehra"},
    )
    assert "Not authorized" in (result.get("error") or "")


def test_cancel_booking_denied_for_other_patient_slot(monkeypatch):
    """patient_chat cancel by slot_id where booked_by_patient_id != active
    must be denied BEFORE the cancel runs."""
    from tools import appointments
    monkeypatch.setattr(appointments, "list_all_bookings", lambda *a, **kw: [
        {"slot_id": 99, "booked_by_patient_id": "fhir:other",
         "confirmation_no": "AGS-OTHER"},
    ])
    called = {"tool": 0}

    def _should_not_run(*args, **kwargs):
        called["tool"] += 1
        return {"status": "cancelled"}

    monkeypatch.setattr(agent_tools, "_tool_cancel_booking", _should_not_run)
    result = agent_tools.dispatch(
        "cancel_booking", {"slot_id": 99},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert "Not authorized" in (result.get("error") or "")
    assert called["tool"] == 0


def test_cancel_booking_by_confirmation_also_checks_ownership(monkeypatch):
    """confirmation_no path must also resolve to ownership before
    cancelling — leaked confirmation numbers shouldn't be weaponizable."""
    from tools import appointments
    monkeypatch.setattr(appointments, "list_all_bookings", lambda *a, **kw: [
        {"slot_id": 99, "booked_by_patient_id": "fhir:other",
         "confirmation_no": "AGS-LEAKED"},
    ])
    monkeypatch.setattr(agent_tools, "_tool_cancel_booking",
                        lambda **kw: {"status": "cancelled"})
    result = agent_tools.dispatch(
        "cancel_booking", {"confirmation_no": "AGS-LEAKED"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert "Not authorized" in (result.get("error") or "")


def test_cancel_own_booking_allowed(monkeypatch):
    from tools import appointments
    monkeypatch.setattr(appointments, "list_all_bookings", lambda *a, **kw: [
        {"slot_id": 7, "booked_by_patient_id": "fhir:active",
         "confirmation_no": "AGS-OK"},
    ])
    monkeypatch.setattr(agent_tools, "_tool_cancel_booking",
                        lambda **kw: {"status": "cancelled", "slot_id": 7})
    result = agent_tools.dispatch(
        "cancel_booking", {"slot_id": 7},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert result.get("status") == "cancelled"


def test_cancel_booking_denied_when_no_active_patient(monkeypatch):
    """Walk-in scope can't cancel anything — the agent's session is not
    proof of identity."""
    monkeypatch.setattr(agent_tools, "_tool_cancel_booking",
                        lambda **kw: {"status": "cancelled"})
    result = agent_tools.dispatch(
        "cancel_booking", {"slot_id": 7},
        scope={"role": "patient_chat"},
    )
    assert "Not authorized" in (result.get("error") or "")


def test_cancel_booking_string_slot_id_checks_ownership(monkeypatch):
    """slot_id can arrive as a string from the LLM tool-call layer. The
    pre-auth comparison was using == against the integer booking row, so
    "99" != 99 silently let the tool run and cancel another patient's
    slot. Dispatch must normalize to int before the ownership check."""
    from tools import appointments
    monkeypatch.setattr(appointments, "list_all_bookings", lambda *a, **kw: [
        {"slot_id": 99, "booked_by_patient_id": "fhir:other",
         "confirmation_no": "AGS-X"},
    ])
    called = {"tool": 0}

    def _should_not_run(**kw):
        called["tool"] += 1
        return {"status": "cancelled"}

    monkeypatch.setattr(agent_tools, "_tool_cancel_booking", _should_not_run)
    result = agent_tools.dispatch(
        "cancel_booking", {"slot_id": "99"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    err = result.get("error") or ""
    assert "Not authorized" in err
    assert called["tool"] == 0, (
        "string slot_id bypassed ownership check — tool ran on another "
        "patient's slot")


def test_cancel_booking_invalid_slot_id_denied(monkeypatch):
    """A non-integer slot_id should deny cleanly, not raise and not run."""
    called = {"tool": 0}

    def _should_not_run(**kw):
        called["tool"] += 1
        return {"status": "cancelled"}

    monkeypatch.setattr(agent_tools, "_tool_cancel_booking", _should_not_run)
    result = agent_tools.dispatch(
        "cancel_booking", {"slot_id": "abc"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    err = result.get("error") or ""
    assert "Not authorized" in err
    assert "slot_id" in err
    assert called["tool"] == 0


def test_upsert_patient_denied_when_updating_other_patient(monkeypatch):
    """An existing record for someone else cannot be modified through
    patient_chat, even if the agent claims it's their record."""
    from tools import ehr
    monkeypatch.setattr(
        ehr, "find_patient_by_name",
        lambda name, settings=None, actor=None: {"patient_id": "fhir:other", "name": name},
    )
    called = {"tool": 0}

    def _should_not_run(**kwargs):
        called["tool"] += 1
        return {"operation": "update"}

    monkeypatch.setattr(agent_tools, "_tool_upsert_patient", _should_not_run)
    result = agent_tools.dispatch(
        "upsert_patient", {"name": "David Thompson", "age": 60},
        scope={"role": "patient_chat", "patient_id": "fhir:anjali-mehra"},
    )
    assert "Not authorized" in (result.get("error") or "")
    assert called["tool"] == 0, "upsert ran despite cross-patient update attempt"


def test_upsert_new_patient_allowed_when_no_existing(monkeypatch):
    """New patient name (no existing record): allowed — registration flow."""
    from tools import ehr
    monkeypatch.setattr(ehr, "find_patient_by_name",
                        lambda name, settings=None, actor=None: None)
    monkeypatch.setattr(agent_tools, "_tool_upsert_patient", lambda **kw: {
        "operation": "insert", "patient_id": "fhir:active",
    })
    result = agent_tools.dispatch(
        "upsert_patient", {"name": "Brand New", "age": 30},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert result.get("operation") == "insert"


def test_walk_in_upsert_existing_patient_denied_before_tool_runs(monkeypatch):
    """Walk-in patient_chat (no active patient) must not be able to update
    an existing record. The dispatcher should deny BEFORE the tool runs —
    otherwise the underlying EHR layer happily updates the existing row
    when a walk-in name collides with a known patient."""
    from tools import ehr
    monkeypatch.setattr(
        ehr, "find_patient_by_name",
        lambda name, settings=None, actor=None: {
            "patient_id": "fhir:david-thompson", "name": "David Thompson"},
    )
    called = {"tool": 0}

    def _should_not_run(**kwargs):
        called["tool"] += 1
        return {"operation": "update"}

    monkeypatch.setattr(agent_tools, "_tool_upsert_patient", _should_not_run)
    result = agent_tools.dispatch(
        "upsert_patient", {"name": "David Thompson", "summary": "drive-by note"},
        scope={"role": "patient_chat"},  # no active patient
    )
    err = result.get("error") or ""
    assert "Not authorized" in err
    assert "walk-in" in err.lower() or "fhir:david-thompson" in err
    assert called["tool"] == 0, (
        "walk-in upsert ran against existing record — pre-auth bypassed")


def test_upsert_own_record_allowed(monkeypatch):
    from tools import ehr
    monkeypatch.setattr(
        ehr, "find_patient_by_name",
        lambda name, settings=None, actor=None: {"patient_id": "fhir:active",
                                                  "name": "Active Patient"},
    )
    monkeypatch.setattr(agent_tools, "_tool_upsert_patient", lambda **kw: {
        "operation": "update", "patient_id": "fhir:active",
    })
    result = agent_tools.dispatch(
        "upsert_patient", {"name": "Active Patient", "summary": "new note"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert result.get("operation") == "update"


def test_list_my_bookings_denied_when_no_active_patient(monkeypatch):
    monkeypatch.setattr(agent_tools, "_tool_list_my_bookings", lambda **kw: [])
    result = agent_tools.dispatch(
        "list_my_bookings", {"patient_id": "fhir:anyone"},
        scope={"role": "patient_chat"},
    )
    assert "Not authorized" in (result.get("error") or "")


def test_get_audit_log_denied_when_no_active_patient(monkeypatch):
    monkeypatch.setattr(agent_tools, "_tool_get_audit_log", lambda **kw: [])
    result = agent_tools.dispatch(
        "get_audit_log", {"patient_id": "fhir:anyone"},
        scope={"role": "patient_chat"},
    )
    assert "Not authorized" in (result.get("error") or "")


def test_find_patient_denied_when_no_active_patient(monkeypatch):
    """find_patient is itself a probe — walk-in scope can't fish."""
    monkeypatch.setattr(agent_tools, "_tool_find_patient",
                        lambda name: {"patient_id": "fhir:x"})
    result = agent_tools.dispatch(
        "find_patient", {"name": "Anyone"},
        scope={"role": "patient_chat"},
    )
    assert "Not authorized" in (result.get("error") or "")


def test_walk_in_can_still_book_a_doctor(monkeypatch):
    """The blanket no-active-patient policy must not break walk-in booking.
    book_appointment is not in the PHI-reading set."""
    monkeypatch.setattr(agent_tools, "_tool_book_appointment",
                        lambda **kw: {"confirmation_no": "AGS-NEW",
                                       "slot_id": 1})
    result = agent_tools.dispatch(
        "book_appointment", {"patient_name": "Walk-In Person",
                              "specialty": "general_practice"},
        scope={"role": "patient_chat"},
    )
    assert result.get("confirmation_no") == "AGS-NEW"


def test_walk_in_can_still_search_medical_info(monkeypatch):
    monkeypatch.setattr(agent_tools, "_tool_medical_search",
                        lambda **kw: [{"title": "x"}])
    result = agent_tools.dispatch(
        "medical_search", {"query": "flu"},
        scope={"role": "patient_chat"},
    )
    assert result == [{"title": "x"}]


# ---------- fail-closed on EHR lookup failures ----------
#
# _pre_authorize_resolutions used to swallow find_patient_by_name
# exceptions and treat them as "no existing record" → allow. That broke
# the fail-closed contract when the EHR layer was flaky: we'd let
# mutations/reads through without verifying ownership. These tests lock
# in that lookup failures now produce a denial, not a silent allow.


def _raise_ehr_failure(*args, **kwargs):
    raise RuntimeError("EHR connection refused (simulated)")


def test_get_patient_history_denied_when_pre_auth_lookup_fails(monkeypatch):
    from tools import ehr
    monkeypatch.setattr(ehr, "find_patient_by_name", _raise_ehr_failure)
    called = {"tool": 0}

    def _should_not_run(**kw):
        called["tool"] += 1
        return {"record": {}}

    monkeypatch.setattr(agent_tools, "_tool_get_patient_history",
                        _should_not_run)
    result = agent_tools.dispatch(
        "get_patient_history", {"patient_name": "Anjali Mehra"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert "Not authorized" in (result.get("error") or "")
    assert called["tool"] == 0


def test_find_patient_denied_when_pre_auth_lookup_fails(monkeypatch):
    from tools import ehr
    monkeypatch.setattr(ehr, "find_patient_by_name", _raise_ehr_failure)
    called = {"tool": 0}

    def _should_not_run(**kw):
        called["tool"] += 1
        return {"patient_id": "fhir:x"}

    monkeypatch.setattr(agent_tools, "_tool_find_patient", _should_not_run)
    result = agent_tools.dispatch(
        "find_patient", {"name": "Anjali Mehra"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert "Not authorized" in (result.get("error") or "")
    assert called["tool"] == 0


def test_list_my_bookings_denied_when_patient_name_lookup_fails(monkeypatch):
    from tools import ehr
    monkeypatch.setattr(ehr, "find_patient_by_name", _raise_ehr_failure)
    called = {"tool": 0}

    def _should_not_run(**kw):
        called["tool"] += 1
        return []

    monkeypatch.setattr(agent_tools, "_tool_list_my_bookings", _should_not_run)
    result = agent_tools.dispatch(
        "list_my_bookings", {"patient_name": "Anjali Mehra"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert "Not authorized" in (result.get("error") or "")
    assert called["tool"] == 0


def test_upsert_patient_denied_when_pre_auth_lookup_fails(monkeypatch):
    """If we can't verify ownership, refuse to mutate. The old behavior
    swallowed the exception, treated the patient as "not existing", and
    happily ran upsert — turning a flaky EHR layer into a silent
    cross-patient write window."""
    from tools import ehr
    monkeypatch.setattr(ehr, "find_patient_by_name", _raise_ehr_failure)
    called = {"tool": 0}

    def _should_not_run(**kw):
        called["tool"] += 1
        return {"operation": "insert"}

    monkeypatch.setattr(agent_tools, "_tool_upsert_patient", _should_not_run)
    result = agent_tools.dispatch(
        "upsert_patient", {"name": "Some Patient", "summary": "note"},
        scope={"role": "patient_chat", "patient_id": "fhir:active"},
    )
    assert "Not authorized" in (result.get("error") or "")
    assert called["tool"] == 0
