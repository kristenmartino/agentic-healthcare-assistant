"""Tool definitions for the agent-loop node.

Mirror of the MCP server's tool surface, formatted as Anthropic tool
schemas. Each entry pairs:
  - the JSON schema sent to Claude (`name`, `description`, `input_schema`)
  - the Python callable that actually executes

When Claude returns a `tool_use` block, the loop looks the tool up by name
and dispatches with the parsed args. Every tool returns JSON-serialisable
data — strings, dicts, or lists.

Why not just call the MCP tools? The MCP server's tools are designed for
external clients (no `actor` plumbing, no Settings injection). The agent
runs inside our process with full Settings access and should tag every
call with `actor="agent"` so the audit log distinguishes agent-driven
access from direct UI traffic.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


# ---------- tool implementations ----------

def _tool_book_appointment(
    patient_name: str,
    specialty: str,
    preferred_date: str | None = None,
) -> dict:
    """Book the earliest available slot."""
    from config import load_settings
    from tools.appointments import book_appointment
    from tools.audit import log_access
    from tools.ehr import find_patient_by_name

    s = load_settings()
    patient = find_patient_by_name(patient_name, s, actor="agent")
    patient_id = patient["patient_id"] if patient else f"walkin-{abs(hash(patient_name)) % 10**8:08d}"
    try:
        appt = book_appointment(
            s.appointments_db_path,
            patient_id=patient_id,
            patient_name=patient_name,
            specialty=specialty,
            preferred_date=preferred_date,
        )
    except ValueError as exc:
        log_access("agent", "appointment.book", "Appointment", None,
                   patient_id=patient_id, outcome="error",
                   details={"specialty": specialty, "error": str(exc)})
        return {"error": str(exc), "specialty": specialty}
    log_access("agent", "appointment.book", "Appointment",
               str(appt.get("slot_id")), patient_id=patient_id,
               details={"specialty": specialty,
                        "doctor_name": appt.get("doctor_name"),
                        "confirmation_no": appt.get("confirmation_no")})
    return appt


def _tool_cancel_booking(slot_id: int | None = None, confirmation_no: str | None = None) -> dict:
    """Cancel a booking by slot_id or by confirmation number."""
    from config import load_settings
    from tools.appointments import cancel_booking, list_all_bookings
    from tools.audit import log_access

    s = load_settings()
    if confirmation_no and not slot_id:
        for b in list_all_bookings(s.appointments_db_path):
            if b.get("confirmation_no") == confirmation_no:
                slot_id = int(b["slot_id"])
                break
    if not slot_id:
        return {"status": "not_found", "reason": "provide slot_id or confirmation_no"}
    result = cancel_booking(s.appointments_db_path, int(slot_id))
    log_access("agent", "appointment.cancel", "Appointment", str(slot_id),
               details={"status": result.get("status")})
    return result


def _tool_list_doctors(specialty: str | None = None) -> list[dict]:
    """List doctors, optionally filtered by specialty."""
    from config import load_settings
    from tools.appointments import get_doctors_for_dashboard, list_doctors_for_specialty
    s = load_settings()
    return (list_doctors_for_specialty(s.appointments_db_path, specialty)
            if specialty else get_doctors_for_dashboard(s.appointments_db_path))


def _tool_get_doctor_schedule(doctor_name: str, days_ahead: int = 7) -> dict:
    """Return upcoming slots for a doctor — booked and available."""
    from config import load_settings
    from tools.appointments import get_doctor_schedule, get_doctors_for_dashboard
    from tools.audit import log_access

    s = load_settings()
    needle = doctor_name.lower().strip()
    doctors = get_doctors_for_dashboard(s.appointments_db_path)
    match = next((d for d in doctors if needle in d["name"].lower()), None)
    if not match:
        return {"error": f"No doctor matching '{doctor_name}'",
                "available_names": [d["name"] for d in doctors[:10]]}
    schedule = get_doctor_schedule(s.appointments_db_path, match["doctor_id"],
                                   days_ahead=days_ahead)
    log_access("agent", "appointment.schedule_query", "Doctor", match["doctor_id"],
               details={"doctor_name": match["name"], "days_ahead": days_ahead,
                        "slots_returned": len(schedule)})
    return {"doctor": match, "schedule": schedule}


def _tool_list_my_bookings(patient_id: str | None = None,
                           patient_name: str | None = None,
                           upcoming_only: bool = True) -> list[dict]:
    """List a patient's bookings (or all bookings if no patient specified)."""
    from config import load_settings
    from tools.appointments import list_all_bookings
    from tools.audit import log_access
    from tools.ehr import find_patient_by_name

    s = load_settings()
    if patient_name and not patient_id:
        rec = find_patient_by_name(patient_name, s, actor="agent")
        patient_id = rec.get("patient_id") if rec else None
    bookings = list_all_bookings(s.appointments_db_path, upcoming_only=upcoming_only)
    if patient_id:
        bookings = [b for b in bookings if b.get("booked_by_patient_id") == patient_id]
    log_access("agent", "appointment.list", "Appointment", None,
               patient_id=patient_id,
               details={"upcoming_only": upcoming_only, "returned": len(bookings)})
    return bookings


def _tool_find_patient(name: str) -> dict | None:
    from config import load_settings
    from tools.ehr import find_patient_by_name
    return find_patient_by_name(name, load_settings(), actor="agent")


def _tool_list_patients() -> list[dict]:
    from config import load_settings
    from tools.ehr import list_patients
    return list_patients(load_settings(), actor="agent")


def _tool_get_patient_history(patient_name: str) -> dict:
    """Returns the structured record + FHIR Conditions + recent Observations."""
    from config import load_settings
    from tools.ehr import find_patient_by_name, get_patient_clinical_context
    s = load_settings()
    record = find_patient_by_name(patient_name, s, actor="agent")
    if not record:
        return {"error": f"No patient matching '{patient_name}'"}
    clinical = get_patient_clinical_context(record["patient_id"], s, actor="agent")
    return {"record": record, "conditions": clinical.get("conditions", []),
            "observations": clinical.get("observations", [])}


def _tool_upsert_patient(name: str, age: int | None = None,
                          gender: str | None = None,
                          phone: str | None = None,
                          email: str | None = None,
                          summary: str | None = None) -> dict:
    from config import load_settings
    from tools.ehr import add_or_update_patient
    return add_or_update_patient({
        "name": name, "age": age, "gender": gender,
        "phone_raw": phone, "email": email, "summary": summary,
    }, load_settings(), actor="agent")


def _tool_medical_search(query: str, top_k: int = 4) -> list[dict]:
    """Search trusted medical sources (MedlinePlus, WHO, CDC, NIH, Mayo)."""
    from config import load_settings
    from tools.audit import log_access
    from tools.medical_search import medical_search
    s = load_settings()
    results = medical_search(query, top_k=top_k, tavily_api_key=s.tavily_api_key)
    log_access("agent", "medical_search.query", "WebSearch", None,
               details={"query": query, "results": len(results)})
    return results


def _tool_get_audit_log(patient_id: str | None = None,
                        action_prefix: str | None = None,
                        limit: int = 25) -> list[dict]:
    """Read the PHI audit log. Patients can ask 'who accessed my records?'."""
    from tools.audit import log_access, query_audit
    events = query_audit(patient_id=patient_id, action_prefix=action_prefix,
                         limit=max(1, min(int(limit), 200)))
    log_access("agent", "audit.read", "AuditLog", None,
               details={"returned": len(events), "limit": limit,
                        "filter_patient": patient_id})
    return events


# ---------- schemas ----------
#
# JSON schemas Claude reads to decide which tool to call. Descriptions are
# the LLM's only signal — make them concrete + state when to use vs. when
# NOT to use, since Claude follows that guidance reliably.

TOOL_SPECS: list[dict] = [
    {
        "name": "book_appointment",
        "description": ("Book the earliest available appointment slot for a "
                        "given medical specialty. Use when the user wants to "
                        "schedule, book, or set up a new visit. Returns the "
                        "doctor, time, and confirmation number."),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string",
                                 "description": "Patient's full name; pass 'Walk-in Patient' if unknown."},
                "specialty": {"type": "string",
                              "description": "One of: general_practice, cardiology, endocrinology, "
                                             "nephrology, neurology, pulmonology, oncology, "
                                             "psychiatry, dermatology."},
                "preferred_date": {"type": "string",
                                   "description": "Optional ISO date (YYYY-MM-DD); booking will be on or after."},
            },
            "required": ["patient_name", "specialty"],
        },
    },
    {
        "name": "cancel_booking",
        "description": ("Cancel an existing appointment. Provide EITHER slot_id "
                        "or confirmation_no. Idempotent."),
        "input_schema": {
            "type": "object",
            "properties": {
                "slot_id": {"type": "integer"},
                "confirmation_no": {"type": "string",
                                    "description": "Like 'AGS-123456'."},
            },
        },
    },
    {
        "name": "list_doctors",
        "description": "List doctors in the system. Filter by specialty if given.",
        "input_schema": {
            "type": "object",
            "properties": {
                "specialty": {"type": "string"},
            },
        },
    },
    {
        "name": "get_doctor_schedule",
        "description": ("Return a doctor's upcoming slots — both booked and "
                        "available. Use when the user asks 'what's on Dr. X's "
                        "calendar' or 'when is Dr. X next available'. The "
                        "match is case-insensitive substring; full or partial "
                        "name works ('Nair', 'Dr. Priya Nair')."),
        "input_schema": {
            "type": "object",
            "properties": {
                "doctor_name": {"type": "string"},
                "days_ahead": {"type": "integer",
                               "description": "How many days forward to include. Default 7."},
            },
            "required": ["doctor_name"],
        },
    },
    {
        "name": "list_my_bookings",
        "description": ("List appointments — optionally filtered to one "
                        "patient. Use when the user asks 'what are my "
                        "upcoming appointments', 'show recent bookings'."),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "patient_name": {"type": "string",
                                 "description": "Alternative to patient_id."},
                "upcoming_only": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "find_patient",
        "description": "Look up one patient by name. Returns null if not found.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "list_patients",
        "description": ("List all patients in the system. Use sparingly "
                        "(every PHI dump is audited)."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_patient_history",
        "description": ("Retrieve a patient's structured record PLUS FHIR "
                        "Conditions (SNOMED-coded) and recent Observations "
                        "(LOINC-coded). Use when the user asks for medical "
                        "history, past visits, current conditions, lab "
                        "values."),
        "input_schema": {
            "type": "object",
            "properties": {"patient_name": {"type": "string"}},
            "required": ["patient_name"],
        },
    },
    {
        "name": "upsert_patient",
        "description": ("Add a new patient or update an existing one. "
                        "Idempotent on (name, age) → patient_id."),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "gender": {"type": "string"},
                "phone": {"type": "string"},
                "email": {"type": "string"},
                "summary": {"type": "string",
                            "description": "Brief clinical note, e.g. 'hypertension'."},
            },
            "required": ["name"],
        },
    },
    {
        "name": "medical_search",
        "description": ("Search trusted medical sources (MedlinePlus, WHO, "
                        "CDC, NIH, Mayo Clinic, NHS) for general medical "
                        "information. Use for 'what are the symptoms of X', "
                        "'latest treatment for Y'. Returns title + snippet + "
                        "URL per result."),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 4},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_audit_log",
        "description": ("Read the PHI access audit log. Use when the user "
                        "asks 'who looked at my records', 'show me recent "
                        "access events', or for compliance review."),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string",
                               "description": "Filter to one patient."},
                "action_prefix": {"type": "string",
                                  "description": "e.g. 'ehr.', 'appointment.'."},
                "limit": {"type": "integer", "default": 25},
            },
        },
    },
]


# Map name → callable for the dispatcher.
TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "book_appointment": _tool_book_appointment,
    "cancel_booking": _tool_cancel_booking,
    "list_doctors": _tool_list_doctors,
    "get_doctor_schedule": _tool_get_doctor_schedule,
    "list_my_bookings": _tool_list_my_bookings,
    "find_patient": _tool_find_patient,
    "list_patients": _tool_list_patients,
    "get_patient_history": _tool_get_patient_history,
    "upsert_patient": _tool_upsert_patient,
    "medical_search": _tool_medical_search,
    "get_audit_log": _tool_get_audit_log,
}


def dispatch(name: str, args: dict) -> Any:
    """Execute a tool by name with parsed args. Catches exceptions so a
    misbehaving tool surfaces as a tool result, not a graph crash."""
    fn = TOOL_FUNCTIONS.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**args)
    except TypeError as exc:
        # Likely a bad argument shape from the LLM.
        return {"error": f"Bad arguments to {name}: {exc}"}
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return {"error": f"{name} raised {type(exc).__name__}: {exc}"}


def tool_to_intent(tool_name: str) -> str:
    """Map a tool name to one of the legacy intent labels so the eval +
    trace UI keep working without redesigning their intent column."""
    return {
        "book_appointment": "booking",
        "cancel_booking": "booking",
        "list_doctors": "booking",
        "get_doctor_schedule": "schedule",
        "list_my_bookings": "schedule",
        "find_patient": "records",
        "list_patients": "records",
        "upsert_patient": "records",
        "get_patient_history": "history",
        "medical_search": "medical_search",
        "get_audit_log": "audit",
    }.get(tool_name, "general")
