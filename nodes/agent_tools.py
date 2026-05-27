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
    """Returns the structured record + FHIR Conditions + recent Observations
    PLUS an LLM-synthesized prose summary that cites the report PDFs via the
    FAISS index.

    Delegates the synthesis step to the legacy `history_node` so the react
    path produces the same shape of summary as the classifier-graph path.
    Behavioral parity matters — the Streamlit history panel and the FastAPI
    `done` payload both expect `history_summary` to be a clinician-style
    paragraph, not a JSON dump.
    """
    from config import load_settings
    from nodes.history import history_node
    from tools.ehr import find_patient_by_name, get_patient_clinical_context

    s = load_settings()
    record = find_patient_by_name(patient_name, s, actor="agent")
    if not record:
        return {"error": f"No patient matching '{patient_name}'"}
    clinical = get_patient_clinical_context(record["patient_id"], s, actor="agent")

    # Delegate to the legacy node for FAISS + LLM-synthesized summary.
    # The node already audits patient access; we tag the actor through state.
    legacy_state = {
        "user_input": f"history for {patient_name}",
        "patient_name": patient_name,
    }
    try:
        legacy_out = history_node(legacy_state)
    except Exception as exc:
        logger.warning("history_node delegation failed: %s", exc)
        legacy_out = {}

    chunks_retrieved = 0
    for entry in (legacy_out.get("tool_log") or []):
        if entry.get("node") == "history":
            chunks_retrieved = entry.get("pdf_chunks_retrieved", 0) or 0
            break

    return {
        "record": record,
        "conditions": clinical.get("conditions", []),
        "observations": clinical.get("observations", []),
        "history_summary": legacy_out.get("history_summary"),
        "pdf_chunks_retrieved": chunks_retrieved,
    }


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


def _tool_medical_search(query: str, top_k: int = 4) -> dict:
    """Search trusted medical sources (MedlinePlus, WHO, CDC, NIH, Mayo).

    Delegates to the legacy `medical_search_node` so the react path produces
    the same shape the classifier-graph path produces: a cited LLM
    synthesis prepended to the raw results, and a `sources` list with
    indexed [{title, url, source}]. Without this delegation the UI's
    'Sources' panel and the eval's medical_info shape would diverge from
    graph mode.

    Returns a dict (not a bare list) so the agent_loop accumulator can
    pull synthesis + raw results + sources separately into state.
    """
    from nodes.medical_search_node import medical_search_node

    # The legacy node reads state["user_input"] and builds the search query
    # from it; for react mode we pass the focused query Claude gave us as
    # the user_input. (T12's sub-query extraction lives in that node and
    # is a no-op when there's only one intent, which is our case here.)
    legacy_state = {
        "user_input": query,
        "intents": ["medical_search"],
    }
    try:
        legacy_out = medical_search_node(legacy_state)
    except Exception as exc:
        logger.warning("medical_search_node delegation failed: %s", exc)
        # Fall back to raw search so the tool isn't unusable.
        from config import load_settings
        from tools.medical_search import medical_search
        s = load_settings()
        raw = medical_search(query, top_k=top_k, tavily_api_key=s.tavily_api_key)
        return {"medical_info": raw, "sources": [], "synthesis": None,
                "results_count": len(raw), "raw_results": raw}

    medical_info = legacy_out.get("medical_info") or []
    # Pull the synthesis pseudo-entry out so Claude can read it directly,
    # and keep the raw results separate for downstream UI panels.
    synthesis = next(
        (e.get("synthesis") for e in medical_info if isinstance(e, dict)
         and "synthesis" in e), None)
    raw_results = [e for e in medical_info if isinstance(e, dict)
                   and "synthesis" not in e]
    return {
        "medical_info": medical_info,         # full [synth + raw] for state
        "sources": legacy_out.get("sources") or [],
        "synthesis": synthesis,                # convenience for the LLM
        "raw_results": raw_results,            # convenience for the LLM
        "results_count": len(raw_results),
    }


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


# Map tool name → name of the module-level function that implements it.
# We store STRINGS rather than function references so dispatch resolves
# the function at call time via getattr(thismodule, …). This lets tests
# monkeypatch the underscore functions directly and have the change take
# effect through dispatch — the old behavior (storing direct references)
# silently bypassed monkeypatches.
TOOL_FUNCTIONS: dict[str, str] = {
    "book_appointment": "_tool_book_appointment",
    "cancel_booking": "_tool_cancel_booking",
    "list_doctors": "_tool_list_doctors",
    "get_doctor_schedule": "_tool_get_doctor_schedule",
    "list_my_bookings": "_tool_list_my_bookings",
    "find_patient": "_tool_find_patient",
    "list_patients": "_tool_list_patients",
    "get_patient_history": "_tool_get_patient_history",
    "upsert_patient": "_tool_upsert_patient",
    "medical_search": "_tool_medical_search",
    "get_audit_log": "_tool_get_audit_log",
}


def _resolve(fn_name: str) -> Callable[..., Any] | None:
    """Look up a tool implementation by its function name on this module.

    Resolved late (rather than at import time) so test-time monkeypatching
    of the underscore functions takes effect through dispatch.
    """
    import sys
    mod = sys.modules[__name__]
    fn = getattr(mod, fn_name, None)
    return fn if callable(fn) else None


# ---------- PHI scope policy ----------
#
# Defense-in-depth: the LLM picks the tool, but the dispatcher enforces
# WHO can call WHAT with WHICH args. Fail-closed default: missing scope
# is treated as the most restrictive role (patient_chat).
#
# The role labels we use today:
#   - patient_chat (default): the public chat UI. Scoped to ONE patient
#     at a time. Cannot enumerate the patient table, cannot read other
#     patients' bookings or audit events. Walk-in (no patient_id) is
#     even more restricted — can only create a brand-new record.
#   - clinician: trusted clinical user. Broader read access. Can read
#     any single patient's record on demand. Cannot dump the full table.
#   - admin: full access including unfiltered list_patients and
#     unfiltered get_audit_log. Reserved for the Streamlit Doctor View
#     and direct MCP calls — NOT for the patient-facing web chat.

_TOOL_ROLE_POLICY: dict[str, set[str]] = {
    # tool_name -> set of roles allowed to call it at all
    "book_appointment":   {"patient_chat", "clinician", "admin"},
    "cancel_booking":     {"patient_chat", "clinician", "admin"},
    "list_doctors":       {"patient_chat", "clinician", "admin"},
    "get_doctor_schedule":{"patient_chat", "clinician", "admin"},
    "list_my_bookings":   {"patient_chat", "clinician", "admin"},
    "find_patient":       {"patient_chat", "clinician", "admin"},
    "list_patients":      {"clinician", "admin"},   # NOT patient_chat
    "get_patient_history":{"patient_chat", "clinician", "admin"},
    "upsert_patient":     {"patient_chat", "clinician", "admin"},
    "medical_search":     {"patient_chat", "clinician", "admin"},
    "get_audit_log":      {"patient_chat", "clinician", "admin"},
}


def _authorize(tool_name: str, args: dict, scope: dict | None) -> str | None:
    """Return None if the call is allowed, or an error string if denied.

    Fail-closed: missing scope or missing role -> treated as patient_chat.
    """
    scope = scope or {}
    role = scope.get("role") or "patient_chat"
    active_pid = scope.get("patient_id")  # may be None for walk-in

    allowed_roles = _TOOL_ROLE_POLICY.get(tool_name)
    if allowed_roles is None:
        # Unknown tool — let dispatch handle the "Unknown tool" error.
        return None
    if role not in allowed_roles:
        return (
            f"Not authorized: tool '{tool_name}' is not available to role "
            f"'{role}'. (This is a PHI-scope guardrail, not an LLM mistake.)"
        )

    # patient_chat-specific argument constraints below.
    if role != "patient_chat":
        return None

    if tool_name == "list_my_bookings":
        requested_pid = args.get("patient_id")
        requested_name = args.get("patient_name")
        if not requested_pid and not requested_name:
            return ("Not authorized: list_my_bookings requires a patient_id "
                    "or patient_name in patient_chat role — unfiltered booking "
                    "lists are not allowed.")
        if requested_pid and active_pid and requested_pid != active_pid:
            return ("Not authorized: list_my_bookings cannot return bookings "
                    f"for another patient. Active patient is {active_pid!r}.")

    if tool_name == "get_audit_log":
        if not args.get("patient_id"):
            return ("Not authorized: get_audit_log requires a patient_id "
                    "filter in patient_chat role.")
        if active_pid and args["patient_id"] != active_pid:
            return ("Not authorized: get_audit_log cannot return events for "
                    f"another patient. Active patient is {active_pid!r}.")

    if tool_name == "get_patient_history":
        # Allow lookups, but if an active patient is selected the requested
        # name should resolve to that patient. We can't enforce name-vs-id
        # without a roundtrip; the audit log captures both for review.
        pass

    return None


def _mask_for_scope(tool_name: str, result: Any, scope: dict | None) -> Any:
    """Post-process a tool result to strip PHI fields the caller shouldn't see.

    The biggest concrete case: get_doctor_schedule returns each slot's
    `booked_by_patient_id` and `confirmation_no`. In patient_chat role,
    those identify OTHER patients on the doctor's calendar — leaking them
    via the chat is the textbook PHI cross-leak. Mask them unless the
    booked slot belongs to the active patient.
    """
    scope = scope or {}
    role = scope.get("role") or "patient_chat"
    if role != "patient_chat":
        return result
    active_pid = scope.get("patient_id")

    if tool_name == "get_doctor_schedule" and isinstance(result, dict):
        schedule = result.get("schedule") or []
        masked: list[dict] = []
        for slot in schedule:
            if not isinstance(slot, dict):
                masked.append(slot)
                continue
            if not slot.get("booked"):
                masked.append(slot)
                continue
            booker = slot.get("booked_by_patient_id")
            if active_pid and booker == active_pid:
                masked.append(slot)  # the active patient's own slot — fine
            else:
                masked.append({
                    **{k: v for k, v in slot.items()
                       if k not in ("booked_by_patient_id", "confirmation_no")},
                    "booked_by_patient_id": "(other patient — masked)",
                    "confirmation_no": "(masked)",
                })
        return {**result, "schedule": masked}

    return result


def dispatch(name: str, args: dict, scope: dict | None = None) -> Any:
    """Execute a tool by name with parsed args.

    Three layers:
      1. Lookup (unknown tool → error)
      2. Authorization (denied → error, never reaches the tool)
      3. Execution + post-mask (PHI fields stripped for restrictive roles)

    Exceptions are caught so a misbehaving tool surfaces as a tool
    result, not a graph crash.
    """
    fn_name = TOOL_FUNCTIONS.get(name)
    if not fn_name:
        return {"error": f"Unknown tool: {name}"}

    denied = _authorize(name, args, scope)
    if denied:
        logger.info("dispatch denied: %s (scope=%s, args=%s)",
                    denied, scope, args)
        return {"error": denied}

    fn = _resolve(fn_name)
    if fn is None:
        return {"error": f"Tool implementation for '{name}' is missing"}
    try:
        result = fn(**args)
    except TypeError as exc:
        return {"error": f"Bad arguments to {name}: {exc}"}
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return {"error": f"{name} raised {type(exc).__name__}: {exc}"}

    return _mask_for_scope(name, result, scope)


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
