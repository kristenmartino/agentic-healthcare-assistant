"""State schema for the Healthcare Assistant LangGraph workflow.

LangGraph merges partial state updates from each node. Without a reducer,
the default is "last write wins", which can drop information when two
parallel branches (booking + medical_search, or booking + history) both
write to the same field.

Fields with explicit reducers:

- `error`: parallel branches can each report a failure. We concatenate them
  with " · " so neither failure is silently dropped before the composer
  decides how to surface them.
- `tool_log`: every node may append. Use `add` so entries from parallel
  branches are merged rather than overwritten.
- `sources`: appended by medical_search and (potentially) history. Use `add`.
- `intents`: the classifier may set multiple intents for fan-out queries; use
  `add` so the merge from parallel branches is safe.

Single-writer fields don't need reducers but are typed permissively so
LangGraph doesn't reject empty-list returns.
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, TypedDict

# Intents the workflow can produce. The classifier (legacy `graph` mode)
# emits one or more from this set; the agent_loop (`react` mode) maps each
# tool call back to one of these labels for the trace UI + eval scoring.
#
# `schedule` and `audit` were promoted to first-class in PR #6 because the
# tool surface added real capabilities (get_doctor_schedule, get_audit_log)
# that aren't accurately described as `booking` or `records`. Hiding them
# under those aliases would silently lose product behavior in traces, eval,
# and the UI intent badges.
Intent = Literal[
    "booking",          # book an appointment
    "records",          # add or update a patient record
    "history",          # retrieve / summarize a patient's history
    "medical_search",   # search Medline / WHO / general web for medical info
    "schedule",         # query a doctor's calendar OR list a patient's bookings
    "audit",            # PHI access audit log query
    "emergency",        # safety classifier fired (set in nodes/safety.py)
    "general",          # fallback (greeting, off-topic, definitions)
]

# Specialties the appointment system supports. The classifier extracts these
# from user input where possible; if absent, the booking node prompts.
Specialty = Literal[
    "general_practice",
    "cardiology",
    "endocrinology",
    "nephrology",
    "neurology",
    "pulmonology",
    "oncology",
    "psychiatry",
    "dermatology",
]


def _merge_error(left: str | None, right: str | None) -> str | None:
    """Reducer for the `error` field — concatenate distinct errors."""
    if not left:
        return right
    if not right:
        return left
    if left == right:
        return left
    return f"{left} · {right}"


def _merge_timings(
    left: dict[str, float] | None, right: dict[str, float] | None
) -> dict[str, float]:
    """Reducer for `node_timings` — union the per-node durations so parallel
    fan-out branches don't clobber each other's entries."""
    if not left:
        return right or {}
    if not right:
        return left
    return {**left, **right}


class HealthcareState(TypedDict, total=False):
    # User input + identification
    user_input: str
    patient_id: str | None              # SHA1 hash; matches ehr.sqlite
    patient_name: str | None            # display name
    requested_specialty: Specialty | None
    requested_date: str | None          # ISO date string if extracted

    # PHI access scope. Used by the agent_loop dispatcher to gate tools.
    # Default is patient_chat (most restrictive). The Streamlit Doctor View
    # / MCP paths can pass role="clinician" or "admin" to unlock broader
    # tools like list_patients and unfiltered get_audit_log.
    role: str | None

    # Intent routing
    intent: Intent                         # primary intent
    intents: Annotated[list[Intent], add]  # multi-intent (parallel fan-out)

    # Per-branch outputs (set by the legacy branch nodes; also populated by
    # the agent_loop accumulator so the UI artifact panels + eval keep
    # working regardless of which reasoning strategy ran).
    appointment: dict[str, Any] | None  # {doctor, datetime, slot_id, confirmation_no}
    record_change: dict[str, Any] | None  # {operation, fields, before, after}
    history_summary: str | None
    medical_info: list[dict[str, Any]] | None  # [{title, snippet, url, source}]

    # Agent-only structured outputs (populated by the react path). The
    # legacy graph never sets these because it doesn't expose the
    # capabilities; the artifact rows are agent-mode-only.
    schedule_results: dict[str, Any] | None     # get_doctor_schedule
    bookings_results: list[dict[str, Any]] | None  # list_my_bookings
    audit_results: list[dict[str, Any]] | None     # get_audit_log
    doctor_results: list[dict[str, Any]] | None    # list_doctors
    patient_listing: list[dict[str, Any]] | None   # list_patients (admin only)

    # Composer output
    response: str
    sources: Annotated[list[dict[str, Any]], add]

    # Safety pre-classifier — set by nodes/safety.py before classify_intent
    is_emergency: bool
    emergency_categories: list[str]

    # Cross-cutting
    error: Annotated[str | None, _merge_error]
    tool_log: Annotated[list[dict[str, Any]], add]   # for the logs UI
    history: list[dict[str, str]]                    # conversation history (Streamlit feeds this)

    # Per-node wall-clock durations (ms), keyed by node name. Populated by the
    # timing wrapper in graph.py so the trace log can show where time goes.
    node_timings: Annotated[dict[str, float], _merge_timings]
