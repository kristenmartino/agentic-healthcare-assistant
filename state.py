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
from typing import Annotated, Any, Literal, Optional, TypedDict

# The 5 intents the classifier can return. "general" is a fallback for
# greetings, definitions, and chit-chat that don't fit the other 4.
Intent = Literal[
    "booking",          # book an appointment
    "records",          # add or update a patient record
    "history",          # retrieve / summarize a patient's history
    "medical_search",   # search Medline / WHO / general web for medical info
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


def _merge_error(left: Optional[str], right: Optional[str]) -> Optional[str]:
    """Reducer for the `error` field — concatenate distinct errors."""
    if not left:
        return right
    if not right:
        return left
    if left == right:
        return left
    return f"{left} · {right}"


class HealthcareState(TypedDict, total=False):
    # User input + identification
    user_input: str
    patient_id: Optional[str]              # SHA1 hash; matches ehr.sqlite
    patient_name: Optional[str]            # display name
    requested_specialty: Optional[Specialty]
    requested_date: Optional[str]          # ISO date string if extracted

    # Intent routing
    intent: Intent                         # primary intent
    intents: Annotated[list[Intent], add]  # multi-intent (parallel fan-out)

    # Per-branch outputs
    appointment: Optional[dict[str, Any]]  # {doctor, datetime, slot_id, confirmation_no}
    record_change: Optional[dict[str, Any]]  # {operation, fields, before, after}
    history_summary: Optional[str]
    medical_info: Optional[list[dict[str, Any]]]  # [{title, snippet, url, source}]

    # Composer output
    response: str
    sources: Annotated[list[dict[str, Any]], add]

    # Safety pre-classifier — set by nodes/safety.py before classify_intent
    is_emergency: bool
    emergency_categories: list[str]

    # Cross-cutting
    error: Annotated[Optional[str], _merge_error]
    tool_log: Annotated[list[dict[str, Any]], add]   # for the logs UI
    history: list[dict[str, str]]                    # conversation history (Streamlit feeds this)
