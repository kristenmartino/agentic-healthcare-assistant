"""Composer node — assembles the final response from per-branch outputs."""
from __future__ import annotations

import logging

from llm import LLMUnavailable, chat
from prompts import COMPOSER_PROMPT
from state import HealthcareState
from utils import format_appointment_time

logger = logging.getLogger(__name__)


_DISCLAIMER = (
    "ℹ️ This assistant provides informational support only and is not a "
    "substitute for advice from a licensed clinician."
)

# Cap how many prior conversation turns we replay into the composer prompt.
_HISTORY_TURN_CAP = 8


def _build_context_block(state: HealthcareState) -> str:
    """Render the available branch outputs as a structured prompt section."""
    sections: list[str] = []

    appt = state.get("appointment")
    if appt:
        sections.append(
            f"=== Appointment ===\n"
            f"Doctor: {appt['doctor_name']} ({appt['specialty']})\n"
            f"Time: {format_appointment_time(appt['start_time'])}\n"
            f"Confirmation #: {appt['confirmation_no']}\n"
            f"Patient: {appt.get('patient_name', '?')}"
        )

    rec = state.get("record_change")
    if rec:
        sections.append(
            f"=== Record change ===\n"
            f"Operation: {rec['operation']}\n"
            f"Patient ID: {rec['patient_id']}\n"
            f"After: {rec.get('after')}"
        )

    hist = state.get("history_summary")
    if hist:
        sections.append(f"=== History summary ===\n{hist}")

    info = state.get("medical_info") or []
    if info:
        # First entry may be a synthesis dict; rest are raw results
        synthesis = next((e["synthesis"] for e in info if "synthesis" in e), "")
        results = [e for e in info if "synthesis" not in e]
        block = "=== Medical info ==="
        if synthesis:
            block += f"\n{synthesis}"
        if results:
            block += "\n\nRaw sources:"
            for i, r in enumerate(results, 1):
                block += f"\n[{i}] {r.get('title')} — {r.get('url')}"
        sections.append(block)

    err = state.get("error")
    if err:
        sections.append(f"=== Errors during processing ===\n{err}")

    return "\n\n".join(sections) or "(no branch produced output)"


def _fallback_template(state: HealthcareState) -> str:
    """Deterministic response when no LLM is available."""
    parts: list[str] = []

    appt = state.get("appointment")
    if appt:
        parts.append(
            f"✅ Booked **{appt['doctor_name']}** ({appt['specialty'].replace('_',' ')}) "
            f"for **{format_appointment_time(appt['start_time'])}**. "
            f"Confirmation #: `{appt['confirmation_no']}`."
        )

    rec = state.get("record_change")
    if rec:
        parts.append(
            f"✅ Record {rec['operation']}d for patient {rec['after'].get('name', '?')} "
            f"(ID {rec['patient_id']})."
        )

    hist = state.get("history_summary")
    if hist:
        parts.append(f"📋 History summary:\n{hist}")

    info = state.get("medical_info") or []
    if info:
        synthesis = next((e["synthesis"] for e in info if "synthesis" in e), "")
        if synthesis:
            parts.append(f"🔎 Medical info:\n{synthesis}")
        else:
            parts.append("🔎 Medical info: no synthesis available.")

    err = state.get("error")
    if err:
        parts.append(f"⚠️ Note: {err}")

    if not parts:
        parts.append(
            "Hello! I can help with booking appointments, managing patient records, "
            "retrieving medical history, and answering general medical questions. "
            "What would you like to do?"
        )

    parts.append(_DISCLAIMER)
    return "\n\n".join(parts)


def compose_response_node(state: HealthcareState) -> dict:
    context = _build_context_block(state)
    user_query = state.get("user_input", "")

    # Skip the LLM entirely when in stub mode — the stub matcher in llm.py
    # would mis-route on user-query keywords like "summarize" and produce
    # garbled output. The fallback template is more reliable in stub mode.
    from config import load_settings
    settings = load_settings()
    if settings.llm_provider == "stub":
        return {
            "response": _fallback_template(state),
            "tool_log": [{
                "node": "compose_response",
                "mode": "template (stub LLM provider)",
                "branches_present": [
                    k for k in ("appointment", "record_change", "history_summary", "medical_info")
                    if state.get(k)
                ],
            }],
        }

    user_block = (
        f"User query: {user_query}\n\n"
        f"Branch outputs:\n{context}\n\n"
        f"Compose the final response now."
    )

    # Thread prior conversation turns so the composer can resolve references
    # to earlier turns ("what about her cholesterol?", "book him in too").
    # Inserted between the system prompt and the current turn. Bounded to the
    # last few turns to keep tokens in check.
    messages: list[dict] = [{"role": "system", "content": COMPOSER_PROMPT}]
    for turn in (state.get("history") or [])[-_HISTORY_TURN_CAP:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if content and role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_block})

    try:
        response = chat(
            messages=messages,
            temperature=0.3,
            max_tokens=400,
        ).strip()
        if _DISCLAIMER not in response:
            response += "\n\n" + _DISCLAIMER
    except LLMUnavailable as exc:
        logger.warning("Composer LLM unavailable: %s — using template fallback", exc)
        response = _fallback_template(state)

    return {
        "response": response,
        "tool_log": [{
            "node": "compose_response",
            "branches_present": [
                k for k in ("appointment", "record_change", "history_summary", "medical_info")
                if state.get(k)
            ],
        }],
    }
