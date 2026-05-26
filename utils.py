"""Small UI helpers shared by app.py, pages/, and the composer template."""
from __future__ import annotations

from datetime import datetime


def format_appointment_time(iso_str: str | None) -> str:
    """ISO datetime → 'Tue, May 5 · 9:00 AM'. Returns the input unchanged on parse failure."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", ""))
    except (ValueError, TypeError):
        return str(iso_str)
    # %-d / %-I are macOS/Linux; on Windows use %#d / %#I. We assume POSIX (the lab VM is Linux).
    try:
        return dt.strftime("%a, %b %-d · %-I:%M %p")
    except ValueError:
        # Windows fallback
        return dt.strftime("%a, %b %#d · %#I:%M %p")


def format_date(iso_str: str | None) -> str:
    """ISO date → 'Tue, May 5'."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "")[:10])
    except (ValueError, TypeError):
        return str(iso_str)
    try:
        return dt.strftime("%a, %b %-d")
    except ValueError:
        return dt.strftime("%a, %b %#d")


# Friendly labels for LangGraph node names (used by the streaming UI).
NODE_LABELS = {
    "classify_intent": "Routing intent",
    "booking_node": "Booking appointment",
    "records_node": "Updating records",
    "history_node": "Retrieving history",
    "medical_search_node": "Searching medical sources",
    "compose_response": "Composing response",
}


def node_label(node_name: str) -> str:
    return NODE_LABELS.get(node_name, node_name.replace("_", " ").title())


def summarize_node_update(node_name: str, partial: dict) -> str:
    """One-line summary of what a node produced, for the streaming progress UI."""
    if node_name == "classify_intent":
        intents = partial.get("intents") or [partial.get("intent", "?")]
        return f"Routed to: {', '.join(intents)}"
    if node_name == "booking_node":
        appt = partial.get("appointment") or {}
        if appt:
            return (
                f"Booked: {appt.get('doctor_name', '?')} "
                f"({appt.get('specialty', '?')}) — {appt.get('confirmation_no', '?')}"
            )
        err = partial.get("error")
        return f"Booking failed: {err}" if err else "(no booking)"
    if node_name == "records_node":
        rec = partial.get("record_change") or {}
        if rec:
            return f"{rec.get('operation', '?')}: {(rec.get('after') or {}).get('name', '?')}"
        err = partial.get("error")
        return f"Records skipped: {err}" if err else "(no record change)"
    if node_name == "history_node":
        summary = partial.get("history_summary") or ""
        return f"History summary ({len(summary)} chars)"
    if node_name == "medical_search_node":
        info = partial.get("medical_info") or []
        # The first entry can be a synthesis dict; count actual results.
        results = [e for e in info if "synthesis" not in e]
        return f"Found {len(results)} sources"
    if node_name == "compose_response":
        resp = partial.get("response") or ""
        return f"Composed ({len(resp)} chars)"
    return ""
