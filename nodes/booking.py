"""Booking node — books an appointment via the mock Doctor Schedule API."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from config import load_settings
from state import HealthcareState
from tools.appointments import book_appointment
from tools.ehr import find_patient_by_name

logger = logging.getLogger(__name__)


def booking_node(state: HealthcareState) -> dict:
    settings = load_settings()
    specialty = state.get("requested_specialty") or "general_practice"
    patient_name = state.get("patient_name") or "Walk-in Patient"
    patient_id = state.get("patient_id")

    # Resolve patient_id from name if not already set
    if not patient_id and patient_name and patient_name != "Walk-in Patient":
        try:
            patient = find_patient_by_name(patient_name, settings)
            if patient:
                patient_id = patient["patient_id"]
        except Exception as exc:
            logger.warning("Could not look up patient by name: %s", exc)

    if not patient_id:
        # Synthesize a session-only ID for walk-ins
        import hashlib
        patient_id = "walkin-" + hashlib.sha1(patient_name.encode()).hexdigest()[:8]

    # Default to "any time tomorrow or later"
    preferred_date = state.get("requested_date") or (
        datetime.now() + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    try:
        appointment = book_appointment(
            settings.appointments_db_path,
            patient_id=patient_id,
            patient_name=patient_name,
            specialty=specialty,
            preferred_date=preferred_date,
        )
        log_entry = {
            "node": "booking",
            "tool": "book_appointment",
            "args": {
                "specialty": specialty,
                "preferred_date": preferred_date,
                "patient_name": patient_name,
            },
            "result": "success",
            "confirmation_no": appointment["confirmation_no"],
        }
        return {
            "patient_id": patient_id,
            "appointment": appointment,
            "tool_log": [log_entry],
        }
    except ValueError as exc:
        logger.warning("Booking failed: %s", exc)
        return {
            "error": f"Booking failed: {exc}",
            "tool_log": [{
                "node": "booking",
                "tool": "book_appointment",
                "args": {"specialty": specialty},
                "result": "failed",
                "error": str(exc),
            }],
        }
    except Exception as exc:
        logger.exception("Unexpected booking error")
        return {
            "error": f"Unexpected booking error: {exc}",
            "tool_log": [{"node": "booking", "result": "exception", "error": str(exc)}],
        }
