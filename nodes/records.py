"""Records node — adds or updates a patient record in the EHR DB.

Lightweight LLM-free parsing: looks for explicit "key: value" pairs and
common phrasings ("age 45", "phone +91...", "summary: hypertension").
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from config import load_settings
from state import HealthcareState
from tools.ehr import add_or_update_patient, find_patient_by_name

logger = logging.getLogger(__name__)


def _parse_fields(text: str, name: Optional[str]) -> dict:
    """Extract a record dict from free-form text."""
    fields: dict = {}
    if name:
        fields["name"] = name

    # Age: "age 45", "45 years old", "aged 45"
    age_match = re.search(r"\b(?:age[d]?\s*(\d{1,3})|(\d{1,3})\s*years?\s*old)\b", text, re.I)
    if age_match:
        fields["age"] = int(age_match.group(1) or age_match.group(2))

    # Gender: "male" / "female" / "other"
    if re.search(r"\bfemale\b", text, re.I):
        fields["gender"] = "Female"
    elif re.search(r"\bmale\b", text, re.I):
        fields["gender"] = "Male"

    # Phone: any sequence of 8+ digits, optionally with prefix
    phone_match = re.search(r"\+?[\d\-\(\) ]{8,}", text)
    if phone_match:
        fields["phone_raw"] = phone_match.group(0).strip()

    # Email
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if email_match:
        fields["email"] = email_match.group(0)

    # Explicit "key: value" parses (summary, address, condition)
    for key in ("summary", "address", "condition", "diagnosis", "notes"):
        m = re.search(rf"\b{key}\s*:\s*([^\.]+?)(?:\.|$)", text, re.I)
        if m:
            target = "summary" if key in ("summary", "condition", "diagnosis", "notes") else key
            fields[target] = m.group(1).strip()
            break

    # If we matched a condition keyword without "summary:", capture as a note
    if "summary" not in fields:
        for cond in ("hypertension", "diabetes", "asthma", "kidney disease",
                     "heart disease", "cancer", "depression", "anxiety"):
            if cond in text.lower():
                fields["summary"] = f"Reported {cond}; details to follow."
                break

    return fields


def records_node(state: HealthcareState) -> dict:
    settings = load_settings()
    user_input = state.get("user_input") or ""
    patient_name = state.get("patient_name")

    if not patient_name:
        return {
            "error": "Records update requires a patient name; none was extracted.",
            "tool_log": [{
                "node": "records",
                "result": "skipped",
                "reason": "no patient_name in state",
            }],
        }

    fields = _parse_fields(user_input, patient_name)

    # If patient already exists, this is an update; merge with existing fields
    existing = find_patient_by_name(patient_name, settings)
    if existing:
        fields["patient_id"] = existing["patient_id"]

    try:
        result = add_or_update_patient(fields, settings)
        return {
            "patient_id": result["patient_id"],
            "record_change": result,
            "tool_log": [{
                "node": "records",
                "tool": "add_or_update_patient",
                "ehr_backend": settings.ehr_backend,
                "operation": result["operation"],
                "patient_id": result["patient_id"],
                "fields_set": list(fields.keys()),
            }],
        }
    except Exception as exc:
        logger.exception("Records update failed")
        return {
            "error": f"Records update failed: {exc}",
            "tool_log": [{
                "node": "records",
                "result": "failed",
                "error": str(exc),
            }],
        }
