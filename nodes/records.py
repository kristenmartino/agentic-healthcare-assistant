"""Records node — adds or updates a patient record in the EHR DB.

Lightweight LLM-free parsing: looks for explicit "key: value" pairs and
common phrasings ("age 45", "phone +91...", "summary: hypertension").

Identity vs. value (issue #11): "update my name to X" means *set name = X
on the speaker's record* — X is the new VALUE, not the patient to look up.
The record being changed is resolved from context (the active patient, or
an appointment confirmation number), never from the new name. When no
identity can be resolved for an update, we refuse rather than insert a
junk record.

PHI scope: writes are gated the same fail-closed way as reads (see
nodes/history.py and the agent dispatcher in nodes/agent_tools.py). A
patient_chat caller may only write to a record it can prove a link to —
the authenticated active patient, or a record resolved from an appointment
confirmation number — plus insert a brand-new record. It may NOT edit an
existing record purely by naming it (that targets someone else's PHI).
Clinician/admin callers are unrestricted.
"""
from __future__ import annotations

import logging
import re

from config import load_settings
from state import HealthcareState
from tools.audit import log_access
from tools.ehr import add_or_update_patient, find_patient_by_name, list_patients

logger = logging.getLogger(__name__)

# Roles that may write to any patient's record. patient_chat is scoped to
# the active / confirmation-resolved patient (plus new inserts); everything
# else here is a trusted caller.
_UNRESTRICTED_ROLES = {"clinician", "admin"}


# Appointment confirmation numbers look like "AGS-681558".
_CONFIRMATION_RE = re.compile(r"\bAGS-\d+\b", re.I)

# Phrases that mean "this is an update to an existing record", not a new
# registration. Used to decide refuse-vs-insert when no identity resolves.
_UPDATE_PHRASES = ("update", "change", "rename", "edit", "correct", "fix", "modify")
_REGISTER_PHRASES = ("add ", "register", "new patient", "create", "enroll", "sign up")


def _extract_confirmation_no(text: str) -> str | None:
    m = _CONFIRMATION_RE.search(text)
    return m.group(0).upper() if m else None


def _extract_new_name_value(text: str) -> str | None:
    """For 'update my name to X' / 'change name to X' / 'rename ... to X' /
    'name: X', return X — the NEW name value. None if no such phrasing.

    Captures 1-3 capitalised words so 'Kristen Martino' is kept whole but a
    trailing sentence isn't swallowed.
    """
    name_val = r"([A-Z][a-z]+(?:\s+[A-Z][a-z'.-]+){0,2})"
    patterns = [
        rf"\bname\s*(?:to|:|=|should be|is now|as)\s+{name_val}",
        rf"\brename\b[^.]*?\bto\s+{name_val}",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return m.group(1).strip()
    return None


def _looks_like_registration(text: str) -> bool:
    t = text.lower()
    if any(k in t for k in _UPDATE_PHRASES):
        return False
    return any(k in t for k in _REGISTER_PHRASES)


def _resolve_patient_by_confirmation(conf: str, settings) -> dict | None:
    """Resolve an appointment confirmation number to the patient who booked
    it: {patient_id, name}. None if no active booking matches."""
    from tools.appointments import list_all_bookings
    try:
        bookings = list_all_bookings(settings.appointments_db_path)
    except Exception as exc:
        logger.warning("confirmation lookup failed: %s", exc)
        return None
    for b in bookings:
        if (b.get("confirmation_no") or "").upper() == conf:
            pid = b.get("booked_by_patient_id")
            if not pid:
                return None
            existing = next(
                (p for p in list_patients(settings, actor="patient_chat")
                 if p.get("patient_id") == pid),
                None,
            )
            return {"patient_id": pid,
                    "name": existing.get("name") if existing else None}
    return None


def _parse_fields(text: str) -> dict:
    """Extract record field changes from free-form text.

    Confirmation numbers are stripped first so they can't be mistaken for a
    phone number, and the phone matcher requires >= 10 actual digits so a
    short id like '681558' is never captured as a phone.
    """
    fields: dict = {}
    cleaned = _CONFIRMATION_RE.sub(" ", text)

    # Age: "age 45", "45 years old", "aged 45"
    age_match = re.search(
        r"\b(?:age[d]?\s*(\d{1,3})|(\d{1,3})\s*years?\s*old)\b", cleaned, re.I)
    if age_match:
        fields["age"] = int(age_match.group(1) or age_match.group(2))

    # Gender
    if re.search(r"\bfemale\b", cleaned, re.I):
        fields["gender"] = "Female"
    elif re.search(r"\bmale\b", cleaned, re.I):
        fields["gender"] = "Male"

    # Phone: a run of phone-ish characters with at least 10 real digits.
    phone_match = re.search(r"\+?[\d\-\(\) ]{10,}", cleaned)
    if phone_match and sum(c.isdigit() for c in phone_match.group(0)) >= 10:
        fields["phone_raw"] = phone_match.group(0).strip()

    # Email
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", cleaned)
    if email_match:
        fields["email"] = email_match.group(0)

    # Explicit "key: value" parses (summary, address, condition)
    for key in ("summary", "address", "condition", "diagnosis", "notes"):
        m = re.search(rf"\b{key}\s*:\s*([^\.]+?)(?:\.|$)", cleaned, re.I)
        if m:
            target = "summary" if key in ("summary", "condition", "diagnosis", "notes") else key
            fields[target] = m.group(1).strip()
            break

    # Bare condition keyword without "summary:"
    if "summary" not in fields:
        for cond in ("hypertension", "diabetes", "asthma", "kidney disease",
                     "heart disease", "cancer", "depression", "anxiety"):
            if cond in cleaned.lower():
                fields["summary"] = f"Reported {cond}; details to follow."
                break

    return fields


def _refuse(message: str, reason: str) -> dict:
    return {
        "error": message,
        "tool_log": [{"node": "records", "result": "refused", "reason": reason}],
    }


def records_node(state: HealthcareState) -> dict:
    settings = load_settings()
    user_input = state.get("user_input") or ""
    role = (state.get("role") or "patient_chat").lower()
    trusted = role in _UNRESTRICTED_ROLES

    new_name = _extract_new_name_value(user_input)

    # --- Resolve WHO this change targets, independent of any new name value. ---
    target_pid = state.get("patient_id")  # active patient (sidebar selection)
    target_name = None
    resolution = "active_patient" if target_pid else None

    if not target_pid:
        conf = _extract_confirmation_no(user_input)
        if conf:
            resolved = _resolve_patient_by_confirmation(conf, settings)
            if resolved:
                target_pid = resolved["patient_id"]
                target_name = resolved.get("name")
                resolution = "confirmation_no"
            else:
                return _refuse(
                    f"I couldn't find an active booking matching {conf}, so I "
                    "can't tell whose record to update. Please double-check the "
                    "confirmation number or select the patient first.",
                    "confirmation_unresolved",
                )

    # Classic "update <Name>'s record" — the named patient IS the subject,
    # but ONLY when there's no explicit 'set name to X' value in the message
    # (otherwise the name in the text is the new value, not the subject).
    #
    # PHI scope: resolving an EXISTING record purely by name is only reached
    # when there's no active patient (a walk-in), so for patient_chat it can
    # only ever target SOMEONE ELSE's record. Restrict edit-by-name to
    # trusted roles; patient_chat falls through to register-or-refuse.
    if not target_pid and not new_name and trusted:
        subject = state.get("patient_name")
        if subject:
            existing = find_patient_by_name(subject, settings, actor=role)
            if existing:
                target_pid = existing["patient_id"]
                target_name = existing.get("name")
                resolution = "name_subject"

    # --- Parse field changes (name handled separately above). ---
    fields = _parse_fields(user_input)
    if new_name:
        fields["name"] = new_name

    # --- Update an identified record. ---
    if target_pid:
        fields["patient_id"] = target_pid
        # Preserve the existing name on a non-rename update so we don't blank it.
        if "name" not in fields and target_name:
            fields["name"] = target_name
        return _apply(fields, settings, resolution)

    # --- No identity resolved. Register vs. refuse. ---
    if _looks_like_registration(user_input):
        reg_name = new_name or state.get("patient_name")
        if not reg_name:
            return _refuse(
                "To register a new patient I need their full name.",
                "registration_no_name",
            )
        # PHI scope: add_or_update_patient upserts by name, so a "registration"
        # whose name collides with an existing patient would silently UPDATE
        # that patient's record. For patient_chat (no proven link to that
        # record) refuse rather than write to someone else's PHI.
        if not trusted:
            existing = find_patient_by_name(reg_name, settings, actor="patient_chat")
            if existing:
                log_access(
                    "patient_chat", "ehr.write", "Patient", existing.get("patient_id"),
                    patient_id=existing.get("patient_id"), outcome="denied",
                    details={"reason": "registration_name_collision",
                             "reg_name": reg_name},
                )
                return _refuse(
                    "A patient named "
                    f"{reg_name} already exists. I can't modify an existing "
                    "record from a walk-in session — select that patient (or "
                    "provide their appointment confirmation number) and try "
                    "again.",
                    "registration_name_collision",
                )
        fields["name"] = reg_name
        return _apply(fields, settings, "registration")

    return _refuse(
        "I couldn't tell which patient's record to update. Select the patient "
        "first, or include the appointment confirmation number (e.g. AGS-123456) "
        "so I can find the right record.",
        "no_identity",
    )


def _apply(fields: dict, settings, resolution: str | None) -> dict:
    try:
        result = add_or_update_patient(fields, settings, actor="patient_chat")
        return {
            "patient_id": result["patient_id"],
            "record_change": result,
            "tool_log": [{
                "node": "records",
                "tool": "add_or_update_patient",
                "ehr_backend": settings.ehr_backend,
                "operation": result["operation"],
                "patient_id": result["patient_id"],
                "identity_resolution": resolution,
                "fields_set": list(fields.keys()),
            }],
        }
    except Exception as exc:
        logger.exception("Records update failed")
        return {
            "error": f"Records update failed: {exc}",
            "tool_log": [{"node": "records", "result": "failed", "error": str(exc)}],
        }
