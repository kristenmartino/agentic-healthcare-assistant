"""History node — retrieves and summarizes a patient's medical history.

Combines the structured EHR record with FAISS-retrieved chunks from the
patient's PDF reports, then asks the LLM for a concise summary.

PHI scope: this is the patient-facing read path, so it enforces the same
fail-closed authorization the agent dispatcher does (nodes/agent_tools.py).
In `patient_chat` role a walk-in (no authenticated active patient) cannot
read any existing record, and an authenticated patient can only read their
OWN. Clinician/admin callers — the Doctor View, MCP, and the agent loop's
post-authorization delegation — are unrestricted.
"""
from __future__ import annotations

import logging

from config import load_settings
from llm import LLMUnavailable, chat
from prompts import HISTORY_SUMMARY_PROMPT
from state import HealthcareState
from tools.audit import log_access
from tools.ehr import find_patient_by_name, get_patient_clinical_context
from tools.fhir_client import condition_summary
from tools.vector_index import search_index

logger = logging.getLogger(__name__)

# Roles that may read any patient's history. patient_chat is scoped to the
# authenticated active patient; everything else here is a trusted caller.
_UNRESTRICTED_ROLES = {"clinician", "admin"}


def _phi_refusal(message: str, reason: str, patient_name: str) -> dict:
    """Deterministic refusal for a denied PHI read.

    Returns a `history_summary` (relayed by the composer) plus a denied
    audit entry. No record lookup result and no LLM summary are produced,
    so nothing about the requested patient leaks back to the caller.
    """
    log_access(
        "patient_chat", "ehr.read", "Patient", None,
        outcome="denied",
        details={"reason": reason, "search_name": patient_name},
    )
    return {
        "history_summary": message,
        "tool_log": [{
            "node": "history",
            "result": "denied",
            "reason": reason,
        }],
    }


def history_node(state: HealthcareState) -> dict:
    settings = load_settings()
    patient_name = state.get("patient_name")
    role = (state.get("role") or "patient_chat").lower()
    active_pid = state.get("patient_id")

    if not patient_name:
        return {
            "history_summary": "No patient was specified; cannot retrieve history.",
            "tool_log": [{
                "node": "history",
                "result": "skipped",
                "reason": "no patient_name",
            }],
        }

    # PHI-scope guard (patient_chat only). A walk-in session has no
    # authenticated identity, so it gets no record reads at all — denied
    # before any lookup so the requested patient's existence never leaks.
    if role not in _UNRESTRICTED_ROLES and not active_pid:
        return _phi_refusal(
            "I can't share another patient's medical history. Medical records "
            "are only available to the patient they belong to — please select "
            "or sign in to your own patient profile to view your history.",
            reason="walk_in_no_active_patient",
            patient_name=patient_name,
        )

    # 1. Look up structured record
    record = find_patient_by_name(patient_name, settings, actor="patient_chat")

    # An authenticated patient_chat caller may only read their OWN record.
    # If the requested name resolves to a different patient (or to nothing),
    # refuse rather than serve or probe.
    if role not in _UNRESTRICTED_ROLES and (
        record is None or record.get("patient_id") != active_pid
    ):
        return _phi_refusal(
            "I can only show your own medical history, not another patient's "
            "record.",
            reason="cross_patient",
            patient_name=patient_name,
        )
    record_block = ""
    if record:
        parts = [f"Name: {record['name']}"]
        if record.get("age"):
            parts.append(f"Age: {record['age']}")
        if record.get("gender"):
            parts.append(f"Gender: {record['gender']}")
        if record.get("summary"):
            parts.append(f"Summary: {record['summary']}")
        if record.get("address"):
            parts.append(f"Address: {record['address']}")
        # 1b. Enrich with FHIR Conditions + recent Observations when available
        clinical = get_patient_clinical_context(record["patient_id"], settings, actor="patient_chat")
        cond_text = condition_summary(clinical.get("conditions") or [])
        if cond_text:
            parts.append(f"Active conditions: {cond_text}")
        obs = clinical.get("observations") or []
        if obs:
            obs_lines = []
            for o in obs:
                val = o.get("value")
                unit = o.get("unit") or ""
                date = o.get("date") or ""
                obs_lines.append(f"  - {o.get('name')}: {val} {unit} ({date})".rstrip())
            parts.append("Recent observations:\n" + "\n".join(obs_lines))
        record_block = "\n".join(parts)
    else:
        record_block = f"No structured record found for {patient_name}."

    # 2. Search FAISS for matching chunks
    pdf_excerpts = ""
    chunks = []
    try:
        chunks = search_index(
            patient_name,
            settings.faiss_index_path,
            settings.faiss_chunks_path,
            top_k=4,
        )
        if chunks:
            pdf_excerpts = "\n\n".join(
                f"[from {c['doc']}, score={c['score']:.2f}]\n{c['text'][:500]}"
                for c in chunks
            )
    except Exception as exc:
        logger.warning("FAISS lookup failed: %s", exc)
        pdf_excerpts = "(No PDF excerpts available)"

    # 3. Ask LLM to summarize
    user_block = (
        f"=== Structured record ===\n{record_block}\n\n"
        f"=== Report excerpts ===\n{pdf_excerpts or '(none retrieved)'}\n\n"
        f"Now summarize this patient's history."
    )

    try:
        summary = chat(
            messages=[
                {"role": "system", "content": HISTORY_SUMMARY_PROMPT},
                {"role": "user", "content": user_block},
            ],
            temperature=0.0,
            max_tokens=300,
        ).strip()
    except LLMUnavailable as exc:
        logger.warning("History summarizer unavailable: %s — returning raw record", exc)
        summary = (
            "(No LLM summary available; configure GROQ_API_KEY or OPENAI_API_KEY)\n\n"
            f"Record:\n{record_block}"
        )

    return {
        "history_summary": summary,
        "tool_log": [{
            "node": "history",
            "tool": "find_patient_by_name+search_index",
            "ehr_backend": settings.ehr_backend,
            "patient_name": patient_name,
            "patient_id": record.get("patient_id") if record else None,
            "record_found": bool(record),
            "pdf_chunks_retrieved": len(chunks),
        }],
    }
