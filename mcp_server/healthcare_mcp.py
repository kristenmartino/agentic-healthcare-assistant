"""MCP server exposing the Healthcare Assistant's tools.

Lets other MCP clients (Claude Desktop, custom Claude API agents, etc.) call
the same booking / records / history / medical-search functions the LangGraph
agent uses internally.

This is the **stretch goal** referenced in the WRITEUP — the capstone problem
statement does not require MCP, but the precedent in NewsGenie's
`mcp_server/news_mcp_server.py` makes it natural to expose the same tools
through MCP for cross-system reuse.

Run:
    python -m mcp_server.healthcare_mcp           # stdio transport (for Claude Desktop)
    python -m mcp_server.healthcare_mcp --http    # HTTP transport (for HTTP clients)

Tool list (every one returns JSON-serialisable dicts):
- book_appointment(patient_name, specialty, preferred_date?)
- list_doctors(specialty?)
- find_patient(name)
- list_patients()
- add_or_update_patient(name, age?, gender?, phone?, email?, address?, summary?)
- get_history(patient_name)
- medical_search(query, top_k=4)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_settings
from tools.appointments import (
    book_appointment as _book,
)
from tools.appointments import (
    get_doctors_for_dashboard,
    list_doctors_for_specialty,
)
from tools.audit import log_access, query_audit
from tools.ehr import (
    add_or_update_patient as _upsert_patient,
)
from tools.ehr import (
    find_patient_by_name,
)
from tools.ehr import (
    list_patients as _list_patients,
)
from tools.medical_search import medical_search as _medical_search
from tools.vector_index import search_index

logger = logging.getLogger(__name__)


def _try_import_mcp():
    """Try the FastMCP API; fall back to a noop server class for testing."""
    try:
        from mcp.server.fastmcp import FastMCP
        return FastMCP
    except ImportError:
        logger.warning(
            "mcp[cli] not installed — running in dry-run mode. "
            "Install with: pip install 'mcp[cli]'"
        )
        return None


def _build_settings():
    """Cache settings at server start; the MCP process is long-lived."""
    return load_settings()


# ---------- Tool implementations (used by both real MCP and dry-run) ----------

def tool_book_appointment(
    patient_name: str,
    specialty: str,
    preferred_date: str | None = None,
) -> dict:
    """Book the earliest available appointment slot for the given specialty.

    Args:
        patient_name: Name of the patient to book for.
        specialty: Medical specialty (cardiology / nephrology / general_practice / etc.).
        preferred_date: Optional ISO date (YYYY-MM-DD); booking will be on or after this date.

    Returns:
        Booking record with doctor_name, specialty, start_time, end_time,
        confirmation_no, slot_id.
    """
    s = _build_settings()
    # Look up patient_id from the EHR; fall back to a synthetic walk-in ID.
    patient = find_patient_by_name(patient_name, s, actor="mcp")
    patient_id = patient["patient_id"] if patient else f"walkin-{abs(hash(patient_name)) % 10**8:08d}"
    appointment = _book(
        s.appointments_db_path,
        patient_id=patient_id,
        patient_name=patient_name,
        specialty=specialty,
        preferred_date=preferred_date,
    )
    log_access(
        "mcp", "appointment.book", "Appointment",
        str(appointment.get("slot_id")),
        patient_id=patient_id,
        details={
            "specialty": specialty,
            "doctor_name": appointment.get("doctor_name"),
            "confirmation_no": appointment.get("confirmation_no"),
        },
    )
    return appointment


def tool_list_doctors(specialty: str | None = None) -> list[dict]:
    """List doctors. Filter by specialty if provided."""
    s = _build_settings()
    if specialty:
        return list_doctors_for_specialty(s.appointments_db_path, specialty)
    return get_doctors_for_dashboard(s.appointments_db_path)


def tool_find_patient(name: str) -> dict | None:
    """Find a patient by case-insensitive name match. Returns first hit or None."""
    s = _build_settings()
    return find_patient_by_name(name, s, actor="mcp")


def tool_list_patients() -> list[dict]:
    """List all patients (id, name, age, gender, summary)."""
    s = _build_settings()
    return _list_patients(s, actor="mcp")


def tool_upsert_patient(
    name: str,
    age: int | None = None,
    gender: str | None = None,
    phone: str | None = None,
    email: str | None = None,
    address: str | None = None,
    summary: str | None = None,
) -> dict:
    """Add a new patient or update an existing one.

    Returns: {operation: 'insert'|'update', patient_id, before, after}.
    """
    s = _build_settings()
    return _upsert_patient(
        {
            "name": name,
            "age": age,
            "gender": gender,
            "phone_raw": phone,
            "email": email,
            "address": address,
            "summary": summary,
        },
        s,
        actor="mcp",
    )


def tool_get_history(patient_name: str) -> dict:
    """Get a patient's structured record + matching PDF chunks from the FAISS index.

    Returns: {record, chunks, has_record}.
    Note: this is the *raw* RAG retrieval — no LLM summarization. Callers can
    summarize themselves with their own model.
    """
    s = _build_settings()
    record = find_patient_by_name(patient_name, s, actor="mcp")
    chunks = []
    try:
        chunks = search_index(
            patient_name,
            s.faiss_index_path,
            s.faiss_chunks_path,
            top_k=4,
        )
    except Exception as exc:
        logger.warning("History RAG failed: %s", exc)
    log_access(
        "mcp", "history.retrieve", "Patient",
        record["patient_id"] if record else None,
        patient_id=record["patient_id"] if record else None,
        details={"chunks": len(chunks), "search_name": patient_name},
    )
    return {
        "record": record,
        "chunks": chunks,
        "has_record": record is not None,
    }


def tool_medical_search(query: str, top_k: int = 4) -> list[dict]:
    """Search trusted medical sources (MedlinePlus, WHO, CDC, NIH, Mayo, NHS).

    Returns a list of {title, snippet, url, source}.
    """
    s = _build_settings()
    results = _medical_search(query, top_k=top_k, tavily_api_key=s.tavily_api_key)
    log_access(
        "mcp", "medical_search.query", "WebSearch", None,
        details={"query": query, "top_k": top_k, "results": len(results)},
    )
    return results


def tool_get_audit_log(
    patient_id: str | None = None,
    action_prefix: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Read the PHI audit log.

    Args:
        patient_id: filter to one patient (e.g. 'fhir:abc').
        action_prefix: filter by action dotted-prefix (e.g. 'ehr.', 'appointment.').
        limit: max events to return (default 50, max 500).

    Returns the events ordered newest-first.
    """
    limit = max(1, min(int(limit), 500))
    events = query_audit(
        patient_id=patient_id,
        action_prefix=action_prefix,
        limit=limit,
    )
    log_access("mcp", "audit.read", "AuditLog", None,
               details={"returned": len(events), "limit": limit,
                        "filter_patient": patient_id,
                        "filter_action_prefix": action_prefix})
    return events


TOOLS = {
    "book_appointment": tool_book_appointment,
    "list_doctors": tool_list_doctors,
    "find_patient": tool_find_patient,
    "list_patients": tool_list_patients,
    "upsert_patient": tool_upsert_patient,
    "get_history": tool_get_history,
    "medical_search": tool_medical_search,
    "get_audit_log": tool_get_audit_log,
}


# ---------- Server bootstrap ----------

def build_server():
    """Build the FastMCP server and register all tools."""
    FastMCP = _try_import_mcp()
    if FastMCP is None:
        return None

    mcp = FastMCP("healthcare-assistant")

    # Register each tool with FastMCP. The decorator pattern is the standard
    # idiom; we use the explicit registration form for clarity.
    for name, fn in TOOLS.items():
        mcp.tool(name=name, description=(fn.__doc__ or "").split("\n")[0])(fn)

    return mcp


def dry_run() -> int:
    """Test the tool implementations directly without MCP — useful for CI."""
    print("=" * 60)
    print(" Healthcare MCP — dry-run smoke test")
    print(" (set up because mcp[cli] is not installed or --dry-run was passed)")
    print("=" * 60)

    print("\n[1] tool_list_doctors(specialty='nephrology')")
    print(json.dumps(tool_list_doctors("nephrology"), default=str, indent=2)[:500])

    print("\n[2] tool_find_patient('Anjali Mehra')")
    print(json.dumps(tool_find_patient("Anjali Mehra"), default=str, indent=2)[:500])

    print("\n[3] tool_get_history('Ramesh Kulkarni')")
    h = tool_get_history("Ramesh Kulkarni")
    print(f"  has_record={h['has_record']}, chunks_returned={len(h['chunks'])}")

    print("\n[4] tool_medical_search('symptoms of pneumonia')")
    r = tool_medical_search("symptoms of pneumonia", top_k=2)
    print(f"  results={len(r)}, first_title={r[0].get('title','?') if r else '—'}")

    print("\n[5] tool_book_appointment('Test Patient', 'general_practice')")
    a = tool_book_appointment("Test Patient", "general_practice")
    print(f"  confirmation_no={a.get('confirmation_no')}, doctor={a.get('doctor_name')}")

    print("\n" + "=" * 60)
    print(" Dry-run complete. All tools callable without MCP runtime.")
    print(" To run as a real MCP server, install mcp: pip install 'mcp[cli]'")
    print("=" * 60)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if "--dry-run" in sys.argv:
        return dry_run()

    server = build_server()
    if server is None:
        # mcp[cli] not installed — fall back to dry-run
        return dry_run()

    transport = "http" if "--http" in sys.argv else "stdio"
    print(f"Healthcare MCP server starting (transport={transport}) ...", file=sys.stderr)

    if transport == "http":
        # FastMCP HTTP defaults to port 8000
        try:
            server.run(transport="streamable-http")
        except TypeError:
            # Older FastMCP versions: run() takes no transport kwarg
            server.run()
    else:
        server.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
