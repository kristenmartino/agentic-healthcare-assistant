"""PHI access audit log.

Every read or write of patient-identifiable data should produce one audit
event. HIPAA's Security Rule (45 CFR 164.312(b)) requires the ability to
"examine activity in information systems that contain or use ePHI". This
module is the lightest credible implementation of that requirement: an
append-only SQLite table with the actor, action, resource type/id, patient
id, timestamp, and a JSON details blob.

The store is intentionally separate from the EHR (`data/audit.sqlite` vs
`data/ehr.sqlite`) — in real deployments the audit log lives in a different
trust boundary than the data it audits, so a compromise of the EHR shouldn't
silently erase the audit trail.

Usage:
    from tools.audit import log_access
    log_access("patient_chat", "ehr.read", "Patient", "fhir:abc",
               patient_id="fhir:abc", details={"node": "history"})
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from config import load_settings

logger = logging.getLogger(__name__)

# Single write lock so concurrent Streamlit threads can't corrupt the WAL.
_write_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    patient_id TEXT,
    outcome TEXT NOT NULL DEFAULT 'success',
    details TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_log(patient_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action, ts DESC);
"""


@contextmanager
def _connect(db_path: str) -> Iterator[sqlite3.Connection]:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_schema(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)


def log_access(
    actor: str,
    action: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    *,
    patient_id: str | None = None,
    outcome: str = "success",
    details: dict[str, Any] | None = None,
    db_path: str | None = None,
) -> None:
    """Write one audit event.

    Best-effort: on failure, log a warning and return — we never raise from
    here, because an audit failure should not break a user-facing tool call.
    The risk model is "missing entry, not crashed app".

    Args:
        actor: who initiated the access (e.g. "patient_chat", "doctor_view",
            "mcp:claude_desktop", "system:eval").
        action: dotted noun.verb (e.g. "ehr.read", "ehr.write", "appointment.book",
            "history.retrieve", "medical_search.query").
        resource_type: FHIR resource type or internal resource family
            ("Patient", "Appointment", "Condition", "Slot", "WebSearch").
        resource_id: stable id of the resource accessed.
        patient_id: when the access is patient-scoped, the patient_id for
            indexing. May equal resource_id.
        outcome: "success", "denied", "error".
        details: free-form dict (logged as JSON). Use this for query terms,
            field-level breakdowns, etc.
        db_path: override; defaults to settings.audit_db_path.
    """
    if db_path is None:
        db_path = load_settings().audit_db_path

    payload = json.dumps(details or {}, default=str, separators=(",", ":"))
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    try:
        with _write_lock, _connect(db_path) as conn:
            _ensure_schema_inline(conn)
            conn.execute(
                """
                INSERT INTO audit_log
                    (ts, actor, action, resource_type, resource_id,
                     patient_id, outcome, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, actor, action, resource_type, resource_id,
                 patient_id, outcome, payload),
            )
    except Exception as exc:
        logger.warning("Audit log write failed (%s); event dropped: actor=%s action=%s",
                       exc, actor, action)


def _ensure_schema_inline(conn: sqlite3.Connection) -> None:
    """Schema bootstrap inside an open connection. Idempotent."""
    conn.executescript(SCHEMA)


def query_audit(
    *,
    patient_id: str | None = None,
    action_prefix: str | None = None,
    actor: str | None = None,
    since: str | None = None,
    limit: int = 100,
    db_path: str | None = None,
) -> list[dict]:
    """Read recent audit events. Used by the Audit Log UI page."""
    if db_path is None:
        db_path = load_settings().audit_db_path
    if not Path(db_path).exists():
        return []

    clauses: list[str] = []
    params: list[Any] = []
    if patient_id:
        clauses.append("patient_id = ?")
        params.append(patient_id)
    if action_prefix:
        clauses.append("action LIKE ?")
        params.append(f"{action_prefix}%")
    if actor:
        clauses.append("actor = ?")
        params.append(actor)
    if since:
        clauses.append("ts >= ?")
        params.append(since)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    with _connect(db_path) as conn:
        _ensure_schema_inline(conn)
        rows = conn.execute(
            f"""
            SELECT id, ts, actor, action, resource_type, resource_id,
                   patient_id, outcome, details
              FROM audit_log
              {where}
             ORDER BY ts DESC
             LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        try:
            d["details"] = json.loads(d["details"]) if d["details"] else {}
        except json.JSONDecodeError:
            pass
        out.append(d)
    return out


def audit_summary(db_path: str | None = None) -> dict:
    """Counts by action — for the Audit page header strip."""
    if db_path is None:
        db_path = load_settings().audit_db_path
    if not Path(db_path).exists():
        return {"total": 0, "by_action": {}, "by_actor": {}}

    with _connect(db_path) as conn:
        _ensure_schema_inline(conn)
        total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        by_action = {
            row["action"]: row["n"]
            for row in conn.execute(
                "SELECT action, COUNT(*) AS n FROM audit_log GROUP BY action ORDER BY n DESC"
            )
        }
        by_actor = {
            row["actor"]: row["n"]
            for row in conn.execute(
                "SELECT actor, COUNT(*) AS n FROM audit_log GROUP BY actor ORDER BY n DESC"
            )
        }
    return {"total": total, "by_action": by_action, "by_actor": by_actor}
