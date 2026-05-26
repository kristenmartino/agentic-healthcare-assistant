"""Tests for the PHI audit log."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools import audit
from tools.audit import audit_summary, log_access, query_audit


@pytest.fixture
def audit_db(tmp_path: Path) -> str:
    return str(tmp_path / "audit.sqlite")


def test_log_creates_schema_on_first_write(audit_db):
    assert not Path(audit_db).exists()
    log_access("test", "ehr.read", "Patient", "abc",
               patient_id="abc", db_path=audit_db)
    assert Path(audit_db).exists()
    rows = query_audit(db_path=audit_db)
    assert len(rows) == 1
    assert rows[0]["actor"] == "test"
    assert rows[0]["action"] == "ehr.read"


def test_log_round_trips_json_details(audit_db):
    log_access("test", "ehr.write", "Patient", "abc",
               patient_id="abc",
               details={"fields_set": ["age", "name"], "n": 2},
               db_path=audit_db)
    rows = query_audit(db_path=audit_db)
    assert rows[0]["details"] == {"fields_set": ["age", "name"], "n": 2}


def test_query_filters(audit_db):
    log_access("patient_chat", "ehr.read", "Patient", "p1", patient_id="p1", db_path=audit_db)
    log_access("patient_chat", "ehr.write", "Patient", "p1", patient_id="p1", db_path=audit_db)
    log_access("doctor_view", "ehr.read", "Patient", "p2", patient_id="p2", db_path=audit_db)

    by_patient = query_audit(patient_id="p1", db_path=audit_db)
    assert len(by_patient) == 2

    by_action = query_audit(action_prefix="ehr.read", db_path=audit_db)
    assert len(by_action) == 2
    assert all(r["action"] == "ehr.read" for r in by_action)

    by_actor = query_audit(actor="doctor_view", db_path=audit_db)
    assert len(by_actor) == 1
    assert by_actor[0]["patient_id"] == "p2"


def test_log_swallows_errors(monkeypatch, caplog):
    """An audit write failure must not raise — that would break the user's request."""
    # Point at a path under a missing parent that we then make un-creatable
    monkeypatch.setattr(audit, "_connect", _broken_connect)
    # Should not raise
    log_access("test", "anything", "Patient", "id", db_path="/nonexistent/audit.sqlite")
    assert "Audit log write failed" in caplog.text


def _broken_connect(*args, **kwargs):  # pragma: no cover - helper
    raise RuntimeError("disk full")


def test_summary_counts(audit_db):
    log_access("a", "x.1", "Patient", "1", db_path=audit_db)
    log_access("a", "x.2", "Patient", "1", db_path=audit_db)
    log_access("b", "x.1", "Patient", "2", db_path=audit_db)
    s = audit_summary(db_path=audit_db)
    assert s["total"] == 3
    assert s["by_action"]["x.1"] == 2
    assert s["by_actor"]["a"] == 2


def test_summary_empty_when_no_db(tmp_path):
    path = str(tmp_path / "never.sqlite")
    assert audit_summary(db_path=path) == {"total": 0, "by_action": {}, "by_actor": {}}
    assert query_audit(db_path=path) == []
