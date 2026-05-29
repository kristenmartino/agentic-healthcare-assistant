"""Tests for the records node's identity resolution (issue #11).

The reported bug: "This is my appointment - AGS-681558 - can you update my
name to Kristen Martino" created a brand-new junk patient record (the new
name was used as the lookup key) and stored "-681558 -" as a phone. These
tests lock in the fix: identity comes from context (active patient or
confirmation number), the new name is a value not a key, the confirmation
number is never parsed as a phone, and an unresolvable update refuses
rather than inserting.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import load_settings
from tools import ehr


@pytest.fixture
def fixture_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture_dir = tmp_path / "fhir_fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "patients.json").write_text(json.dumps([
        {
            "resourceType": "Patient",
            "id": "ramesh-kulkarni",
            "name": [{"given": ["Ramesh"], "family": "Kulkarni", "text": "Ramesh Kulkarni"}],
            "gender": "male",
            "birthDate": "1972",
        },
    ]))
    monkeypatch.setenv("EHR_BACKEND", "fhir_fixture")
    monkeypatch.setenv("FHIR_FIXTURE_DIR", str(fixture_dir))
    ehr.clear_backend_cache()
    yield load_settings()
    ehr.clear_backend_cache()


# ---------- the exact reported bug ----------

def test_update_my_name_with_active_patient_renames_in_place(fixture_settings):
    """End-to-end: with Ramesh active, 'update my name to Kristen Martino'
    renames Ramesh's record — it must NOT insert a new one, and the
    appointment number must NOT land in the phone field."""
    from nodes.records import records_node

    out = records_node({
        "user_input": "This is my appointment - AGS-681558 - can you update my name to Kristen Martino",
        "patient_id": "fhir:ramesh-kulkarni",
    })

    assert "error" not in out
    rc = out["record_change"]
    assert rc["operation"] == "update"
    assert rc["patient_id"] == "fhir:ramesh-kulkarni"
    assert rc["after"]["name"] == "Kristen Martino"
    # Appointment number was NOT captured as a phone.
    assert not rc["after"].get("phone_raw")

    ehr.clear_backend_cache()
    patients = ehr.list_patients(fixture_settings)
    assert len(patients) == 1  # no junk record minted
    assert patients[0]["name"] == "Kristen Martino"
    assert patients[0]["patient_id"] == "fhir:ramesh-kulkarni"


# ---------- field parsing ----------

def test_confirmation_number_not_parsed_as_phone():
    from nodes.records import _parse_fields
    fields = _parse_fields("update my name to Kristen Martino, appointment AGS-681558")
    assert "phone_raw" not in fields


def test_real_phone_still_parsed():
    from nodes.records import _parse_fields
    fields = _parse_fields("set phone +1 (415) 555-0192 please")
    assert fields.get("phone_raw")
    assert sum(c.isdigit() for c in fields["phone_raw"]) >= 10


def test_extract_new_name_value():
    from nodes.records import _extract_new_name_value
    assert _extract_new_name_value("update my name to Kristen Martino") == "Kristen Martino"
    assert _extract_new_name_value("please rename me to Priya Sharma") == "Priya Sharma"
    assert _extract_new_name_value("summary: hypertension") is None


# ---------- refuse rather than insert junk ----------

def test_update_phrasing_without_identity_refuses(fixture_settings):
    """No active patient, no confirmation number → an update request must
    refuse, not create a record. patient_name is the classifier-extracted
    NEW name (the exact input that used to mint a junk record)."""
    from nodes.records import records_node

    out = records_node({
        "user_input": "can you update my name to Kristen Martino",
        "patient_name": "Kristen Martino",  # what the classifier extracts
    })
    assert "error" in out
    assert "record_change" not in out
    # Nothing was written.
    ehr.clear_backend_cache()
    assert all(p["name"] != "Kristen Martino" for p in ehr.list_patients(fixture_settings))


def test_unresolvable_confirmation_refuses(monkeypatch, fixture_settings):
    """The exact production repro: confirmation present but not an active
    booking, classifier extracted the new name → refuse, no junk insert."""
    from nodes import records
    monkeypatch.setattr(
        "tools.appointments.list_all_bookings",
        lambda db_path, **kw: [],
    )
    out = records.records_node({
        "user_input": "This is my appointment - AGS-999999 - update my name to Kristen Martino",
        "patient_name": "Kristen Martino",
    })
    assert "error" in out
    assert "AGS-999999" in out["error"]
    assert "record_change" not in out
    ehr.clear_backend_cache()
    assert all(p["name"] != "Kristen Martino" for p in ehr.list_patients(fixture_settings))


# ---------- confirmation number resolves to the booking's patient ----------

def test_confirmation_number_resolves_and_updates(monkeypatch):
    from nodes import records

    monkeypatch.setattr(
        "tools.appointments.list_all_bookings",
        lambda db_path, **kw: [
            {"confirmation_no": "AGS-681558", "booked_by_patient_id": "fhir:ramesh-kulkarni"},
        ],
    )
    monkeypatch.setattr(
        records, "list_patients",
        lambda settings, actor=None: [
            {"patient_id": "fhir:ramesh-kulkarni", "name": "Ramesh Kulkarni"},
        ],
    )
    captured: dict = {}
    monkeypatch.setattr(
        records, "add_or_update_patient",
        lambda fields, settings, actor=None: captured.update(fields=fields)
        or {"operation": "update", "patient_id": fields.get("patient_id"),
            "after": {"name": fields.get("name")}},
    )

    out = records.records_node({
        "user_input": "This is my appointment AGS-681558 - update my name to Kristen Martino",
    })
    assert out["record_change"]["operation"] == "update"
    assert captured["fields"]["patient_id"] == "fhir:ramesh-kulkarni"
    assert captured["fields"]["name"] == "Kristen Martino"


# ---------- existing flows preserved ----------

def test_registration_still_inserts(fixture_settings):
    from nodes.records import records_node

    # The classifier extracts the new patient's name into patient_name for
    # the records intent; records_node receives it the same way here.
    out = records_node({
        "user_input": "Register a new patient John Smith, age 45, male",
        "patient_name": "John Smith",
    })
    assert "error" not in out
    assert out["record_change"]["operation"] == "insert"
    assert out["record_change"]["after"]["name"] == "John Smith"


def test_classic_update_by_name_preserved(fixture_settings):
    """'update Anjali's record summary: ...' with the name as subject (no
    'name to X' phrasing) resolves the named patient and updates."""
    from nodes.records import records_node

    out = records_node({
        "user_input": "update the record, summary: hypertension",
        "patient_name": "Ramesh Kulkarni",
    })
    assert "error" not in out
    rc = out["record_change"]
    assert rc["operation"] == "update"
    assert rc["patient_id"] == "fhir:ramesh-kulkarni"
