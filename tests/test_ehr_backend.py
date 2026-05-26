"""Tests for the EHR abstraction across all three backends.

The Protocol guarantee is: every backend must satisfy the same list/find/upsert
contract. These tests assert that contract holds for the fixture backend (the
sqlite + live FHIR backends are exercised by their own modules' tests).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import load_settings
from tools import ehr


@pytest.fixture
def fixture_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the EHR at a fresh temp fixture dir with a couple of patients."""
    fixture_dir = tmp_path / "fhir_fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "patients.json").write_text(json.dumps([
        {
            "resourceType": "Patient",
            "id": "anjali-mehra",
            "name": [{"given": ["Anjali"], "family": "Mehra", "text": "Anjali Mehra"}],
            "gender": "female",
            "birthDate": "1986",
        },
        {
            "resourceType": "Patient",
            "id": "david-thompson",
            "name": [{"given": ["David"], "family": "Thompson", "text": "David Thompson"}],
            "gender": "male",
            "birthDate": "1968",
        },
    ]))
    (fixture_dir / "conditions.json").write_text(json.dumps([
        {
            "resourceType": "Condition",
            "subject": {"reference": "Patient/anjali-mehra"},
            "code": {"text": "Type 2 diabetes mellitus"},
        }
    ]))
    (fixture_dir / "observations.json").write_text(json.dumps([
        {
            "resourceType": "Observation",
            "subject": {"reference": "Patient/anjali-mehra"},
            "code": {"text": "HbA1c"},
            "effectiveDateTime": "2026-04-12",
            "valueQuantity": {"value": 7.4, "unit": "%"},
        }
    ]))

    monkeypatch.setenv("EHR_BACKEND", "fhir_fixture")
    monkeypatch.setenv("FHIR_FIXTURE_DIR", str(fixture_dir))
    ehr.clear_backend_cache()
    s = load_settings()
    yield s
    ehr.clear_backend_cache()


def test_list_patients_returns_internal_shape(fixture_settings):
    patients = ehr.list_patients(fixture_settings)
    assert len(patients) == 2
    by_name = {p["name"]: p for p in patients}
    assert "Anjali Mehra" in by_name
    assert by_name["Anjali Mehra"]["patient_id"] == "fhir:anjali-mehra"
    # Summary auto-populated from Condition.code.text
    assert "diabetes" in (by_name["Anjali Mehra"]["summary"] or "").lower()


def test_find_patient_by_name_is_case_insensitive(fixture_settings):
    hit = ehr.find_patient_by_name("anjali", fixture_settings)
    assert hit is not None
    assert hit["name"] == "Anjali Mehra"


def test_find_patient_returns_none_for_missing(fixture_settings):
    assert ehr.find_patient_by_name("Nonexistent Person", fixture_settings) is None


def test_add_new_patient_inserts_to_overlay(fixture_settings, tmp_path):
    result = ehr.add_or_update_patient(
        {"name": "Jane Doe", "age": 40, "gender": "Female", "summary": "hypertension"},
        fixture_settings,
    )
    assert result["operation"] == "insert"
    assert result["after"]["name"] == "Jane Doe"
    # Round-trip: list_patients should now contain Jane
    ehr.clear_backend_cache()
    patients = ehr.list_patients(fixture_settings)
    assert any(p["name"] == "Jane Doe" for p in patients)


def test_update_existing_patient(fixture_settings):
    result = ehr.add_or_update_patient(
        {"name": "Anjali Mehra", "summary": "diabetes — well controlled"},
        fixture_settings,
    )
    assert result["operation"] == "update"
    assert result["before"] is not None


def test_clinical_context_returns_conditions_and_observations(fixture_settings):
    ctx = ehr.get_patient_clinical_context("fhir:anjali-mehra", fixture_settings)
    assert len(ctx["conditions"]) == 1
    assert len(ctx["observations"]) == 1
    assert ctx["observations"][0]["name"] == "HbA1c"


def test_clinical_context_non_fhir_id_returns_empty(fixture_settings):
    ctx = ehr.get_patient_clinical_context("some-sqlite-id", fixture_settings)
    assert ctx == {"conditions": [], "observations": []}
