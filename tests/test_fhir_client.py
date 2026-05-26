"""Tests for the FHIR R4 REST client and the FHIR↔internal mapping helpers.

The client is exercised against `responses`-mocked HTTP traffic so the tests
don't require any network. The shape of the mock responses matches the FHIR R4
Bundle/Patient/Condition/Observation specifications.
"""
from __future__ import annotations

import pytest

responses = pytest.importorskip("responses")

from tools.fhir_client import (  # noqa: E402
    FHIRClient,
    FHIRError,
    condition_summary,
    fhir_id_to_patient_id,
    from_internal_patient,
    observation_summary,
    patient_id_to_fhir_id,
    to_internal_patient,
)

BASE = "https://fhir.example.com/baseR4"


def _bundle(*resources: dict) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(resources),
        "entry": [{"resource": r} for r in resources],
    }


@pytest.fixture
def client():
    return FHIRClient(BASE, timeout=1.0)


@responses.activate
def test_search_patients_by_name(client):
    responses.add(
        responses.GET,
        f"{BASE}/Patient",
        json=_bundle({
            "resourceType": "Patient",
            "id": "abc",
            "name": [{"given": ["Anjali"], "family": "Mehra"}],
            "gender": "female",
            "birthDate": "1986",
        }),
        status=200,
    )
    hits = client.search_patients(name="Anjali")
    assert len(hits) == 1
    assert hits[0]["id"] == "abc"


@responses.activate
def test_get_patient_returns_none_on_404(client):
    responses.add(responses.GET, f"{BASE}/Patient/missing", status=404, json={})
    assert client.get_patient("missing") is None


@responses.activate
def test_get_patient_raises_on_other_errors(client):
    responses.add(responses.GET, f"{BASE}/Patient/abc", status=500, json={})
    with pytest.raises(FHIRError):
        client.get_patient("abc")


@responses.activate
def test_get_conditions(client):
    responses.add(
        responses.GET,
        f"{BASE}/Condition",
        json=_bundle({
            "resourceType": "Condition",
            "id": "c1",
            "subject": {"reference": "Patient/abc"},
            "code": {"text": "Hypertension"},
        }),
        status=200,
    )
    conds = client.get_conditions("abc")
    assert conds[0]["code"]["text"] == "Hypertension"


@responses.activate
def test_get_observations_passes_category(client):
    responses.add(
        responses.GET,
        f"{BASE}/Observation",
        json=_bundle(),
        status=200,
    )
    client.get_observations("abc", category="laboratory")
    call = responses.calls[0].request
    assert "category=laboratory" in call.url
    assert "patient=abc" in call.url


@responses.activate
def test_upsert_patient_post_when_no_id(client):
    responses.add(
        responses.POST,
        f"{BASE}/Patient",
        json={"resourceType": "Patient", "id": "server-assigned", "name": [{"text": "X Y"}]},
        status=201,
    )
    stored = client.upsert_patient({"name": [{"text": "X Y"}]})
    assert stored["id"] == "server-assigned"


@responses.activate
def test_upsert_patient_put_when_id_present(client):
    responses.add(
        responses.PUT,
        f"{BASE}/Patient/abc",
        json={"resourceType": "Patient", "id": "abc"},
        status=200,
    )
    stored = client.upsert_patient({"id": "abc", "name": [{"text": "A B"}]})
    assert stored["id"] == "abc"


# ---------- mapping helpers ----------


def test_to_internal_patient_minimal():
    p = to_internal_patient({
        "resourceType": "Patient",
        "id": "abc",
        "name": [{"given": ["Anjali"], "family": "Mehra"}],
        "gender": "female",
    })
    assert p["patient_id"] == "fhir:abc"
    assert p["name"] == "Anjali Mehra"
    assert p["gender"] == "Female"
    assert p["fhir_id"] == "abc"


def test_to_internal_patient_telecom_and_address():
    p = to_internal_patient({
        "resourceType": "Patient",
        "id": "abc",
        "name": [{"text": "Test"}],
        "telecom": [
            {"system": "phone", "value": "+1-555-0100"},
            {"system": "email", "value": "a@b.com"},
        ],
        "address": [{"text": "123 Main St"}],
        "birthDate": "1990-01-01",
    })
    assert p["phone_raw"] == "+1-555-0100"
    assert p["email"] == "a@b.com"
    assert p["address"] == "123 Main St"
    # birthDate 1990 → 2026 means age ~36 (current date in this sandbox)
    assert p["age"] is not None and p["age"] >= 30


def test_from_internal_patient_round_trip():
    resource = from_internal_patient({
        "name": "Jane Doe",
        "gender": "Female",
        "age": 40,
        "phone_raw": "+1-555-9999",
        "email": "jane@example.com",
        "address": "1 First Ave",
    })
    assert resource["resourceType"] == "Patient"
    assert resource["gender"] == "female"
    assert any(t["system"] == "phone" for t in resource["telecom"])
    assert resource["name"][0]["family"] == "Doe"


def test_from_internal_patient_carries_id():
    resource = from_internal_patient({"name": "X Y", "patient_id": "fhir:abc"})
    assert resource["id"] == "abc"


def test_condition_summary_picks_text_or_coding():
    out = condition_summary([
        {"code": {"text": "Hypertension"}},
        {"code": {"coding": [{"display": "Type 2 diabetes"}]}},
    ])
    assert "Hypertension" in out and "diabetes" in out


def test_observation_summary_normalizes_value_units():
    out = observation_summary([
        {
            "code": {"text": "HbA1c"},
            "effectiveDateTime": "2026-04-12",
            "valueQuantity": {"value": 7.4, "unit": "%"},
        }
    ])
    assert out[0]["name"] == "HbA1c"
    assert out[0]["value"] == 7.4
    assert out[0]["unit"] == "%"


def test_id_roundtrip_helpers():
    assert fhir_id_to_patient_id("abc") == "fhir:abc"
    assert patient_id_to_fhir_id("fhir:abc") == "abc"
    assert patient_id_to_fhir_id("sqlite-id") is None
