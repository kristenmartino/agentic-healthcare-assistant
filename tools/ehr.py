"""EHR abstraction — dispatches `list_patients`, `find_patient_by_name`, and
`add_or_update_patient` to one of three backends based on `EHR_BACKEND`:

- `sqlite` (default): the existing records.xlsx → SQLite pipeline. Suitable
  for the course-end demo + offline use.
- `fhir`: a live FHIR R4 server (HAPI, Microsoft FHIR, AWS HealthLake, Epic
  sandbox). Set `FHIR_BASE_URL` to the base path.
- `fhir_fixture`: reads pre-baked FHIR resource JSON from `data/fhir_fixtures/`
  for offline development against the FHIR shape (no network).

All three return the same internal patient dict shape:
    {patient_id, name, age, gender, phone_raw, phone_normalized,
     email, address, summary, [fhir_id], [fhir_resource]}

This is the only module nodes/MCP should import for patient access. They
must NOT import `tools.ehr_db` (sqlite-only) or `tools.fhir_client` (fhir-only)
directly — that would defeat the point of the abstraction.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Protocol

from config import Settings, load_settings

logger = logging.getLogger(__name__)


# ---------- Protocol ----------

class EHRBackend(Protocol):
    """Every backend implements these three operations."""

    def list_patients(self) -> list[dict]: ...
    def find_patient_by_name(self, name: str) -> Optional[dict]: ...
    def add_or_update_patient(self, fields: dict) -> dict: ...
    def get_patient_clinical_context(self, patient_id: str) -> dict:
        """Optional: conditions, observations. Default: empty dict."""
        ...


# ---------- SQLite backend ----------

class _SqliteBackend:
    """Wraps the existing tools.ehr_db module so it satisfies the Protocol."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def list_patients(self) -> list[dict]:
        from tools.ehr_db import list_patients as _list
        return _list(self.db_path)

    def find_patient_by_name(self, name: str) -> Optional[dict]:
        from tools.ehr_db import find_patient_by_name as _find
        return _find(self.db_path, name)

    def add_or_update_patient(self, fields: dict) -> dict:
        from tools.ehr_db import add_or_update_patient as _upsert
        return _upsert(self.db_path, fields)

    def get_patient_clinical_context(self, patient_id: str) -> dict:
        # SQLite backend stores the freeform summary on the patient row;
        # there are no separate Condition/Observation tables.
        return {"conditions": [], "observations": []}


# ---------- FHIR backend (live server) ----------

class _FhirBackend:
    """Backs onto a live FHIR R4 server via `tools.fhir_client.FHIRClient`."""

    def __init__(self, base_url: str, timeout: float):
        from tools.fhir_client import FHIRClient
        self._client = FHIRClient(base_url, timeout=timeout)

    def list_patients(self) -> list[dict]:
        from tools.fhir_client import condition_summary, to_internal_patient
        out: list[dict] = []
        for resource in self._client.list_patients(count=50):
            patient = to_internal_patient(resource)
            # Best-effort enrich with primary condition summary
            try:
                conditions = self._client.get_conditions(resource["id"], count=5)
                patient["summary"] = condition_summary(conditions) or None
            except Exception as exc:
                logger.debug("Could not fetch conditions for %s: %s", resource.get("id"), exc)
            out.append(patient)
        return out

    def find_patient_by_name(self, name: str) -> Optional[dict]:
        from tools.fhir_client import condition_summary, to_internal_patient
        hits = self._client.search_patients(name=name, count=1)
        if not hits:
            return None
        patient = to_internal_patient(hits[0])
        try:
            conditions = self._client.get_conditions(hits[0]["id"], count=5)
            patient["summary"] = condition_summary(conditions) or None
        except Exception as exc:
            logger.debug("Could not fetch conditions for %s: %s", hits[0].get("id"), exc)
        return patient

    def add_or_update_patient(self, fields: dict) -> dict:
        from tools.fhir_client import (
            fhir_id_to_patient_id,
            from_internal_patient,
            to_internal_patient,
        )
        before: Optional[dict] = None
        if fields.get("name"):
            before = self.find_patient_by_name(fields["name"])
        # Carry forward the existing FHIR id if we matched
        if before and before.get("fhir_id") and not fields.get("patient_id"):
            fields = {**fields, "patient_id": fhir_id_to_patient_id(before["fhir_id"])}

        resource = from_internal_patient(fields)
        stored = self._client.upsert_patient(resource)
        after = to_internal_patient(stored)
        return {
            "operation": "update" if before else "insert",
            "patient_id": after["patient_id"],
            "before": before,
            "after": after,
        }

    def get_patient_clinical_context(self, patient_id: str) -> dict:
        from tools.fhir_client import (
            observation_summary,
            patient_id_to_fhir_id,
        )
        fhir_id = patient_id_to_fhir_id(patient_id)
        if not fhir_id:
            return {"conditions": [], "observations": []}
        try:
            conditions = self._client.get_conditions(fhir_id, count=10)
        except Exception as exc:
            logger.warning("Conditions fetch failed for %s: %s", fhir_id, exc)
            conditions = []
        try:
            observations = self._client.get_observations(fhir_id, count=10)
        except Exception as exc:
            logger.warning("Observations fetch failed for %s: %s", fhir_id, exc)
            observations = []
        return {
            "conditions": conditions,  # raw FHIR for downstream summarization
            "observations": observation_summary(observations),
        }


# ---------- FHIR fixture backend (offline) ----------

class _FhirFixtureBackend:
    """Reads pre-baked FHIR JSON from a local directory.

    Expected layout:
        fhir_fixtures/
            patients.json       — list of Patient resources
            conditions.json     — list of Condition resources
            observations.json   — list of Observation resources

    Writes (`add_or_update_patient`) append to a `patients_writes.json` overlay
    so the fixture files themselves are read-only and survive re-seeds.
    """

    def __init__(self, fixture_dir: str):
        self.fixture_dir = Path(fixture_dir)
        self._patients_cache: Optional[list[dict]] = None
        self._writes_path = self.fixture_dir / "patients_writes.json"

    def _load(self, filename: str) -> list[dict]:
        path = self.fixture_dir / filename
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("Bad fixture %s: %s", path, exc)
            return []

    def _all_patients_raw(self) -> list[dict]:
        base = self._load("patients.json")
        if self._writes_path.exists():
            overlay = self._load("patients_writes.json")
        else:
            overlay = []
        # Overlay wins on matching id
        by_id: dict[str, dict] = {r["id"]: r for r in base if r.get("id")}
        for r in overlay:
            if r.get("id"):
                by_id[r["id"]] = r
        return list(by_id.values())

    def list_patients(self) -> list[dict]:
        from tools.fhir_client import condition_summary, to_internal_patient
        conditions_by_patient = self._conditions_by_patient()
        out: list[dict] = []
        for resource in self._all_patients_raw():
            patient = to_internal_patient(resource)
            patient["summary"] = condition_summary(
                conditions_by_patient.get(resource["id"], [])
            ) or None
            out.append(patient)
        return out

    def find_patient_by_name(self, name: str) -> Optional[dict]:
        if not name:
            return None
        needle = name.strip().lower()
        for patient in self.list_patients():
            if needle in (patient.get("name") or "").lower():
                return patient
        return None

    def add_or_update_patient(self, fields: dict) -> dict:
        from tools.fhir_client import (
            from_internal_patient,
            synthetic_fhir_id,
            to_internal_patient,
        )
        before = None
        if fields.get("name"):
            before = self.find_patient_by_name(fields["name"])
        if before and before.get("fhir_id"):
            fields = {**fields, "patient_id": f"fhir:{before['fhir_id']}"}
        else:
            fields = {
                **fields,
                "patient_id": f"fhir:{synthetic_fhir_id(fields.get('name', ''))}",
            }
        resource = from_internal_patient(fields)

        # Persist into the overlay
        self.fixture_dir.mkdir(parents=True, exist_ok=True)
        overlay = self._load("patients_writes.json")
        overlay = [r for r in overlay if r.get("id") != resource.get("id")]
        overlay.append(resource)
        self._writes_path.write_text(json.dumps(overlay, indent=2))

        after = to_internal_patient(resource)
        return {
            "operation": "update" if before else "insert",
            "patient_id": after["patient_id"],
            "before": before,
            "after": after,
        }

    def _conditions_by_patient(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for c in self._load("conditions.json"):
            ref = (c.get("subject") or {}).get("reference", "")
            # "Patient/abc" → "abc"
            pid = ref.split("/", 1)[1] if "/" in ref else ref
            out.setdefault(pid, []).append(c)
        return out

    def _observations_by_patient(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for o in self._load("observations.json"):
            ref = (o.get("subject") or {}).get("reference", "")
            pid = ref.split("/", 1)[1] if "/" in ref else ref
            out.setdefault(pid, []).append(o)
        return out

    def get_patient_clinical_context(self, patient_id: str) -> dict:
        from tools.fhir_client import observation_summary, patient_id_to_fhir_id
        fhir_id = patient_id_to_fhir_id(patient_id)
        if not fhir_id:
            return {"conditions": [], "observations": []}
        return {
            "conditions": self._conditions_by_patient().get(fhir_id, []),
            "observations": observation_summary(
                self._observations_by_patient().get(fhir_id, [])
            ),
        }


# ---------- factory ----------

@lru_cache(maxsize=4)
def _backend_for(backend: str, db_path: str, fhir_url: str,
                  fhir_dir: str, fhir_timeout: float) -> EHRBackend:
    if backend == "fhir":
        logger.info("EHR backend: FHIR R4 server (%s)", fhir_url)
        return _FhirBackend(fhir_url, fhir_timeout)
    if backend == "fhir_fixture":
        logger.info("EHR backend: FHIR fixture (%s)", fhir_dir)
        return _FhirFixtureBackend(fhir_dir)
    logger.info("EHR backend: SQLite (%s)", db_path)
    return _SqliteBackend(db_path)


def get_backend(settings: Optional[Settings] = None) -> EHRBackend:
    """Return the configured backend (cached on settings tuple)."""
    s = settings or load_settings()
    return _backend_for(
        s.ehr_backend,
        s.ehr_db_path,
        s.fhir_base_url,
        s.fhir_fixture_dir,
        s.fhir_timeout_seconds,
    )


# ---------- convenience pass-throughs ----------
#
# The `actor` arg threads through to the audit log so calls originating from
# the patient chat, doctor view, MCP clients, and eval runs are distinguishable
# in the audit table.

def list_patients(
    settings: Optional[Settings] = None,
    *,
    actor: str = "system",
) -> list[dict]:
    from tools.audit import log_access
    rows = get_backend(settings).list_patients()
    log_access(actor, "ehr.list", "Patient", None,
               details={"count": len(rows), "backend": (settings or load_settings()).ehr_backend})
    return rows


def find_patient_by_name(
    name: str,
    settings: Optional[Settings] = None,
    *,
    actor: str = "system",
) -> Optional[dict]:
    from tools.audit import log_access
    record = get_backend(settings).find_patient_by_name(name)
    log_access(
        actor, "ehr.read", "Patient",
        record["patient_id"] if record else None,
        patient_id=record["patient_id"] if record else None,
        outcome="success" if record else "not_found",
        details={"search_name": name,
                 "backend": (settings or load_settings()).ehr_backend},
    )
    return record


def add_or_update_patient(
    fields: dict,
    settings: Optional[Settings] = None,
    *,
    actor: str = "system",
) -> dict:
    from tools.audit import log_access
    try:
        result = get_backend(settings).add_or_update_patient(fields)
    except Exception as exc:
        log_access(
            actor, "ehr.write", "Patient", None,
            patient_id=fields.get("patient_id"),
            outcome="error",
            details={"error": str(exc), "fields_set": list(fields.keys())},
        )
        raise
    log_access(
        actor, "ehr.write", "Patient", result.get("patient_id"),
        patient_id=result.get("patient_id"),
        details={
            "operation": result.get("operation"),
            "fields_set": list(fields.keys()),
            "backend": (settings or load_settings()).ehr_backend,
        },
    )
    return result


def get_patient_clinical_context(
    patient_id: str,
    settings: Optional[Settings] = None,
    *,
    actor: str = "system",
) -> dict:
    from tools.audit import log_access
    ctx = get_backend(settings).get_patient_clinical_context(patient_id)
    log_access(
        actor, "ehr.read", "Condition+Observation", patient_id,
        patient_id=patient_id,
        details={
            "conditions_returned": len(ctx.get("conditions") or []),
            "observations_returned": len(ctx.get("observations") or []),
        },
    )
    return ctx


def clear_backend_cache() -> None:
    """Invalidate the LRU. Call from tests that switch backends mid-process."""
    _backend_for.cache_clear()
