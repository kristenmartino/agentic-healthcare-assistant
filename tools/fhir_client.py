"""FHIR R4 REST client for the EHR backend.

Talks to any FHIR R4 server (HAPI test server, Microsoft FHIR, AWS HealthLake,
Epic on FHIR sandbox, etc.). Defaults to the public HAPI FHIR test server at
https://hapi.fhir.org/baseR4 — set `FHIR_BASE_URL` to point elsewhere.

The client returns FHIR resource dicts directly. Callers should use the
mapping helpers (`to_internal_patient` etc.) when they want our internal
patient/condition/observation shapes.

Why a thin client (rather than `fhirclient` or `fhir.resources`):
- Keeps the dependency surface small; the library auth/typing overhead is
  not worth it for the ~6 endpoints we touch.
- Easier to test with `responses` — no model layer to keep in sync.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)


class FHIRError(RuntimeError):
    """Raised on any FHIR-side failure (network, HTTP error, malformed bundle)."""


class FHIRClient:
    """Thin REST client for FHIR R4.

    Authentication: pass `auth_header` (e.g., "Bearer eyJ...") to populate the
    Authorization header for protected servers. The public HAPI sandbox needs
    no auth.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        auth_header: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update({
            "Accept": "application/fhir+json",
            "User-Agent": "agentic-healthcare-assistant/1.0",
        })
        if auth_header:
            self._session.headers["Authorization"] = auth_header

    # ---------- low-level ----------

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = self._session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise FHIRError(f"{method} {url} failed: {exc}") from exc
        if not resp.ok:
            raise FHIRError(
                f"{method} {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise FHIRError(f"{method} {url} returned non-JSON body") from exc

    def _search(self, resource_type: str, **params: Any) -> list[dict]:
        """Run a FHIR search; return the list of resources (Bundle.entry[].resource)."""
        # FHIR servers vary in how they accept _count; the form `_count=N` is universal.
        bundle = self._request("GET", resource_type, params=params)
        entries = bundle.get("entry") or []
        return [e["resource"] for e in entries if "resource" in e]

    # ---------- public API ----------

    def search_patients(
        self,
        *,
        name: Optional[str] = None,
        count: int = 20,
    ) -> list[dict]:
        """Search Patient resources. Name match is server-side substring."""
        params: dict[str, Any] = {"_count": count}
        if name:
            params["name"] = name
        return self._search("Patient", **params)

    def get_patient(self, patient_id: str) -> Optional[dict]:
        """Read a single Patient by FHIR resource id. Returns None on 404."""
        try:
            return self._request("GET", f"Patient/{quote(patient_id, safe='')}")
        except FHIRError as exc:
            if "404" in str(exc):
                return None
            raise

    def list_patients(self, count: int = 50) -> list[dict]:
        return self.search_patients(count=count)

    def get_conditions(self, patient_id: str, count: int = 20) -> list[dict]:
        return self._search("Condition", patient=patient_id, _count=count)

    def get_observations(
        self,
        patient_id: str,
        *,
        count: int = 20,
        category: Optional[str] = None,
    ) -> list[dict]:
        params: dict[str, Any] = {"patient": patient_id, "_count": count, "_sort": "-date"}
        if category:
            params["category"] = category
        return self._search("Observation", **params)

    def upsert_patient(self, patient_resource: dict) -> dict:
        """POST a new Patient (no id) or PUT to an existing one.

        Returns the server's stored Patient resource (with assigned id).
        """
        body = dict(patient_resource)
        if body.get("resourceType") != "Patient":
            body["resourceType"] = "Patient"
        if body.get("id"):
            return self._request(
                "PUT",
                f"Patient/{quote(body['id'], safe='')}",
                json=body,
                headers={"Content-Type": "application/fhir+json"},
            )
        return self._request(
            "POST",
            "Patient",
            json=body,
            headers={"Content-Type": "application/fhir+json"},
        )


# ---------- FHIR ↔ internal shape mapping ----------

# The internal "patient" dict shape (matches the existing SQLite schema in tools/ehr_db.py):
#   {patient_id, name, age, gender, phone_raw, phone_normalized, email, address, summary}
#
# `patient_id` for FHIR-backed patients is `fhir:<server-id>` so we can round-trip
# back to the resource without ambiguity vs SQLite ids.


def _human_name(resource: dict) -> str:
    """Concatenate the first HumanName entry into a display string."""
    names = resource.get("name") or []
    if not names:
        return ""
    n = names[0]
    parts: list[str] = []
    given = n.get("given") or []
    if given:
        parts.append(" ".join(given))
    if n.get("family"):
        parts.append(n["family"])
    if parts:
        return " ".join(parts)
    return n.get("text", "")


def _age_from_birthdate(birthdate: Optional[str]) -> Optional[int]:
    """Compute integer age from an ISO birthDate string (YYYY or YYYY-MM-DD)."""
    if not birthdate:
        return None
    try:
        from datetime import date
        year = int(birthdate.split("-")[0])
        # FHIR birthDate can be just YYYY — assume mid-year.
        return max(date.today().year - year, 0)
    except (ValueError, IndexError):
        return None


def _telecom(resource: dict, system: str) -> Optional[str]:
    for t in resource.get("telecom") or []:
        if t.get("system") == system:
            return t.get("value")
    return None


def _address(resource: dict) -> Optional[str]:
    addrs = resource.get("address") or []
    if not addrs:
        return None
    a = addrs[0]
    if a.get("text"):
        return a["text"]
    parts = a.get("line") or []
    for k in ("city", "state", "postalCode", "country"):
        if a.get(k):
            parts.append(a[k])
    return ", ".join(parts) or None


def to_internal_patient(resource: dict) -> dict:
    """Map a FHIR Patient resource → our internal patient dict."""
    fhir_id = resource.get("id", "")
    return {
        "patient_id": f"fhir:{fhir_id}" if fhir_id else "",
        "name": _human_name(resource),
        "age": _age_from_birthdate(resource.get("birthDate")),
        "gender": (resource.get("gender") or "").title() or None,
        "phone_raw": _telecom(resource, "phone"),
        "phone_normalized": None,
        "email": _telecom(resource, "email"),
        "address": _address(resource),
        "summary": None,  # populated from Conditions when available
        "fhir_id": fhir_id,
        "fhir_resource": resource,
    }


def from_internal_patient(fields: dict) -> dict:
    """Map our internal field dict → a FHIR Patient resource (for POST/PUT).

    Used by `upsert_patient`. Splits the name on whitespace into given+family.
    """
    name = (fields.get("name") or "").strip()
    given = name.split()[:-1] if " " in name else [name]
    family = name.split()[-1] if " " in name else ""
    resource: dict[str, Any] = {
        "resourceType": "Patient",
        "active": True,
        "name": [{"use": "official", "given": given, "family": family, "text": name}],
    }
    if fields.get("gender"):
        resource["gender"] = str(fields["gender"]).lower()
    if fields.get("age") is not None:
        from datetime import date
        resource["birthDate"] = str(date.today().year - int(fields["age"]))
    telecom: list[dict] = []
    if fields.get("phone_raw"):
        telecom.append({"system": "phone", "value": str(fields["phone_raw"])})
    if fields.get("email"):
        telecom.append({"system": "email", "value": str(fields["email"])})
    if telecom:
        resource["telecom"] = telecom
    if fields.get("address"):
        resource["address"] = [{"text": str(fields["address"])}]
    # Strip the "fhir:" prefix off our internal id if present
    pid = fields.get("patient_id") or fields.get("fhir_id")
    if pid:
        resource["id"] = pid.split(":", 1)[1] if str(pid).startswith("fhir:") else pid
    return resource


def condition_summary(conditions: list[dict]) -> str:
    """Short comma-separated condition display, e.g. 'Hypertension, Type 2 diabetes'."""
    names: list[str] = []
    for c in conditions:
        code = c.get("code") or {}
        text = code.get("text")
        if not text:
            for coding in code.get("coding") or []:
                if coding.get("display"):
                    text = coding["display"]
                    break
        if text and text not in names:
            names.append(text)
    return ", ".join(names)


def observation_summary(observations: list[dict], limit: int = 5) -> list[dict]:
    """Compact observation list — code text, value, unit, date — for display."""
    out: list[dict] = []
    for o in observations[:limit]:
        code = o.get("code") or {}
        text = code.get("text") or (
            (code.get("coding") or [{}])[0].get("display") if code.get("coding") else None
        )
        v = o.get("valueQuantity") or {}
        out.append({
            "name": text or "(unnamed)",
            "value": v.get("value"),
            "unit": v.get("unit"),
            "date": o.get("effectiveDateTime") or o.get("issued"),
        })
    return out


def fhir_id_to_patient_id(fhir_id: str) -> str:
    """Convert a raw FHIR resource id → our internal patient_id."""
    return f"fhir:{fhir_id}"


def patient_id_to_fhir_id(patient_id: str) -> Optional[str]:
    """Reverse of `fhir_id_to_patient_id`. Returns None if not a FHIR id."""
    if patient_id.startswith("fhir:"):
        return patient_id.split(":", 1)[1]
    return None


def synthetic_fhir_id(name: str) -> str:
    """Deterministic FHIR id for offline fixtures / walk-ins.

    Used to keep ids stable across seeds so checkpointer threads survive
    re-seeds.
    """
    return hashlib.sha1(name.encode()).hexdigest()[:12]
