"""Centralised configuration: env vars, model selection, and constants.

Supports three LLM backends, picked by env vars at startup:
- Groq (default, free tier): set GROQ_API_KEY
- OpenAI: set OPENAI_API_KEY
- Stub: no key needed; returns deterministic placeholders so the graph
  is testable without any API. Auto-selected when no key is configured.

Search backends are picked at call-time by tools/medical_search.py
(Tavily if TAVILY_API_KEY else DuckDuckGo).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

LLMProvider = Literal["groq", "openai", "stub"]

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    llm_provider: LLMProvider
    llm_model: str
    groq_api_key: str | None
    openai_api_key: str | None
    tavily_api_key: str | None

    enable_persistence: bool
    enable_parallel_fanout: bool

    sqlite_checkpoint_path: str
    ehr_db_path: str
    appointments_db_path: str
    faiss_index_path: str
    faiss_chunks_path: str

    records_xlsx_path: str
    patient_pdf_dir: str


def _detect_provider() -> tuple[LLMProvider, str]:
    """Pick the first provider that has credentials configured."""
    forced = os.getenv("LLM_PROVIDER", "").lower()
    if forced in {"groq", "openai", "stub"}:
        return forced, _model_for(forced)
    if os.getenv("GROQ_API_KEY"):
        return "groq", _model_for("groq")
    if os.getenv("OPENAI_API_KEY"):
        return "openai", _model_for("openai")
    return "stub", "stub"


def _model_for(provider: LLMProvider) -> str:
    overrides = {
        "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "stub": "stub",
    }
    return overrides[provider]


def _resolve_path(env_key: str, default: str) -> str:
    """Resolve a path from env (or default), making it absolute relative to project root."""
    raw = os.getenv(env_key, default)
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)


def load_settings() -> Settings:
    provider, model = _detect_provider()
    return Settings(
        llm_provider=provider,
        llm_model=model,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),

        enable_persistence=os.getenv("ENABLE_PERSISTENCE", "true").lower() == "true",
        enable_parallel_fanout=os.getenv("ENABLE_PARALLEL_FANOUT", "true").lower() == "true",

        sqlite_checkpoint_path=_resolve_path("SQLITE_CHECKPOINT_PATH", "data/checkpoints.sqlite"),
        ehr_db_path=_resolve_path("EHR_DB_PATH", "data/ehr.sqlite"),
        appointments_db_path=_resolve_path("APPOINTMENTS_DB_PATH", "data/appointments.sqlite"),
        faiss_index_path=_resolve_path("FAISS_INDEX_PATH", "data/faiss.index"),
        faiss_chunks_path=_resolve_path("FAISS_CHUNKS_PATH", "data/faiss_chunks.json"),

        records_xlsx_path=_resolve_path(
            "RECORDS_XLSX_PATH",
            "../Datasets_New/Agentic Healthcare Assistant for Medical Task Automation/records.xlsx",
        ),
        patient_pdf_dir=_resolve_path(
            "PATIENT_PDF_DIR",
            "../Datasets_New/Agentic Healthcare Assistant for Medical Task Automation",
        ),
    )


# Constants used elsewhere in the codebase
INTENTS = ("booking", "records", "history", "medical_search", "general")
SPECIALTIES = (
    "general_practice", "cardiology", "endocrinology", "nephrology",
    "neurology", "pulmonology", "oncology", "psychiatry", "dermatology",
)

# Mapping from common condition keywords to the appropriate specialty.
# Used by both the classifier and the booking node when the user describes
# a condition rather than naming a specialty directly.
CONDITION_TO_SPECIALTY: dict[str, str] = {
    "kidney": "nephrology",
    "renal": "nephrology",
    "dialysis": "nephrology",
    "heart": "cardiology",
    "cardiac": "cardiology",
    "chest pain": "cardiology",
    "blood pressure": "cardiology",
    "hypertension": "cardiology",
    "diabetes": "endocrinology",
    "thyroid": "endocrinology",
    "hormone": "endocrinology",
    "stroke": "neurology",
    "seizure": "neurology",
    "migraine": "neurology",
    "neurological": "neurology",
    "lung": "pulmonology",
    "asthma": "pulmonology",
    "respiratory": "pulmonology",
    "cough": "pulmonology",
    "cancer": "oncology",
    "tumor": "oncology",
    "depression": "psychiatry",
    "anxiety": "psychiatry",
    "mental health": "psychiatry",
    "skin": "dermatology",
    "rash": "dermatology",
    "acne": "dermatology",
}
