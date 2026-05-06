"""Intent classifier node.

Routes user queries to one of 5 downstream paths: booking / records / history /
medical_search / general. Multi-intent queries (e.g., "book a doctor AND
summarize treatments") return multiple intents that the graph fans out.

Uses the configured LLM with a deterministic heuristic fallback so the graph
works even when no API key is set or the call fails.
"""
from __future__ import annotations

import logging
import re

from config import CONDITION_TO_SPECIALTY, INTENTS
from llm import LLMUnavailable, _stub_classify_intent, chat
from prompts import (
    INTENT_CLASSIFIER_PROMPT,
    PATIENT_NAME_EXTRACTOR_PROMPT,
    SPECIALTY_EXTRACTOR_PROMPT,
)
from state import HealthcareState

logger = logging.getLogger(__name__)


def _parse_intents(raw: str) -> list[str]:
    """Parse a comma-separated intent string from the LLM output."""
    candidates = [c.strip().lower() for c in raw.replace(";", ",").split(",")]
    valid = [c for c in candidates if c in INTENTS]
    return valid or ["general"]


def _extract_specialty_heuristic(text: str) -> str:
    """Find a specialty by checking condition keywords."""
    lowered = text.lower()
    # Direct specialty mention takes priority
    for spec in (
        "cardiology", "endocrinology", "nephrology", "neurology",
        "pulmonology", "oncology", "psychiatry", "dermatology",
    ):
        if spec in lowered or spec.replace("ology", "ologist") in lowered:
            return spec
    # Condition keyword mapping
    for kw, spec in CONDITION_TO_SPECIALTY.items():
        if kw in lowered:
            return spec
    return "general_practice"


def _extract_name_heuristic(text: str) -> str:
    """Find a Capitalised Name pair (e.g., 'Anjali Mehra')."""
    # First try double-capitalized name
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
    if m:
        # Filter out leading words like "Book", "Show", "Update", "Add"
        candidate = m.group(1)
        leading = candidate.split()[0]
        if leading in {"Book", "Show", "Update", "Add", "Schedule", "Tell", "What", "How", "Find"}:
            # Try the rest of the match
            rest = " ".join(candidate.split()[1:])
            if len(rest.split()) >= 2:
                return rest
            return ""
        return candidate
    return ""


def classify_intent(state: HealthcareState) -> dict:
    user_input = (state.get("user_input") or "").strip()
    if not user_input:
        return {"intent": "general", "intents": ["general"]}

    # --- Step 1: classify intent(s) ---
    try:
        raw = chat(
            messages=[
                {"role": "system", "content": INTENT_CLASSIFIER_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.0,
            max_tokens=24,
        )
        intents = _parse_intents(raw)
    except LLMUnavailable as exc:
        logger.warning("Classifier LLM unavailable (%s); using heuristic.", exc)
        raw = _stub_classify_intent(user_input.lower())
        intents = _parse_intents(raw)
    except Exception as exc:
        logger.exception("Classifier failed: %s", exc)
        raw = _stub_classify_intent(user_input.lower())
        intents = _parse_intents(raw)

    primary = intents[0]

    # --- Step 2: extract auxiliary info if relevant ---
    requested_specialty = None
    patient_name = state.get("patient_name")  # may be set by sidebar selector

    if "booking" in intents:
        requested_specialty = _extract_specialty_heuristic(user_input)
        # Refine via LLM only if the heuristic returned the generic fallback AND a real provider is configured.
        # In stub mode, the LLM "refinement" is just CONDITION_TO_SPECIALTY lookup which is no better than the heuristic
        # and may overwrite a correct heuristic match (e.g., "cardiologist") with general_practice.
        from config import SPECIALTIES, load_settings
        if requested_specialty == "general_practice" and load_settings().llm_provider != "stub":
            try:
                refined = chat(
                    messages=[
                        {"role": "system", "content": SPECIALTY_EXTRACTOR_PROMPT},
                        {"role": "user", "content": user_input},
                    ],
                    temperature=0.0,
                    max_tokens=8,
                ).strip().lower()
                if refined in SPECIALTIES:
                    requested_specialty = refined
            except LLMUnavailable:
                pass

    if not patient_name and ("records" in intents or "history" in intents or "booking" in intents):
        # Heuristic first
        patient_name = _extract_name_heuristic(user_input)
        if not patient_name:
            try:
                extracted = chat(
                    messages=[
                        {"role": "system", "content": PATIENT_NAME_EXTRACTOR_PROMPT},
                        {"role": "user", "content": user_input},
                    ],
                    temperature=0.0,
                    max_tokens=16,
                ).strip()
                if extracted and 1 <= len(extracted.split()) <= 4:
                    patient_name = extracted
            except LLMUnavailable:
                pass

    update: dict = {
        "intent": primary,
        "intents": intents,
        "tool_log": [{
            "node": "classify_intent",
            "raw_llm_output": raw,
            "parsed_intents": intents,
        }],
    }
    if requested_specialty:
        update["requested_specialty"] = requested_specialty
    if patient_name:
        update["patient_name"] = patient_name

    return update
