"""Multi-backend LLM client with a deterministic stub fallback.

Why a stub backend? The graph must be testable without any API key.
The stub returns plausible but deterministic outputs so:
- the graph compiles and routes correctly,
- nodes downstream of an LLM call don't see None,
- the composer produces a sensible (if templated) response.

Real LLM use is the default when GROQ_API_KEY or OPENAI_API_KEY is set.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from config import Settings, load_settings

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """Raised when the configured LLM backend can't be used."""


# Module-level singleton client, lazily created.
_client = None
_settings: Optional[Settings] = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def _get_client():
    """Return a cached client for the configured provider, or None for stub."""
    global _client
    if _client is not None:
        return _client

    settings = _get_settings()
    provider = settings.llm_provider

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
            _client = ChatGroq(
                api_key=settings.groq_api_key,
                model_name=settings.llm_model,
                temperature=0.0,
                max_tokens=512,
            )
            logger.info("LLM provider: Groq (%s)", settings.llm_model)
        except ImportError as exc:
            raise LLMUnavailable(f"langchain_groq not installed: {exc}") from exc

    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            _client = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.llm_model,
                temperature=0.0,
                max_tokens=512,
            )
            logger.info("LLM provider: OpenAI (%s)", settings.llm_model)
        except ImportError as exc:
            raise LLMUnavailable(f"langchain_openai not installed: {exc}") from exc

    else:
        # stub provider
        _client = "stub"
        logger.info("LLM provider: STUB (no API key configured) — using deterministic placeholders")

    return _client


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Send a chat completion. `messages` is OpenAI-style [{role, content}, ...]."""
    client = _get_client()

    if client == "stub":
        return _stub_response(messages)

    # langchain_groq / langchain_openai: convert dicts to LangChain message objects.
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    role_map = {
        "system": SystemMessage,
        "user": HumanMessage,
        "human": HumanMessage,
        "assistant": AIMessage,
        "ai": AIMessage,
    }
    lc_messages = []
    for m in messages:
        cls = role_map.get(m["role"], HumanMessage)
        lc_messages.append(cls(content=m["content"]))

    try:
        result = client.invoke(lc_messages)
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        raise LLMUnavailable(str(exc)) from exc

    return result.content if hasattr(result, "content") else str(result)


def _stub_response(messages: list[dict]) -> str:
    """Deterministic placeholder response for stub provider.

    Returns:
    - For intent classification (system prompt mentions intents) → routes by keywords.
    - For specialty extraction → "general_practice".
    - For history summarization → "Stub summary: see records for details."
    - For composition → echoes the user query with a short header.
    """
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")

    sys_lower = system.lower()
    user_lower = user.lower()

    # Intent classifier
    if "classify" in sys_lower and "intent" in sys_lower:
        return _stub_classify_intent(user_lower)

    # Specialty extraction
    if "specialty" in sys_lower:
        from config import CONDITION_TO_SPECIALTY
        for k, v in CONDITION_TO_SPECIALTY.items():
            if k in user_lower:
                return v
        return "general_practice"

    # Patient name extraction
    if "extract" in sys_lower and "name" in sys_lower:
        # Naive: capitalised pair of words
        m = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", user)
        return m.group(1) if m else ""

    # History summarization
    if "summari" in sys_lower or "summary" in sys_lower:
        return (
            "Stub summary: patient context is available in the EHR records. "
            "Configure GROQ_API_KEY or OPENAI_API_KEY for real summaries."
        )

    # Composition / final response
    if "compose" in sys_lower or "respond" in sys_lower or "assistant" in sys_lower:
        return (
            "[STUB MODE — set GROQ_API_KEY or OPENAI_API_KEY in .env for real responses]\n"
            f"Acknowledged: {user[:200]}"
        )

    # Default fallback
    return "stub"


def _stub_classify_intent(user_lower: str) -> str:
    """Heuristic intent classifier used in stub mode and as LLM fallback."""
    # Check booking first since it can co-occur with medical_search
    booking_kw = ("book", "schedule", "appointment", "see a", "see the", "consult")
    records_kw = ("add record", "update record", "register patient", "new patient", "save record")
    history_kw = ("history", "past visits", "previous", "what did", "summarize record", "patient summary")
    search_kw = ("treatment", "what is", "how is", "symptoms of", "cure for", "latest research", "medline", "who recommend")
    greeting_kw = ("hello", "hi ", "hey", "thanks", "thank you", "good morning")

    has_booking = any(k in user_lower for k in booking_kw)
    has_records = any(k in user_lower for k in records_kw)
    has_history = any(k in user_lower for k in history_kw)
    has_search = any(k in user_lower for k in search_kw)
    has_greeting = any(k in user_lower for k in greeting_kw) and not (has_booking or has_records or has_search)

    if has_records:
        return "records"
    if has_booking and has_search:
        return "booking,medical_search"
    if has_booking and has_history:
        return "booking,history"
    if has_booking:
        return "booking"
    if has_history:
        return "history"
    if has_search:
        return "medical_search"
    if has_greeting:
        return "general"
    return "general"
