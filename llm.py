"""Multi-backend LLM client with a deterministic stub fallback.

Provider priority: Anthropic (Claude Sonnet 4.6, default) → Groq → OpenAI →
Stub. The stub fallback is there so the graph compiles and routes correctly
without any API key — useful for CI, offline development, and the
"plumbing-grades-clean" stance of the Tier 1 eval.

Prompt caching: when the Anthropic provider is in use AND the configured
system prompt is non-trivial (>= 1024 tokens, the Anthropic caching floor),
we tag it with `cache_control={"type": "ephemeral"}` so repeated calls
during a session reuse the prompt at ~10% input cost. Short prompts are
not flagged — below the floor, the API returns an error.
"""
from __future__ import annotations

import logging
import re

from config import Settings, load_settings

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """Raised when the configured LLM backend can't be used."""


# Module-level singleton client, lazily created.
_client = None
_settings: Settings | None = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def build_anthropic_client(
    *, temperature: float = 0.2, max_tokens: int = 1024, timeout: float = 30,
):
    """Build a fresh ChatAnthropic client for callers (like the agent_loop)
    that need a separate instance with their own temperature / max_tokens.

    Centralized here so the API key + model name + import-error path are
    in one place — agent_loop shouldn't have to know which env var holds
    the key. The cached `_get_client()` above is for the single-turn
    `chat()` path; this factory is for streaming / tool-use callers.
    """
    settings = _get_settings()
    if settings.llm_provider != "anthropic":
        raise LLMUnavailable(
            f"build_anthropic_client called but the active provider is "
            f"{settings.llm_provider!r}. Set ANTHROPIC_API_KEY."
        )
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise LLMUnavailable(
            f"langchain_anthropic not installed: {exc}. "
            "Install with: pip install langchain-anthropic"
        ) from exc
    return ChatAnthropic(
        api_key=settings.anthropic_api_key,
        model_name=settings.llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def _get_client():
    """Return a cached client for the configured provider, or None for stub."""
    global _client
    if _client is not None:
        return _client

    settings = _get_settings()
    provider = settings.llm_provider

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            _client = ChatAnthropic(
                api_key=settings.anthropic_api_key,
                model_name=settings.llm_model,
                temperature=0.0,
                max_tokens=512,
            )
            logger.info("LLM provider: Anthropic (%s)", settings.llm_model)
        except ImportError as exc:
            raise LLMUnavailable(
                f"langchain_anthropic not installed: {exc}. "
                "Install with: pip install langchain-anthropic"
            ) from exc

    elif provider == "groq":
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


# Anthropic's prompt-caching minimum: a cache block must be at least 1024
# input tokens (~3-4k chars) — below that the API rejects cache_control.
# We approximate by char length to avoid pulling a tokenizer.
_ANTHROPIC_CACHE_FLOOR_CHARS = 3500


def _to_anthropic_messages(messages: list[dict]) -> list:
    """Convert OpenAI-shape dicts to LangChain messages with cache_control on
    the system prompt when it's long enough to qualify for Anthropic's
    prompt caching minimum.

    Caching is signaled via a `cache_control={"type": "ephemeral"}` block
    inside a structured content array. Anthropic returns cache_creation /
    cache_read tokens in usage metadata so you can see hits on subsequent
    calls in the same 5-minute window.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    settings = _get_settings()

    out = []
    for m in messages:
        role = m["role"]
        text = m["content"]
        if role == "system":
            if (settings.enable_prompt_caching
                    and len(text) >= _ANTHROPIC_CACHE_FLOOR_CHARS):
                # Structured-content shape with ephemeral cache_control.
                out.append(SystemMessage(content=[{
                    "type": "text",
                    "text": text,
                    "cache_control": {"type": "ephemeral"},
                }]))
            else:
                out.append(SystemMessage(content=text))
        elif role in ("user", "human"):
            out.append(HumanMessage(content=text))
        elif role in ("assistant", "ai"):
            out.append(AIMessage(content=text))
        else:
            out.append(HumanMessage(content=text))
    return out


def _to_lc_messages(messages: list[dict]) -> list:
    """Plain LangChain message conversion for non-Anthropic providers."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    role_map = {
        "system": SystemMessage,
        "user": HumanMessage,
        "human": HumanMessage,
        "assistant": AIMessage,
        "ai": AIMessage,
    }
    return [role_map.get(m["role"], HumanMessage)(content=m["content"]) for m in messages]


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Send a chat completion. `messages` is OpenAI-style [{role, content}, ...]."""
    client = _get_client()

    if client == "stub":
        return _stub_response(messages)

    settings = _get_settings()
    if settings.llm_provider == "anthropic":
        lc_messages = _to_anthropic_messages(messages)
    else:
        lc_messages = _to_lc_messages(messages)

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
            "Configure ANTHROPIC_API_KEY (recommended), GROQ_API_KEY, or "
            "OPENAI_API_KEY for real summaries."
        )

    # Composition / final response
    if "compose" in sys_lower or "respond" in sys_lower or "assistant" in sys_lower:
        return (
            "[STUB MODE — set ANTHROPIC_API_KEY (recommended), GROQ_API_KEY, "
            "or OPENAI_API_KEY in .env for real responses]\n"
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
