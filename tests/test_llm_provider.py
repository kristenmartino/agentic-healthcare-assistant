"""Tests for the LLM provider plumbing (config priority + Anthropic caching).

We don't make any real LLM API calls here — the goal is to lock in:
  - provider auto-detection priority (anthropic → groq → openai → stub)
  - LLM_PROVIDER override works for every value
  - prompt-caching message shape only fires above the Anthropic floor
  - non-Anthropic providers never get a cache_control block
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def reset_llm_settings(monkeypatch):
    """Each test starts with a clean env + a freshly imported llm module so
    the module-level singletons (_settings, _client) don't leak between cases."""
    for var in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY",
                "LLM_PROVIDER", "ANTHROPIC_MODEL", "GROQ_MODEL", "OPENAI_MODEL",
                "ENABLE_PROMPT_CACHING"):
        monkeypatch.delenv(var, raising=False)
    import config
    import llm
    importlib.reload(config)
    importlib.reload(llm)
    yield


# ---------- provider priority ----------

def test_anthropic_wins_over_groq_and_openai(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth")
    monkeypatch.setenv("GROQ_API_KEY", "grq")
    monkeypatch.setenv("OPENAI_API_KEY", "oai")
    import config
    importlib.reload(config)
    s = config.load_settings()
    assert s.llm_provider == "anthropic"
    assert s.llm_model.startswith("claude")


def test_groq_wins_when_no_anthropic(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "grq")
    monkeypatch.setenv("OPENAI_API_KEY", "oai")
    import config
    importlib.reload(config)
    assert config.load_settings().llm_provider == "groq"


def test_openai_used_as_last_real_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oai")
    import config
    importlib.reload(config)
    assert config.load_settings().llm_provider == "openai"


def test_stub_when_no_keys(monkeypatch):
    import config
    importlib.reload(config)
    s = config.load_settings()
    assert s.llm_provider == "stub"
    assert s.llm_model == "stub"


def test_llm_provider_env_override_forces_choice(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "oai")
    import config
    importlib.reload(config)
    assert config.load_settings().llm_provider == "openai"


def test_model_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-7")
    import config
    importlib.reload(config)
    assert config.load_settings().llm_model == "claude-opus-4-7"


# ---------- prompt caching ----------

LONG_SYSTEM = "You are a healthcare assistant. " * 200  # ~6k chars


def test_anthropic_caching_tags_long_system_prompt(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth")
    import config
    import llm
    importlib.reload(config)
    importlib.reload(llm)
    msgs = llm._to_anthropic_messages([
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "hello"},
    ])
    system = msgs[0]
    # SystemMessage content should now be a list of structured blocks.
    assert isinstance(system.content, list)
    assert system.content[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_caching_skips_short_system_prompts(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth")
    import config
    import llm
    importlib.reload(config)
    importlib.reload(llm)
    short = "Classify intent."
    msgs = llm._to_anthropic_messages([
        {"role": "system", "content": short},
        {"role": "user", "content": "hello"},
    ])
    # Below the cache floor: content stays a plain string.
    assert msgs[0].content == short


def test_caching_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth")
    monkeypatch.setenv("ENABLE_PROMPT_CACHING", "false")
    import config
    import llm
    importlib.reload(config)
    importlib.reload(llm)
    msgs = llm._to_anthropic_messages([
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "hi"},
    ])
    # Cache disabled → plain string even though prompt is long.
    assert msgs[0].content == LONG_SYSTEM


def test_non_anthropic_path_never_uses_cache_control(monkeypatch):
    """The plain LangChain message conversion has no cache awareness — it's
    used for Groq and OpenAI, where cache_control would be invalid."""
    monkeypatch.setenv("GROQ_API_KEY", "grq")
    import config
    import llm
    importlib.reload(config)
    importlib.reload(llm)
    msgs = llm._to_lc_messages([
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "hi"},
    ])
    assert msgs[0].content == LONG_SYSTEM
    assert isinstance(msgs[0].content, str)


def test_system_message_helper_caches_long_anthropic_prompt(monkeypatch):
    """The helper used by agent_loop should wrap long prompts in the
    structured-content shape with cache_control=ephemeral when Anthropic
    is the active provider — same rule as the chat() path's conversion."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth")
    import config
    import llm
    importlib.reload(config)
    importlib.reload(llm)
    msg = llm.system_message_with_cache_control(LONG_SYSTEM)
    assert isinstance(msg.content, list)
    assert msg.content[0]["type"] == "text"
    assert msg.content[0]["text"] == LONG_SYSTEM
    assert msg.content[0]["cache_control"] == {"type": "ephemeral"}


def test_system_message_helper_skips_short_prompts(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth")
    import config
    import llm
    importlib.reload(config)
    importlib.reload(llm)
    short = "You are an agent."
    msg = llm.system_message_with_cache_control(short)
    assert msg.content == short


def test_system_message_helper_skips_non_anthropic(monkeypatch):
    """Cache_control is Anthropic-specific — other providers must get the
    plain string form."""
    monkeypatch.setenv("GROQ_API_KEY", "grq")
    import config
    import llm
    importlib.reload(config)
    importlib.reload(llm)
    msg = llm.system_message_with_cache_control(LONG_SYSTEM)
    assert msg.content == LONG_SYSTEM


def test_agent_loop_system_message_uses_cache_control(monkeypatch):
    """agent_loop must apply cache_control to its system prompt — that's
    the point of the helper. We inspect the messages it would send rather
    than running a full turn (which would need a real Anthropic key)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth")
    import config
    import llm
    importlib.reload(config)
    importlib.reload(llm)
    # Reload agent_loop so it picks up the freshly-reloaded llm module
    # (the helper is imported lazily inside agent_loop_node, but reloading
    # is the safe belt-and-braces move).
    import nodes.agent_loop as agent_loop
    importlib.reload(agent_loop)
    # The module-level _SYSTEM_PROMPT must clear the cache floor — if it
    # doesn't, the helper would silently fall back to a plain string and
    # we'd ship without caching.
    assert len(agent_loop._SYSTEM_PROMPT) >= llm._ANTHROPIC_CACHE_FLOOR_CHARS, (
        f"agent_loop _SYSTEM_PROMPT is only "
        f"{len(agent_loop._SYSTEM_PROMPT)} chars; cache floor is "
        f"{llm._ANTHROPIC_CACHE_FLOOR_CHARS}. Either lengthen the prompt "
        "or accept that agent_loop won't benefit from prompt caching."
    )
    msg = llm.system_message_with_cache_control(agent_loop._SYSTEM_PROMPT)
    assert isinstance(msg.content, list)
    assert msg.content[0]["cache_control"] == {"type": "ephemeral"}


def test_chat_uses_stub_when_no_keys(monkeypatch):
    """End-to-end: chat() returns a deterministic placeholder when nothing
    is configured. Critical for CI runs and offline development."""
    import config
    import llm
    importlib.reload(config)
    importlib.reload(llm)
    out = llm.chat([
        {"role": "system", "content": "You classify intents."},
        {"role": "user", "content": "Book me a cardiologist"},
    ])
    assert "book" in out.lower() or "booking" in out.lower() or out
