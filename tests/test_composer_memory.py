"""Composer threads conversation history into the LLM prompt (issue #9).

The composer skips the LLM entirely in stub mode (it uses a deterministic
template), so these tests force a non-stub provider to exercise the real
message-assembly path, then capture what gets handed to `chat()`.
"""
from __future__ import annotations

import types


def _force_real_provider(monkeypatch):
    import config
    monkeypatch.setattr(
        config, "load_settings",
        lambda: types.SimpleNamespace(llm_provider="anthropic"),
    )


def test_composer_threads_history_into_messages(monkeypatch):
    import nodes.composer as composer
    _force_real_provider(monkeypatch)

    captured: dict = {}

    def _fake_chat(messages, **kwargs):
        captured["messages"] = messages
        return "Composed reply."

    monkeypatch.setattr(composer, "chat", _fake_chat)

    state = {
        "user_input": "book him a cardiologist",
        "history": [
            {"role": "user", "content": "Show me Ramesh's history"},
            {"role": "assistant", "content": "Ramesh has CKD stage 2."},
        ],
    }
    out = composer.compose_response_node(state)
    assert out["response"].startswith("Composed reply.")

    roles = [m["role"] for m in captured["messages"]]
    # system + 2 history turns + current user block
    assert roles == ["system", "user", "assistant", "user"]
    assert "Show me Ramesh's history" in captured["messages"][1]["content"]
    assert "Ramesh has CKD" in captured["messages"][2]["content"]
    assert "book him a cardiologist" in captured["messages"][-1]["content"]


def test_composer_without_history_has_no_extra_turns(monkeypatch):
    import nodes.composer as composer
    _force_real_provider(monkeypatch)

    captured: dict = {}
    monkeypatch.setattr(
        composer, "chat",
        lambda messages, **kw: captured.update(messages=messages) or "ok",
    )

    composer.compose_response_node({"user_input": "hello", "history": []})
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user"]


def test_composer_caps_history_turns(monkeypatch):
    """History is bounded so a long conversation can't blow the token budget."""
    import nodes.composer as composer
    _force_real_provider(monkeypatch)

    captured: dict = {}
    monkeypatch.setattr(
        composer, "chat",
        lambda messages, **kw: captured.update(messages=messages) or "ok",
    )

    long_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
        for i in range(40)
    ]
    composer.compose_response_node(
        {"user_input": "now", "history": long_history},
    )
    # system + capped history + current user block
    history_msgs = len(captured["messages"]) - 2
    assert history_msgs == composer._HISTORY_TURN_CAP


def test_composer_skips_blank_and_unknown_role_turns(monkeypatch):
    import nodes.composer as composer
    _force_real_provider(monkeypatch)

    captured: dict = {}
    monkeypatch.setattr(
        composer, "chat",
        lambda messages, **kw: captured.update(messages=messages) or "ok",
    )

    state = {
        "user_input": "go",
        "history": [
            {"role": "user", "content": "  "},          # blank → skipped
            {"role": "system", "content": "ignore me"},  # unknown role → skipped
            {"role": "assistant", "content": "kept"},
        ],
    }
    composer.compose_response_node(state)
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "assistant", "user"]
    assert captured["messages"][1]["content"] == "kept"
