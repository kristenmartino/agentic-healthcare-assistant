"""Tests for the tool-using agent loop.

We don't make real Anthropic calls — instead we stub the LangChain
ChatAnthropic client with a scripted sequence of AIMessage responses
that exercises the loop's control flow:

  - end_turn on the first call → loop returns plain text
  - tool_use → execute → tool_result → end_turn → loop returns
  - multiple tool_uses in one turn → all executed before next call
  - max_turns budget → loop bails with an error

The tool dispatch path is covered separately at the function level so a
LangChain or Anthropic SDK change doesn't break the tool unit tests.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


def _ai_text(text: str):
    """Build an AIMessage with no tool_calls — signals end_turn."""
    from langchain_core.messages import AIMessage
    return AIMessage(content=text)


def _ai_tool_call(tool_name: str, args: dict, call_id: str = "call-1"):
    """Build an AIMessage with a structured tool_call attribute."""
    from langchain_core.messages import AIMessage
    msg = AIMessage(content=[
        {"type": "tool_use", "id": call_id, "name": tool_name, "input": args},
    ])
    # LangChain normalises tool_calls; mirror that here so _extract_tool_calls
    # finds them on the attribute (its preferred path).
    msg.tool_calls = [{"id": call_id, "name": tool_name, "args": args}]
    return msg


class _ScriptedClient:
    """A fake ChatAnthropic that returns pre-scripted responses in order."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.invocations: list[list] = []

    def bind_tools(self, _specs):
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        if not self.responses:
            raise RuntimeError("scripted client ran out of responses")
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def _force_anthropic_provider(monkeypatch):
    """The agent_loop guards on settings.llm_provider == 'anthropic'. Pretend
    we're configured even though there's no real key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    monkeypatch.setenv("AGENT_MODE", "react")
    # Reset any cached settings + client so the new env takes effect.
    import importlib

    import config
    import llm
    importlib.reload(config)
    importlib.reload(llm)
    yield


def _patch_client_with(client):
    """Patch the agent_loop's _get_client to return our scripted fake."""
    from nodes import agent_loop
    return patch.object(agent_loop, "_get_client", lambda: client)


# ---------- end_turn on first call ----------

def test_immediate_end_turn_returns_text():
    """Claude responds with plain text → loop returns immediately."""
    from nodes.agent_loop import agent_loop_node
    client = _ScriptedClient([_ai_text("Hello! How can I help?")])
    with _patch_client_with(client):
        out = agent_loop_node({"user_input": "hi"})
    assert "Hello" in out["response"]
    assert "informational" in out["response"]  # disclaimer auto-appended
    assert out["intents"] == ["general"]
    # One call made, no tool_log entries (we still write one summary entry).
    assert len(client.invocations) == 1


# ---------- single tool call → tool result → end_turn ----------

def test_single_tool_call_then_end_turn(monkeypatch):
    """find_patient → returns dict → Claude composes final reply."""
    from nodes import agent_tools
    from nodes.agent_loop import agent_loop_node

    # Stub the tool so we don't hit the real EHR.
    monkeypatch.setattr(agent_tools, "_tool_find_patient",
                        lambda name: {"patient_id": "fhir:test", "name": name})

    client = _ScriptedClient([
        _ai_tool_call("find_patient", {"name": "Anjali Mehra"}, "c1"),
        _ai_text("I found Anjali Mehra (fhir:test)."),
    ])
    with _patch_client_with(client):
        out = agent_loop_node({"user_input": "find Anjali Mehra"})
    assert "Anjali" in out["response"]
    assert "records" in out["intents"]
    # Tool log has the call captured.
    tools_called = [e.get("tool") for e in out["tool_log"] if e.get("tool")]
    assert "find_patient" in tools_called
    # Two LLM invocations: initial + after tool result.
    assert len(client.invocations) == 2


# ---------- multiple tool calls in one turn ----------

def test_multiple_tool_uses_in_one_turn_all_executed(monkeypatch):
    """Claude can return two tool_use blocks in one response — both must
    execute and Both ToolMessages must be in the next call's messages."""
    from langchain_core.messages import AIMessage

    from nodes import agent_tools
    from nodes.agent_loop import agent_loop_node

    monkeypatch.setattr(agent_tools, "_tool_book_appointment",
                        lambda **kwargs: {"confirmation_no": "AGS-111"})
    monkeypatch.setattr(agent_tools, "_tool_medical_search",
                        lambda **kwargs: [{"title": "CKD overview"}])

    combo_call = AIMessage(content=[
        {"type": "tool_use", "id": "c1", "name": "book_appointment",
         "input": {"patient_name": "X", "specialty": "nephrology"}},
        {"type": "tool_use", "id": "c2", "name": "medical_search",
         "input": {"query": "ckd treatment"}},
    ])
    combo_call.tool_calls = [
        {"id": "c1", "name": "book_appointment",
         "args": {"patient_name": "X", "specialty": "nephrology"}},
        {"id": "c2", "name": "medical_search", "args": {"query": "ckd treatment"}},
    ]
    client = _ScriptedClient([
        combo_call,
        _ai_text("Booked nephrology slot and here's CKD info."),
    ])
    with _patch_client_with(client):
        out = agent_loop_node({"user_input": "Book a nephrologist + tell me about CKD"})
    assert {"booking", "medical_search"}.issubset(set(out["intents"]))
    # Second LLM invocation must include both tool results.
    second_call = client.invocations[1]
    tool_result_contents = [
        m.content for m in second_call
        if type(m).__name__ == "ToolMessage"
    ]
    assert len(tool_result_contents) == 2


# ---------- max_turns guardrail ----------

def test_max_turns_exceeded_returns_error(monkeypatch):
    """Loop must not run forever even if Claude keeps emitting tool_use."""
    from nodes import agent_tools
    from nodes.agent_loop import agent_loop_node

    monkeypatch.setattr(agent_tools, "_tool_list_patients", lambda: [])
    forever = [_ai_tool_call("list_patients", {}, f"c{i}") for i in range(20)]
    client = _ScriptedClient(forever)
    with _patch_client_with(client):
        out = agent_loop_node({"user_input": "spin forever"})
    assert "turn budget" in out["response"] or "max" in (out.get("error") or "").lower()


# ---------- LLM failure path ----------

def test_llm_failure_returns_graceful_message():
    from nodes.agent_loop import agent_loop_node

    class Boom:
        def bind_tools(self, _):
            return self

        def invoke(self, _messages):
            raise RuntimeError("anthropic 500")

    with _patch_client_with(Boom()):
        out = agent_loop_node({"user_input": "test"})
    assert "trouble" in out["response"].lower() or "moment" in out["response"].lower()
    assert "informational" in out["response"]
    assert out.get("error")


# ---------- is_agent_mode_active gating ----------

def test_agent_mode_inactive_when_provider_not_anthropic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import importlib

    import config
    importlib.reload(config)
    from nodes.agent_loop import is_agent_mode_active
    assert is_agent_mode_active() is False


def test_agent_mode_inactive_when_mode_env_is_graph(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "graph")
    from nodes.agent_loop import is_agent_mode_active
    assert is_agent_mode_active() is False


# ---------- patient context propagation ----------

def test_patient_context_lands_in_system_prompt(monkeypatch):
    """When the UI has a patient selected, the system prompt should mention
    that patient so Claude defaults to it."""
    from nodes.agent_loop import agent_loop_node

    client = _ScriptedClient([_ai_text("ok")])
    with _patch_client_with(client):
        agent_loop_node({
            "user_input": "what conditions does this patient have?",
            "patient_name": "Anjali Mehra",
            "patient_id": "fhir:anjali-mehra",
        })
    system_msg = client.invocations[0][0]
    assert "Anjali Mehra" in system_msg.content
    assert "fhir:anjali-mehra" in system_msg.content


# ---------- tool→intent label mapping ----------

def test_tool_to_intent_mapping_covers_all_tools():
    from nodes.agent_tools import TOOL_FUNCTIONS, tool_to_intent
    for name in TOOL_FUNCTIONS:
        intent = tool_to_intent(name)
        assert intent in {
            "booking", "schedule", "records", "history",
            "medical_search", "audit", "general",
        }, f"unknown intent label '{intent}' for tool {name}"


# ---------- dispatcher graceful errors ----------

def test_dispatch_unknown_tool_returns_error():
    from nodes.agent_tools import dispatch
    result = dispatch("not_a_real_tool", {})
    assert "error" in result
    assert "Unknown" in result["error"]


def test_dispatch_bad_args_returns_error():
    from nodes.agent_tools import dispatch
    result = dispatch("find_patient", {"wrong_kwarg": "x"})
    assert "error" in result


# ---------- tool spec invariants ----------

def test_every_spec_has_a_function():
    from nodes.agent_tools import TOOL_FUNCTIONS, TOOL_SPECS
    spec_names = {s["name"] for s in TOOL_SPECS}
    fn_names = set(TOOL_FUNCTIONS.keys())
    assert spec_names == fn_names, (
        f"specs vs functions mismatch: only-in-specs={spec_names - fn_names}, "
        f"only-in-functions={fn_names - spec_names}"
    )


def test_every_spec_has_description_and_schema():
    from nodes.agent_tools import TOOL_SPECS
    for spec in TOOL_SPECS:
        assert spec.get("description"), f"spec {spec.get('name')} missing description"
        assert spec.get("input_schema"), f"spec {spec.get('name')} missing input_schema"
        assert spec["input_schema"].get("type") == "object"
