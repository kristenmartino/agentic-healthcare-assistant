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
    """A fake ChatAnthropic that returns pre-scripted responses in order.

    Supports both .invoke() and .stream() so the agent_loop's new
    streaming path can be unit-tested without a real LangChain client.
    stream() yields the same response as a single chunk — tests don't
    need per-token chunking to verify the accumulator + tool-loop
    plumbing.
    """

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

    def stream(self, messages):
        # Yield the next scripted response as a single chunk. The
        # agent_loop's _stream_and_accumulate handles the chunk-or-not
        # case so this works whether we yield one chunk or many.
        self.invocations.append(messages)
        if not self.responses:
            raise RuntimeError("scripted client ran out of responses")
        yield self.responses.pop(0)


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


def test_streaming_path_used_when_client_supports_it(monkeypatch):
    """Fix #13: agent_loop must call client.stream() (not .invoke()) so
    LangGraph's stream_mode='messages' can capture token deltas. This
    test asserts the agent uses the streaming path when available."""
    from nodes.agent_loop import agent_loop_node

    captured = {"stream_called": 0, "invoke_called": 0}

    class _StreamingSpy:
        def bind_tools(self, _):
            return self

        def stream(self, _messages):
            captured["stream_called"] += 1
            yield _ai_text("Streamed response.")

        def invoke(self, _messages):
            captured["invoke_called"] += 1
            return _ai_text("Invoked response.")

    monkeypatch.setattr("nodes.agent_loop._get_client", lambda: _StreamingSpy())
    out = agent_loop_node({"user_input": "hi"})
    # stream() was used; invoke() never reached.
    assert captured["stream_called"] == 1
    assert captured["invoke_called"] == 0
    assert "Streamed" in out["response"]


def test_stream_accumulator_concatenates_multiple_chunks():
    """Test the chunk accumulator directly with a multi-chunk stream
    (simulating what LangChain actually yields in production)."""
    from langchain_core.messages import AIMessageChunk

    from nodes.agent_loop import _stream_and_accumulate

    class _MultiChunkClient:
        def stream(self, _messages):
            yield AIMessageChunk(content="Hello ")
            yield AIMessageChunk(content="world")
            yield AIMessageChunk(content="!")

    result = _stream_and_accumulate(_MultiChunkClient(), [])
    # AIMessageChunk + AIMessageChunk concatenates content per LangChain's
    # __add__ protocol — we should end up with the full message.
    assert "Hello world!" in (result.content if isinstance(result.content, str)
                              else "".join(b.get("text", "") for b in result.content))


def test_stream_accumulator_falls_back_to_invoke_when_no_stream():
    """Defensive: if a future client only implements .invoke(), the
    accumulator should still work."""
    from nodes.agent_loop import _stream_and_accumulate

    class _InvokeOnly:
        def invoke(self, _messages):
            from langchain_core.messages import AIMessage
            return AIMessage(content="fallback")

    result = _stream_and_accumulate(_InvokeOnly(), [])
    assert result.content == "fallback"


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


# ---------- tool delegation to legacy nodes (fix #7) ----------
#
# Behavioral parity: history + medical_search tools must produce the
# same shape of structured artifact the legacy graph nodes produce
# (clinician-style summary with FAISS citations for history; cited
# synthesis + indexed sources for medical_search). Easiest path is to
# have the tool wrapper invoke the legacy node directly.

def test_get_patient_history_delegates_to_history_node(monkeypatch):
    """The tool must call history_node so the resulting history_summary
    is the legacy clinician synthesis, not a raw JSON dump."""
    from nodes import agent_tools, history
    from tools import ehr

    # Stub the EHR lookups (local imports inside the tool function will
    # pick these up because the imports re-resolve from the module).
    monkeypatch.setattr(ehr, "find_patient_by_name", lambda name, settings=None,
                        actor=None: {"patient_id": "fhir:test", "name": name})
    monkeypatch.setattr(ehr, "get_patient_clinical_context",
                        lambda pid, settings=None, actor=None:
                            {"conditions": [], "observations": []})

    called = {"n": 0}

    def _fake_history_node(state):
        called["n"] += 1
        assert state.get("patient_name") == "Anjali Mehra"
        return {
            "history_summary": "DELEGATED SYNTHESIS from legacy node",
            "tool_log": [{"node": "history", "pdf_chunks_retrieved": 3}],
        }

    monkeypatch.setattr(history, "history_node", _fake_history_node)

    result = agent_tools._tool_get_patient_history("Anjali Mehra")
    assert called["n"] == 1, "legacy history_node was not invoked"
    assert result["history_summary"] == "DELEGATED SYNTHESIS from legacy node"
    assert result["pdf_chunks_retrieved"] == 3


def test_medical_search_delegates_to_medical_search_node(monkeypatch):
    """The tool must return a dict shape (not a bare list) with
    synthesis + sources + raw_results, matching legacy graph mode."""
    from nodes import agent_tools, medical_search_node

    def _fake_node(state):
        return {
            "medical_info": [
                {"synthesis": "Test synthesis citing [1]."},
                {"title": "Source One", "url": "https://x", "snippet": "...",
                 "source": "tavily"},
            ],
            "sources": [{"index": 1, "title": "Source One",
                         "url": "https://x", "source": "tavily"}],
        }

    monkeypatch.setattr(medical_search_node, "medical_search_node", _fake_node)
    result = agent_tools._tool_medical_search("pneumonia symptoms")
    assert isinstance(result, dict), "tool should return dict, not list"
    assert result.get("synthesis") == "Test synthesis citing [1]."
    assert len(result.get("sources") or []) == 1
    # raw_results split out for the LLM's convenience
    assert len(result.get("raw_results") or []) == 1


def test_medical_search_falls_back_when_legacy_node_raises(monkeypatch):
    """If the legacy node blows up, the tool must still return SOMETHING
    so the agent loop doesn't lose the search step entirely."""
    from nodes import agent_tools, medical_search_node

    def _broken(_state):
        raise RuntimeError("FAISS index missing")

    monkeypatch.setattr(medical_search_node, "medical_search_node", _broken)
    monkeypatch.setattr(
        "tools.medical_search.medical_search",
        lambda *args, **kwargs: [{"title": "fallback", "snippet": "",
                                   "url": "", "source": "stub"}],
    )
    result = agent_tools._tool_medical_search("anything")
    assert isinstance(result, dict)
    assert result.get("results_count") == 1
    # Synthesis may be None; that's OK as long as the shape is preserved.


# ---------- conversation-history threading ----------
#
# Regression for PR #6 review: the loop must replay prior conversation
# turns AND surface structured artifacts from the previous assistant turn
# so "cancel that appointment" resolves deterministically from state,
# not from lucky text-matching.

def test_conversation_history_is_threaded_into_messages():
    from langchain_core.messages import AIMessage as LCAi
    from langchain_core.messages import HumanMessage as LCHu

    from nodes.agent_loop import agent_loop_node

    client = _ScriptedClient([_ai_text("Got it.")])
    with _patch_client_with(client):
        agent_loop_node({
            "user_input": "follow-up question",
            "history": [
                {"role": "user", "content": "Earlier I asked about Anjali."},
                {"role": "assistant", "content": "Anjali has diabetes."},
            ],
        })
    msgs = client.invocations[0]
    # First message is the system prompt; then the two prior turns; then
    # the current user input. (No prior appointment, so no hint message.)
    human_msgs = [m for m in msgs if isinstance(m, LCHu)]
    ai_msgs = [m for m in msgs if isinstance(m, LCAi)]
    assert any("Earlier I asked about Anjali" in m.content for m in human_msgs)
    assert any("Anjali has diabetes" in m.content for m in ai_msgs)


def test_prior_appointment_is_surfaced_as_system_hint(monkeypatch):
    """If the prior turn produced an appointment, the model must see the
    slot_id + confirmation_no in a system hint so 'cancel that appointment'
    can resolve to the right slot without scraping text from the chat
    history."""
    from langchain_core.messages import SystemMessage as LCSys

    from nodes.agent_loop import agent_loop_node

    client = _ScriptedClient([_ai_text("ok")])
    with _patch_client_with(client):
        agent_loop_node({
            "user_input": "what was that confirmation again?",
            "appointment": {
                "slot_id": 42, "confirmation_no": "AGS-555111",
                "doctor_name": "Dr. Y", "start_time": "2026-06-02T10:00:00",
            },
        })
    sys_msgs = [m for m in client.invocations[0] if isinstance(m, LCSys)]
    hint_text = "\n".join(m.content for m in sys_msgs if isinstance(m.content, str))
    assert "AGS-555111" in hint_text
    assert "slot_id=42" in hint_text


def test_two_turn_book_then_cancel_resolves_deterministically(monkeypatch):
    """End-to-end: turn 1 books, turn 2 says 'cancel that appointment'.
    The second turn must call cancel_booking with the slot_id from the
    first turn's structured state — NOT depend on the LLM scraping
    'AGS-…' from a stringified history."""
    from nodes import agent_tools
    from nodes.agent_loop import agent_loop_node

    booking = {
        "doctor_name": "Dr. Z", "specialty": "cardiology",
        "start_time": "2026-06-03T09:00:00", "end_time": "2026-06-03T09:30:00",
        "slot_id": 77, "confirmation_no": "AGS-CANCEL", "status": "confirmed",
    }
    monkeypatch.setattr(agent_tools, "_tool_book_appointment",
                        lambda **kwargs: booking)
    captured_cancel_args: dict = {}

    def _fake_cancel(**kwargs):
        captured_cancel_args.update(kwargs)
        return {"status": "cancelled", "slot_id": kwargs.get("slot_id")}

    monkeypatch.setattr(agent_tools, "_tool_cancel_booking", _fake_cancel)

    # Stub list_all_bookings so the cancel_booking ownership check sees
    # the active patient as the owner of slot 77. (Fix #11 added a
    # pre-cancel ownership check that needs to resolve the slot.)
    from tools import appointments
    monkeypatch.setattr(
        appointments, "list_all_bookings",
        lambda *args, **kwargs: [
            {"slot_id": 77, "booked_by_patient_id": "fhir:active",
             "confirmation_no": "AGS-CANCEL"},
        ],
    )

    # Turn 1: book. role=admin to skip patient-chat policy on this
    # accumulator-focused test (booking flow is fine in patient_chat too;
    # admin just avoids needing an active_pid in state).
    client1 = _ScriptedClient([
        _ai_tool_call("book_appointment",
                      {"patient_name": "X", "specialty": "cardiology"}, "c1"),
        _ai_text("Booked AGS-CANCEL."),
    ])
    with _patch_client_with(client1):
        turn1 = agent_loop_node({"user_input": "book a cardiologist",
                                  "role": "admin"})
    assert turn1["appointment"]["confirmation_no"] == "AGS-CANCEL"

    # Turn 2: "cancel that appointment". Active patient is set so the
    # ownership check passes (we stubbed list_all_bookings to say slot 77
    # belongs to fhir:active).
    client2 = _ScriptedClient([
        _ai_tool_call("cancel_booking", {"slot_id": 77}, "c2"),
        _ai_text("Cancelled."),
    ])
    with _patch_client_with(client2):
        turn2 = agent_loop_node({
            "user_input": "cancel that appointment",
            "appointment": turn1["appointment"],
            "patient_id": "fhir:active",
            "history": [
                {"role": "user", "content": "book a cardiologist"},
                {"role": "assistant", "content": "Booked AGS-CANCEL."},
            ],
        })
    assert captured_cancel_args == {"slot_id": 77}
    assert (turn2.get("appointment") or {}).get("action") == "cancelled"


# ---------- structured-state accumulator ----------
#
# Regression for PR #6 review: the agent_loop must populate the
# HealthcareState fields (appointment, record_change, history_summary,
# medical_info, sources, plus new agent-only fields) — not just response
# text. The Streamlit panel, FastAPI /chat done payload, the deterministic
# eval, and the Next.js artifact rows all read these fields.

def test_book_appointment_populates_appointment_state(monkeypatch):
    from nodes import agent_tools
    from nodes.agent_loop import agent_loop_node

    fake_appt = {
        "doctor_name": "Dr. Test", "specialty": "cardiology",
        "start_time": "2026-06-01T09:00:00", "end_time": "2026-06-01T09:30:00",
        "slot_id": 42, "confirmation_no": "AGS-999000",
        "patient_id": "fhir:x", "patient_name": "Test", "status": "confirmed",
    }
    monkeypatch.setattr(agent_tools, "_tool_book_appointment",
                        lambda **kwargs: fake_appt)
    client = _ScriptedClient([
        _ai_tool_call("book_appointment",
                      {"patient_name": "Test", "specialty": "cardiology"}, "c1"),
        _ai_text("Booked Dr. Test for June 1."),
    ])
    with _patch_client_with(client):
        out = agent_loop_node({"user_input": "book a cardiologist"})
    assert out.get("appointment") == fake_appt


def test_cancel_booking_marks_appointment_cancelled(monkeypatch):
    """Verifies the accumulator mapping. Uses role=admin to bypass the
    ownership pre-auth check, which is tested separately in
    test_agent_phi_scope.py::test_cancel_*."""
    from nodes import agent_tools
    from nodes.agent_loop import agent_loop_node

    monkeypatch.setattr(agent_tools, "_tool_cancel_booking",
                        lambda **kwargs: {"status": "cancelled", "slot_id": 7})
    client = _ScriptedClient([
        _ai_tool_call("cancel_booking", {"slot_id": 7}, "c1"),
        _ai_text("Cancelled."),
    ])
    with _patch_client_with(client):
        out = agent_loop_node({"user_input": "cancel slot 7", "role": "admin"})
    appt = out.get("appointment") or {}
    assert appt.get("action") == "cancelled"
    assert appt.get("slot_id") == 7


def test_upsert_patient_populates_record_change(monkeypatch):
    from nodes import agent_tools
    from nodes.agent_loop import agent_loop_node

    monkeypatch.setattr(agent_tools, "_tool_upsert_patient", lambda **kwargs: {
        "operation": "insert", "patient_id": "fhir:newpatient",
        "before": None, "after": {"name": kwargs["name"]},
    })
    client = _ScriptedClient([
        _ai_tool_call("upsert_patient", {"name": "Jane Doe", "age": 40}, "c1"),
        _ai_text("Created Jane Doe."),
    ])
    with _patch_client_with(client):
        out = agent_loop_node({"user_input": "add Jane Doe age 40"})
    rec = out.get("record_change") or {}
    assert rec.get("operation") == "insert"
    assert rec.get("patient_id") == "fhir:newpatient"
    # patient_id is also carried forward at the top level for the trace log
    assert out.get("patient_id") == "fhir:newpatient"


def test_get_patient_history_populates_history_summary(monkeypatch):
    from nodes import agent_tools
    from nodes.agent_loop import agent_loop_node

    monkeypatch.setattr(agent_tools, "_tool_get_patient_history", lambda **kwargs: {
        "record": {"name": "Anjali Mehra", "age": 40, "gender": "Female",
                   "summary": "Type 2 diabetes mellitus"},
        "conditions": [{"code": {"text": "Type 2 diabetes mellitus"}}],
        "observations": [{"name": "HbA1c", "value": 7.4, "unit": "%",
                          "date": "2026-04-12"}],
    })
    client = _ScriptedClient([
        _ai_tool_call("get_patient_history",
                      {"patient_name": "Anjali Mehra"}, "c1"),
        _ai_text("Here's the history."),
    ])
    with _patch_client_with(client):
        # role=admin bypasses the patient-chat post-resolution check;
        # the cross-patient denial is tested separately in
        # test_agent_phi_scope.py.
        out = agent_loop_node({"user_input": "show Anjali's history",
                                "role": "admin"})
    summary = out.get("history_summary") or ""
    assert "Anjali Mehra" in summary
    assert "diabetes" in summary.lower()
    assert "HbA1c" in summary


def test_medical_search_populates_medical_info_and_sources(monkeypatch):
    """Fix #7: tool now delegates to medical_search_node, so the shape is
    a dict with medical_info + sources + synthesis. Verify the accumulator
    pulls those directly into state (no re-indexing in the accumulator)."""
    from nodes import agent_tools
    from nodes.agent_loop import agent_loop_node

    raw_results = [
        {"title": "Pneumonia symptoms", "snippet": "Cough, fever…",
         "url": "https://medlineplus.gov/x", "source": "tavily"},
        {"title": "WHO on pneumonia", "snippet": "…",
         "url": "https://who.int/y", "source": "tavily"},
    ]
    medical_info_with_synthesis = [
        {"synthesis": "Pneumonia usually presents with cough, fever, "
                       "shortness of breath [1][2]."}
    ] + raw_results
    sources = [
        {"index": 1, "title": "Pneumonia symptoms",
         "url": "https://medlineplus.gov/x", "source": "tavily"},
        {"index": 2, "title": "WHO on pneumonia",
         "url": "https://who.int/y", "source": "tavily"},
    ]
    monkeypatch.setattr(agent_tools, "_tool_medical_search", lambda **kwargs: {
        "medical_info": medical_info_with_synthesis,
        "sources": sources,
        "synthesis": medical_info_with_synthesis[0]["synthesis"],
        "raw_results": raw_results,
        "results_count": 2,
    })
    client = _ScriptedClient([
        _ai_tool_call("medical_search", {"query": "pneumonia symptoms"}, "c1"),
        _ai_text("Pneumonia commonly presents with…"),
    ])
    with _patch_client_with(client):
        out = agent_loop_node({"user_input": "symptoms of pneumonia"})
    # State carries the synthesized [synth + raw] shape graph mode uses.
    assert out.get("medical_info") == medical_info_with_synthesis
    assert out.get("sources") == sources
    assert "medlineplus" in sources[0]["url"]


def test_doctor_schedule_populates_schedule_results(monkeypatch):
    from nodes import agent_tools
    from nodes.agent_loop import agent_loop_node

    fake = {"doctor": {"name": "Dr. Nair"},
            "schedule": [{"slot_id": 1, "start_time": "2026-06-01T09:00:00",
                          "booked": 0}]}
    monkeypatch.setattr(agent_tools, "_tool_get_doctor_schedule",
                        lambda **kwargs: fake)
    client = _ScriptedClient([
        _ai_tool_call("get_doctor_schedule", {"doctor_name": "Nair"}, "c1"),
        _ai_text("Dr. Nair has openings."),
    ])
    with _patch_client_with(client):
        out = agent_loop_node({"user_input": "Dr. Nair's calendar"})
    assert out.get("schedule_results") == fake


def test_list_my_bookings_populates_bookings_results(monkeypatch):
    from nodes import agent_tools
    from nodes.agent_loop import agent_loop_node

    fake = [{"slot_id": 1, "confirmation_no": "AGS-111", "doctor_name": "X"}]
    monkeypatch.setattr(agent_tools, "_tool_list_my_bookings",
                        lambda **kwargs: fake)
    client = _ScriptedClient([
        _ai_tool_call("list_my_bookings", {"patient_id": "fhir:x"}, "c1"),
        _ai_text("You have one."),
    ])
    with _patch_client_with(client):
        out = agent_loop_node({
            "user_input": "show my bookings",
            "patient_id": "fhir:x",
            "patient_name": "X",
        })
    assert out.get("bookings_results") == fake


def test_get_audit_log_populates_audit_results(monkeypatch):
    from nodes import agent_tools
    from nodes.agent_loop import agent_loop_node

    fake = [{"id": 1, "ts": "2026-05-27T10:00:00Z", "actor": "patient_chat",
             "action": "ehr.read"}]
    monkeypatch.setattr(agent_tools, "_tool_get_audit_log",
                        lambda **kwargs: fake)
    client = _ScriptedClient([
        _ai_tool_call("get_audit_log", {"patient_id": "fhir:x"}, "c1"),
        _ai_text("Recent access events."),
    ])
    with _patch_client_with(client):
        # patient_id passed so the patient-chat audit check accepts it.
        out = agent_loop_node({
            "user_input": "who accessed my records",
            "patient_id": "fhir:x",
        })
    assert out.get("audit_results") == fake


def test_failed_tool_does_not_overwrite_state(monkeypatch):
    """A tool returning an error must not clobber a previously-populated
    state field. Otherwise a follow-up failed call could erase the
    successful booking from one turn back."""
    from nodes import agent_tools
    from nodes.agent_loop import agent_loop_node

    monkeypatch.setattr(agent_tools, "_tool_book_appointment",
                        lambda **kwargs: {"confirmation_no": "AGS-OK",
                                          "doctor_name": "Real", "slot_id": 1})
    monkeypatch.setattr(agent_tools, "_tool_cancel_booking",
                        lambda **kwargs: {"error": "slot not found"})

    from langchain_core.messages import AIMessage
    combo = AIMessage(content=[
        {"type": "tool_use", "id": "a", "name": "book_appointment",
         "input": {"patient_name": "x", "specialty": "cardiology"}},
        {"type": "tool_use", "id": "b", "name": "cancel_booking",
         "input": {"slot_id": 999}},
    ])
    combo.tool_calls = [
        {"id": "a", "name": "book_appointment",
         "args": {"patient_name": "x", "specialty": "cardiology"}},
        {"id": "b", "name": "cancel_booking", "args": {"slot_id": 999}},
    ]
    client = _ScriptedClient([combo, _ai_text("Done.")])
    with _patch_client_with(client):
        out = agent_loop_node({"user_input": "book and cancel"})
    # The booking artifact must survive; the failed cancel doesn't overwrite.
    assert (out.get("appointment") or {}).get("confirmation_no") == "AGS-OK"


# ---------- tool→intent label mapping ----------

def test_tool_to_intent_mapping_covers_all_tools():
    """Every tool name must map to a real Intent literal in state.py.

    `schedule` and `audit` were promoted to first-class intents in PR #6
    rather than aliased to `booking`/`records` — see state.Intent comment.
    """
    from typing import get_args

    from nodes.agent_tools import TOOL_FUNCTIONS, tool_to_intent
    from state import Intent
    valid = set(get_args(Intent))
    for name in TOOL_FUNCTIONS:
        intent = tool_to_intent(name)
        assert intent in valid, (
            f"tool {name} maps to '{intent}' which is not a real Intent literal. "
            f"Valid intents: {sorted(valid)}"
        )


# ---------- dispatcher actually honors monkeypatches ----------
#
# Regression for PR #6 review: the prior `TOOL_FUNCTIONS = {name: fn_ref}`
# captured function references at import time, so patching the underscore
# function on the module didn't change what dispatch saw. The fix
# (`TOOL_FUNCTIONS = {name: fn_name_str}` + late resolve) is what these
# tests guard.

def test_monkeypatching_underscore_function_flows_through_dispatch(monkeypatch):
    """Verifies the dispatcher resolves functions late (via getattr on the
    module) so test monkeypatches take effect. Uses admin scope to bypass
    the new PHI policy checks — those are tested separately in
    test_agent_phi_scope.py."""
    from nodes import agent_tools

    sentinel = {"i-was-stubbed": True}
    monkeypatch.setattr(agent_tools, "_tool_find_patient",
                        lambda name: sentinel)
    result = agent_tools.dispatch("find_patient", {"name": "anyone"},
                                  scope={"role": "admin"})
    assert result is sentinel, (
        "dispatch returned the original function's result, not the stub — "
        "the late-resolve change in TOOL_FUNCTIONS is broken"
    )


def test_dispatch_missing_implementation_returns_error(monkeypatch):
    from nodes import agent_tools
    monkeypatch.delattr(agent_tools, "_tool_find_patient")
    result = agent_tools.dispatch("find_patient", {"name": "x"},
                                  scope={"role": "admin"})
    assert "missing" in result.get("error", "").lower()


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
