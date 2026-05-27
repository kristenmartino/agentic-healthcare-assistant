"""Tool-using ReAct loop — the AGENT_MODE=react alternative to the
classifier-then-graph routing.

Why this exists: the multi-node classifier graph has a fixed intent set,
so novel queries ("what's on Dr. Nair's calendar?") fall through to
`general` and the LLM hallucinates. A tool-using loop lets the LLM pick
the right tool(s) per turn — adding a capability is one new function,
not a new node + classifier examples + graph edges.

Flow:
  1. Safety classifier runs first (unchanged — never bypassed).
  2. If not an emergency and AGENT_MODE=react, this node runs:
     - assemble Claude tool_use call with TOOL_SPECS
     - loop while stop_reason == "tool_use":
         dispatch each tool_use block, append tool_result, recall
     - on end_turn, the final assistant text IS the response
  3. Skips the composer entirely — Claude is its own composer here.

We require Anthropic for this path (Claude is the only common provider
with first-class tool use). When the provider is stub/groq/openai/etc,
the route_after_safety helper falls back to the classifier graph.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from nodes.agent_tools import TOOL_SPECS, dispatch, tool_to_intent
from state import HealthcareState

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are an agentic healthcare assistant operating on \
behalf of a clinic. You have access to tools that read and write the \
patient EHR, the appointment scheduling system, and a medical search \
service. Your job is to answer the user's request using the right \
combination of tools, then compose a concise, faithful summary of what \
the tools returned.

# General behavior

- Pick the tool(s) that answer the user's actual question. Often one \
tool is enough; sometimes you need two (e.g. find_patient → \
get_patient_history). Don't make extra tool calls "just to be safe" — \
they cost latency, money, and audit-log noise.
- Compose tools naturally: if a user asks to "book a cardiologist AND \
summarize CKD treatments", call book_appointment + medical_search in the \
same turn. Don't serialize work that can be parallel.
- Cite real data from tool results in your final answer — do NOT invent \
doctor names, dates, confirmation numbers, medication doses, or medical \
facts. If you find yourself writing a specific number or name that no \
tool returned, stop and reconsider.
- If a tool returned an error, surface it honestly to the user with a \
plain-language explanation and (when applicable) a suggested next step. \
Don't retry the same call with the same args — it will fail the same way.
- Always end the response with a one-line reminder that the assistant is \
informational and not a substitute for clinical care.
- Stay under 250 words in the final response. Use Markdown lists for \
multi-item results (bookings, schedule slots, search results).

# Tool reference

## Patient lookup
- `find_patient(name)` — resolve a patient name to a patient_id. Use \
this when the user names a patient but you don't yet have their ID. \
Returns null if no record matches; in that case, ask the user to clarify \
the name spelling or check whether they want to register a new patient.
- `list_patients()` — admin/clinician only. Returns the full patient \
roster. The patient-chat role cannot call this; if you're refused, do \
not retry.

## Patient records
- `get_patient_history(patient_name)` — returns a clinician-style \
synthesis of conditions, recent observations, and report excerpts. \
Prefer this over composing multiple smaller reads. Patient-chat callers \
can only fetch the active patient's history; trying another name will \
be refused at the dispatcher.
- `upsert_patient(name, age, gender, conditions, summary)` — create or \
update a record. Patient-chat can only update their OWN record; \
clinicians/admins can update any. Always confirm with the user before \
overwriting an existing summary; merge new info into the existing one \
rather than replacing it wholesale.

## Appointments
- `list_doctors(specialty=None)` — directory lookup. Use when the user \
asks "who can I see for X" or "what cardiologists are on staff".
- `get_doctor_schedule(doctor_name, days_ahead=7)` — open slots for a \
specific doctor. Use for "when is Dr. X available", "what's on Dr. Y's \
calendar". Other-patient identifiers in slot results are masked for the \
patient-chat role — surface the time and doctor, not booked-by ids.
- `book_appointment(patient_name, specialty, preferred_date=None)` — \
books the earliest matching slot. If the user gave a date, pass it \
through. Confirm the doctor + start_time + confirmation_no back in your \
final response.
- `list_my_bookings(patient_id=None)` — current and upcoming bookings \
for a patient. For patient-chat callers, patient_id is auto-scoped — \
don't pass it. For clinicians/admins, pass the specific patient_id you \
want.
- `cancel_booking(slot_id=None, confirmation_no=None)` — cancel a \
booking. Patient-chat can only cancel their OWN bookings; trying \
another patient's slot will be refused. If the user said "cancel that \
appointment", look at the prior-turn appointment hint in your context \
for the slot_id / confirmation_no.

## Knowledge
- `medical_search(query, top_k=4)` — RAG-backed medical search with \
citations. Returns synthesis + indexed sources. Use for "what is X", \
"how is X treated", "latest research on Y". Always cite the returned \
sources by index ([1], [2]) when you quote them.

## Audit
- `get_audit_log(patient_id=None, limit=20)` — PHI access log. \
Patient-chat callers see only their own access events; clinicians/admins \
see whatever they pass.

# Safety and scope

- This system handles synthetic PHI from a small FHIR fixture (5 test \
patients: Anjali Mehra, David Thompson, Ramesh Kulkarni, Rebeca Nagle, \
Priya Narayan). Treat it like real PHI: never speculate beyond what \
tools return; never compose responses that include identifiers for \
patients other than the active one.
- Emergencies are routed away from you BEFORE you see the message. If a \
user describes acute symptoms anyway (chest pain, stroke signs, suicidal \
ideation), pause tool use, tell them to call emergency services, then \
add the standard disclaimer.
- You are NOT a clinician. Do not prescribe, dose, or diagnose. You \
can summarize, search, and route — that's it."""


_DISCLAIMER = (
    "ℹ️ This assistant provides informational support only and is not a "
    "substitute for advice from a licensed clinician."
)


def _get_client():
    """Return a configured ChatAnthropic for streaming tool-use calls.

    Delegates to llm.build_anthropic_client so the API key + model name
    + import-error handling live in one place. The agent uses its own
    temperature/max_tokens (richer composition than the single-turn
    classifier calls in llm.chat()).
    """
    from config import load_settings
    from llm import LLMUnavailable, build_anthropic_client
    settings = load_settings()
    if settings.llm_provider != "anthropic":
        raise RuntimeError(
            f"agent_loop requires the anthropic provider; got {settings.llm_provider}. "
            "Set ANTHROPIC_API_KEY or use AGENT_MODE=graph."
        )
    try:
        return build_anthropic_client(temperature=0.2, max_tokens=1024, timeout=30)
    except LLMUnavailable as exc:
        raise RuntimeError(str(exc)) from exc


def _to_text(content: Any) -> str:
    """Flatten a LangChain AIMessage.content (str or list-of-blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content) if content else ""


def _extract_tool_calls(message: Any) -> list[dict]:
    """Extract tool_use blocks from a LangChain AIMessage.

    LangChain normalises Anthropic's `tool_use` content blocks into the
    `tool_calls` attribute: list of {name, args, id}. We use that when
    present, fall back to scanning the raw content.
    """
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return [{"id": tc.get("id"), "name": tc.get("name"),
                 "args": tc.get("args") or {}} for tc in tool_calls]
    out: list[dict] = []
    for block in message.content if isinstance(message.content, list) else []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            out.append({"id": block.get("id"), "name": block.get("name"),
                        "args": block.get("input") or {}})
    return out


# Cap the number of tool-use turns so a runaway agent can't burn the API budget.
_MAX_TURNS = 6

# Cap how many prior conversation turns we replay into the message list.
# At ~150 tokens/turn this is ~1.2k tokens of context budget — well under
# the prompt-caching floor's tax band.
_HISTORY_TURN_CAP = 8


def _stream_and_accumulate(client: Any, messages: list) -> Any:
    """Run client.stream(messages) and concatenate chunks into one
    final message. Falls back to client.invoke() if the client doesn't
    support streaming (e.g. a test stub).

    LangGraph's stream_mode='messages' captures the per-chunk events
    emitted by `.stream()` and forwards them to any SSE consumer that
    subscribed at the workflow boundary. The accumulator preserves the
    same final AIMessage shape `.invoke()` returns so the rest of the
    loop (tool_calls extraction, content flattening) is unchanged.
    """
    stream_fn = getattr(client, "stream", None)
    if not callable(stream_fn):
        return client.invoke(messages)
    accumulated = None
    for chunk in stream_fn(messages):
        accumulated = chunk if accumulated is None else accumulated + chunk
    return accumulated if accumulated is not None else client.invoke(messages)


def _accumulate_state(
    artifacts: dict, tool_name: str, tool_args: dict, tool_result: Any,
) -> None:
    """Map a single tool result into the legacy HealthcareState fields.

    The Streamlit panel, FastAPI `/chat` `done` payload, deterministic eval,
    and Next.js artifact rows ALL read these fields. Without this mapping
    the agent path looks like it works (Claude's prose mentions the booking)
    but the structured-state contract is silently broken.

    Mutates `artifacts` in place. Skips updates when the tool errored so a
    failed call doesn't overwrite a successful one from the same turn.
    """
    if not isinstance(tool_result, (dict, list)) or (
        isinstance(tool_result, dict) and tool_result.get("error")
    ):
        return

    if tool_name == "book_appointment" and isinstance(tool_result, dict):
        artifacts["appointment"] = tool_result

    elif tool_name == "cancel_booking" and isinstance(tool_result, dict):
        # Keep the appointment slot in state but mark it as cancelled so the
        # UI can render a "cancelled X" artifact row.
        artifacts["appointment"] = {**tool_result, "action": "cancelled"}

    elif tool_name == "upsert_patient" and isinstance(tool_result, dict):
        artifacts["record_change"] = tool_result
        # Carry forward the patient_id so subsequent turns / the trace log
        # have the right reference.
        if tool_result.get("patient_id"):
            artifacts["patient_id"] = tool_result["patient_id"]

    elif tool_name == "find_patient" and isinstance(tool_result, dict):
        if tool_result.get("patient_id"):
            artifacts["patient_id"] = tool_result["patient_id"]
        if tool_result.get("name"):
            artifacts["patient_name"] = tool_result["name"]

    elif tool_name == "get_patient_history" and isinstance(tool_result, dict):
        # Preferred path (fix #7): legacy history_node was delegated, so we
        # already have a clinician-style synthesis that cites the report
        # PDFs via FAISS. Use it verbatim. Fall back to a synthesized
        # blurb if delegation failed.
        if tool_result.get("history_summary"):
            artifacts["history_summary"] = tool_result["history_summary"]
        else:
            rec = tool_result.get("record") or {}
            conds = tool_result.get("conditions") or []
            obs = tool_result.get("observations") or []
            parts: list[str] = []
            if rec.get("name"):
                parts.append(f"**{rec['name']}**" + (
                    f" — {rec.get('age')} {rec.get('gender', '')}" if rec.get("age") else ""
                ))
            if rec.get("summary"):
                parts.append(rec["summary"])
            if conds:
                from tools.fhir_client import condition_summary
                cond_text = condition_summary(conds)
                if cond_text:
                    parts.append(f"Active conditions: {cond_text}")
            if obs:
                obs_lines = [
                    f"{o.get('name')}: {o.get('value')} {o.get('unit') or ''}".strip()
                    for o in obs[:4]
                ]
                parts.append("Recent observations — " + "; ".join(obs_lines))
            if parts:
                artifacts["history_summary"] = "\n".join(parts)

    elif tool_name == "medical_search" and isinstance(tool_result, dict):
        # Fix #7: tool now delegates to medical_search_node, so we get
        # cited synthesis + indexed sources in the same shape graph mode
        # produces. medical_info[0] is the synthesis pseudo-entry; rest
        # are raw results.
        if tool_result.get("medical_info"):
            artifacts["medical_info"] = tool_result["medical_info"]
        if tool_result.get("sources"):
            artifacts["sources"] = tool_result["sources"]

    elif tool_name == "get_doctor_schedule" and isinstance(tool_result, dict):
        artifacts["schedule_results"] = tool_result

    elif tool_name == "list_my_bookings" and isinstance(tool_result, list):
        artifacts["bookings_results"] = tool_result

    elif tool_name == "list_doctors" and isinstance(tool_result, list):
        artifacts["doctor_results"] = tool_result

    elif tool_name == "list_patients" and isinstance(tool_result, list):
        artifacts["patient_listing"] = tool_result

    elif tool_name == "get_audit_log" and isinstance(tool_result, list):
        artifacts["audit_results"] = tool_result


def agent_loop_node(state: HealthcareState) -> dict:
    """Single-node alternative to classify→branches→compose."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    from llm import system_message_with_cache_control

    user_input = (state.get("user_input") or "").strip()
    if not user_input:
        return {"response": "", "intents": ["general"]}

    client = _get_client().bind_tools(TOOL_SPECS)

    # Surface the active patient context in the system prompt so Claude knows
    # which patient_name to pass to find_patient / get_patient_history.
    sys_prompt = _SYSTEM_PROMPT
    if state.get("patient_name"):
        sys_prompt += (
            f"\n\nThe user has the following patient selected in the UI: "
            f"`{state['patient_name']}` (patient_id `{state.get('patient_id')}`). "
            "Default to this patient unless the user explicitly names another."
        )

    # Wrap the long, stable system prompt with cache_control=ephemeral so
    # multi-turn sessions reuse it at ~10% input cost. The chat() path's
    # _to_anthropic_messages applies the same rule; this is the parallel
    # treatment for the agent_loop's raw LangChain message path.
    messages: list = [system_message_with_cache_control(sys_prompt)]

    # Thread bounded conversation history so follow-ups like
    # "cancel that appointment" resolve from prior turns rather than from
    # lucky text-matching in the current input. Cap at the last 8 turns
    # to keep prompt tokens bounded.
    prior_turns = state.get("history") or []
    for h in prior_turns[-_HISTORY_TURN_CAP:]:
        role = h.get("role")
        text = (h.get("content") or "").strip()
        if not text:
            continue
        if role == "user":
            messages.append(HumanMessage(content=text))
        elif role == "assistant":
            messages.append(AIMessage(content=text))

    # If the most recent assistant message wrote a booking confirmation
    # to structured state on the prior turn, include it as a SystemMessage
    # hint so the model can resolve "that appointment" without scraping
    # text for AGS-… numbers. This is the deterministic anchor for the
    # cancel-follow-up regression test.
    last_appt = state.get("appointment")
    if last_appt and last_appt.get("action") != "cancelled":
        hint = (
            "Prior turn produced this appointment (use these IDs verbatim "
            "if the user refers to 'that appointment', 'my booking', etc.): "
            f"slot_id={last_appt.get('slot_id')}, "
            f"confirmation_no={last_appt.get('confirmation_no')!r}, "
            f"doctor={last_appt.get('doctor_name')!r}, "
            f"start_time={last_appt.get('start_time')!r}."
        )
        messages.append(SystemMessage(content=hint))

    messages.append(HumanMessage(content=user_input))

    tool_log: list[dict] = []
    intents_seen: set[str] = set()
    # Accumulator for the structured-state contract (HealthcareState fields
    # that the UI panels, eval, and audit log all read).
    artifacts: dict[str, Any] = {}

    # PHI scope passed into the dispatcher on every tool call. Fail-closed
    # default is patient_chat — the most restrictive role. The Streamlit
    # Doctor View / MCP paths can override by passing role="clinician" or
    # "admin" in state when they invoke the workflow.
    scope: dict[str, Any] = {
        "actor": "agent",
        "role": state.get("role") or "patient_chat",
        "patient_id": state.get("patient_id"),
        "patient_name": state.get("patient_name"),
    }

    for turn in range(_MAX_TURNS):
        try:
            # Use .stream() rather than .invoke() so LangGraph's
            # stream_mode='messages' instrumentation captures the
            # token deltas as they're emitted (rather than seeing the
            # whole AIMessage in one shot at the end of .invoke()).
            # We accumulate the chunks into a final AIMessage with the
            # same shape .invoke() would have returned — tool_calls
            # attribute, content (str or list of blocks), etc.
            # Tool-use turns stream the brief "I'll look that up"
            # preamble Claude often emits before tool calls; the SSE
            # filter in api/main.py turns those text deltas into
            # tokens in the chat bubble, matching graph mode's UX.
            response = _stream_and_accumulate(client, messages)
        except Exception as exc:
            logger.exception("agent_loop LLM call failed on turn %d", turn)
            return {
                "response": (
                    "I had trouble reaching the model just now. Please try again "
                    "in a moment.\n\n" + _DISCLAIMER
                ),
                "intents": ["general"],
                "error": f"LLM call failed: {exc}",
                "tool_log": tool_log,
                **artifacts,  # preserve anything we did manage to gather
            }

        tool_calls = _extract_tool_calls(response)
        if not tool_calls:
            # End of conversation — Claude returned plain text.
            text = _to_text(response.content).strip()
            if _DISCLAIMER not in text:
                text = (text + "\n\n" + _DISCLAIMER).strip()
            return {
                "response": text,
                "intents": sorted(intents_seen) or ["general"],
                "tool_log": tool_log + [{
                    "node": "agent_loop",
                    "turn": turn,
                    "stop_reason": "end_turn",
                    "tool_calls": 0,
                }],
                **artifacts,
            }

        # Execute every tool_use block in this turn, in order.
        messages.append(response)
        for call in tool_calls:
            name = call["name"]
            args = call["args"]
            logger.info("agent_loop turn=%d → %s(%s)", turn, name,
                        json.dumps(args, default=str)[:120])
            result = dispatch(name, args, scope=scope)
            intents_seen.add(tool_to_intent(name))
            _accumulate_state(artifacts, name, args, result)
            tool_log.append({
                "node": "agent_loop",
                "turn": turn,
                "tool": name,
                "args": args,
                "result_excerpt": json.dumps(result, default=str)[:300],
            })
            messages.append(ToolMessage(
                content=json.dumps(result, default=str),
                tool_call_id=call["id"] or f"call-{turn}-{name}",
            ))
        # Loop continues to give Claude the tool results.

    # Exceeded turn budget without an end_turn.
    return {
        "response": (
            "I worked through a few tool calls but couldn't reach a final "
            "answer in the turn budget. Please rephrase or break this into "
            "smaller asks.\n\n" + _DISCLAIMER
        ),
        "intents": sorted(intents_seen) or ["general"],
        "tool_log": tool_log + [{"node": "agent_loop",
                                  "stop_reason": "max_turns_exceeded"}],
        "error": "Exceeded max tool-use turns",
        **artifacts,
    }


def is_agent_mode_active() -> bool:
    """True when AGENT_MODE=react AND Anthropic is the active provider.

    NOTE: default is now `graph` (the legacy classifier path). React mode
    is opt-in until the structured-state contract, PHI scoping, and
    conversation-history threading are all in place — see PR review for
    the followup plan. Set AGENT_MODE=react to opt in.
    """
    import os

    from config import load_settings
    mode = os.getenv("AGENT_MODE", "graph").lower()
    if mode != "react":
        return False
    return load_settings().llm_provider == "anthropic"
