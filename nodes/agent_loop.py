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
service.

Behavior:
- Pick the tool(s) that answer the user's actual question. Often one tool \
is enough; sometimes you need two (e.g. find_patient → get_patient_history).
- Compose tools naturally: if a user asks to "book a cardiologist AND \
summarize CKD treatments", call book_appointment + medical_search.
- For schedule/calendar questions ("when is Dr. X available", "what's on \
Dr. Y's calendar"), use get_doctor_schedule.
- Cite real data from tool results in your final answer — do NOT invent \
doctor names, dates, confirmation numbers, or medical facts. If a tool \
returned an error, surface that honestly to the user.
- Always end the response with a one-line reminder that the assistant is \
informational and not a substitute for clinical care.
- Stay under 250 words in the final response.

Synthetic data: all patient names you see come from a 5-patient FHIR \
fixture (Anjali Mehra, David Thompson, Ramesh Kulkarni, Rebeca Nagle, \
Priya Narayan). No real PHI."""


_DISCLAIMER = (
    "ℹ️ This assistant provides informational support only and is not a "
    "substitute for advice from a licensed clinician."
)


def _get_client():
    """Return a configured ChatAnthropic for streaming tool-use calls."""
    from config import load_settings
    settings = load_settings()
    if settings.llm_provider != "anthropic":
        raise RuntimeError(
            f"agent_loop requires the anthropic provider; got {settings.llm_provider}. "
            "Set ANTHROPIC_API_KEY or use AGENT_MODE=graph."
        )
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise RuntimeError(
            "langchain-anthropic not installed; pip install langchain-anthropic"
        ) from exc
    return ChatAnthropic(
        api_key=settings.anthropic_api_key,
        model_name=settings.llm_model,
        temperature=0.2,
        max_tokens=1024,
        timeout=30,
    )


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
        # Build a compact text history_summary from the structured result.
        # This is the naive version — fix #7 will swap in the legacy
        # history_node's FAISS+LLM synthesis for proper parity.
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

    elif tool_name == "medical_search" and isinstance(tool_result, list):
        # Match the legacy medical_search_node shape:
        #   medical_info = [{"synthesis": ...}] + raw_results
        #   sources      = [{"index": i, "title", "url", "source"}, ...]
        # We don't synthesize here — fix #7 will route this through the
        # legacy node so we get the same cited synthesis as graph mode.
        artifacts["medical_info"] = list(tool_result)
        artifacts["sources"] = [
            {"index": i + 1, "title": r.get("title", ""),
             "url": r.get("url", ""), "source": r.get("source", "unknown")}
            for i, r in enumerate(tool_result)
        ]

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
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

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

    messages: list = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_input),
    ]

    tool_log: list[dict] = []
    intents_seen: set[str] = set()
    # Accumulator for the structured-state contract (HealthcareState fields
    # that the UI panels, eval, and audit log all read).
    artifacts: dict[str, Any] = {}

    for turn in range(_MAX_TURNS):
        try:
            response = client.invoke(messages)
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
            result = dispatch(name, args)
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
