"""LangGraph workflow assembly for the Healthcare Assistant.

Two reasoning strategies share the same safety pre-classifier:

                                START
                                  │
                                  ▼
                              safety  ───(is_emergency)──► END
                                  │  (hardcoded 911 template; LLM never runs)
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              AGENT_MODE=react              AGENT_MODE=graph  (default)
              (opt-in; needs Anthropic)         │
                    │                           ▼
                    ▼                     classify_intent
              agent_loop                       │
        ┌────────── │ ─────────┐  ┌─────┬──────┴─────┬───────────┐
        │ Claude tool-use      │  ▼     ▼            ▼           ▼
        │ loop, 11 tools (book │ booking records   history   medical_search
        │ /cancel/schedule/   │  │     │            │           │
        │ list/find/history/  │  └─────┴──────┬─────┴───────────┘
        │ audit/search/...).  │               ▼
        │ Dispatcher enforces │         compose_response
        │ PHI scope per role. │               │
        └─────────────────────┘               │
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                                END

graph (default): predictable classifier-then-fan-out. Preserves the
structured-state contract the eval, audit log, and UI artifact panels
depend on. Multi-intent fan-out runs branches in parallel; LangGraph
waits for all incoming edges before executing the composer, so the
merge is implicit.

react (opt-in): a single agent_loop node calls Claude with 11 tool
schemas and lets it pick + compose tools per turn. Closes the
fixed-intent ceiling — new capabilities ("what's on Dr. X's calendar",
"cancel that appointment", "who accessed my records") work without
adding classifier intents. Same structured-state contract: an
accumulator inside agent_loop maps tool results back into
appointment / record_change / history_summary / medical_info /
sources / schedule_results / bookings_results / audit_results /
doctor_results so every downstream consumer (Streamlit panel,
FastAPI /chat done event, deterministic eval, Next.js artifact rows)
keeps working regardless of which strategy ran.

Persistence: SqliteSaver keyed by thread_id (typically a patient_id)
keeps state across Streamlit reruns and shared between both strategies.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from config import Settings, load_settings
from nodes.agent_loop import agent_loop_node, is_agent_mode_active
from nodes.booking import booking_node
from nodes.classifier import classify_intent
from nodes.composer import compose_response_node
from nodes.history import history_node
from nodes.medical_search_node import medical_search_node
from nodes.records import records_node
from nodes.safety import safety_node
from state import HealthcareState

logger = logging.getLogger(__name__)


def _route_after_safety(state: HealthcareState) -> str:
    """First-fork: emergency → END (skip everything), otherwise pick the
    reasoning strategy.

    `react` (default when Anthropic is the active provider): single
    agent_loop node that picks tools dynamically. Adds new capabilities
    without classifier edits. Better at novel queries.

    `graph` (legacy, also the fallback when Anthropic isn't configured):
    fixed classifier → branch fan-out → composer. Predictable, easy to
    eval, but limited to the 5 intents.
    """
    if state.get("is_emergency"):
        return "__skip_to_end__"
    return "agent_loop" if is_agent_mode_active() else "classify_intent"


def _route_after_classify(state: HealthcareState) -> list[str]:
    """Return the list of next-node names based on classified intent(s).

    LangGraph's add_conditional_edges supports either str or list-of-str returns;
    a list triggers parallel fan-out.
    """
    intents = state.get("intents") or [state.get("intent", "general")]

    targets: list[str] = []
    if "booking" in intents:
        targets.append("booking_node")
    if "records" in intents:
        targets.append("records_node")
    if "history" in intents:
        targets.append("history_node")
    if "medical_search" in intents:
        targets.append("medical_search_node")

    # If only general, route straight to compose with empty branches
    if not targets:
        targets.append("compose_response")

    return targets


def build_workflow(*, settings: Settings | None = None, with_checkpoint: bool = True):
    """Build and compile the Healthcare Assistant LangGraph workflow."""
    settings = settings or load_settings()

    workflow = StateGraph(HealthcareState)

    workflow.add_node("safety", safety_node)
    workflow.add_node("agent_loop", agent_loop_node)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("booking_node", booking_node)
    workflow.add_node("records_node", records_node)
    workflow.add_node("history_node", history_node)
    workflow.add_node("medical_search_node", medical_search_node)
    workflow.add_node("compose_response", compose_response_node)

    workflow.add_edge(START, "safety")

    # Three-way fork after safety:
    #   - emergency → END (safety has already populated `response`)
    #   - react mode + anthropic → agent_loop (single node, tool-using)
    #   - everything else → classifier graph (legacy path)
    workflow.add_conditional_edges(
        "safety",
        _route_after_safety,
        {
            "agent_loop": "agent_loop",
            "classify_intent": "classify_intent",
            "__skip_to_end__": END,
        },
    )
    # agent_loop produces its own final response — go straight to END.
    workflow.add_edge("agent_loop", END)

    # ---- legacy classifier-graph branches ----
    branch_targets = {
        "booking_node": "booking_node",
        "records_node": "records_node",
        "history_node": "history_node",
        "medical_search_node": "medical_search_node",
        "compose_response": "compose_response",
    }
    workflow.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        branch_targets,
    )

    for branch in ("booking_node", "records_node", "history_node", "medical_search_node"):
        workflow.add_edge(branch, "compose_response")

    workflow.add_edge("compose_response", END)

    checkpointer = (
        _build_checkpointer(settings)
        if with_checkpoint and settings.enable_persistence
        else None
    )

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


def _build_checkpointer(settings: Settings):
    """SqliteSaver for cross-session memory.

    NOTE on API: in `langgraph-checkpoint-sqlite>=2.0`, `SqliteSaver.from_conn_string`
    is a context manager (yields a saver, then closes the connection). It can't
    be used inline because the graph outlives any `with` block. Instead we open
    a long-lived `sqlite3` connection ourselves and construct the saver
    directly. `check_same_thread=False` is required because Streamlit serves
    HTTP requests across threads.

    Returns None if the package isn't installed; the graph still works without
    persistence.
    """
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        logger.warning(
            "SqliteSaver not available (%s); install `langgraph-checkpoint-sqlite` "
            "for persistence. Continuing without checkpointing.",
            exc,
        )
        return None

    try:
        from pathlib import Path
        Path(settings.sqlite_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(settings.sqlite_checkpoint_path, check_same_thread=False)
        return SqliteSaver(conn)
    except Exception as exc:
        logger.warning(
            "Could not initialize SqliteSaver (%s); continuing without persistence.",
            exc,
        )
        return None


# CLI: python graph.py "your query"
if __name__ == "__main__":
    import json as _json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    query = sys.argv[1] if len(sys.argv) > 1 else (
        "My 70-year-old father has chronic kidney disease. "
        "Book a nephrologist for him and summarize the latest treatment methods."
    )

    graph = build_workflow(with_checkpoint=False)
    config = {"configurable": {"thread_id": "cli-demo"}}
    from tools.tracing import trace_run
    with trace_run("cli-demo", query, actor="cli") as trace_event:
        result = graph.invoke(
            {"user_input": query, "history": []},
            config=config,
        )
        trace_event.update({
            "intents": result.get("intents"),
            "is_emergency": result.get("is_emergency", False),
            "patient_id": result.get("patient_id"),
            "node_count": len(result.get("tool_log") or []),
            "had_error": bool(result.get("error")),
        })

    print("\n" + "=" * 70)
    print("USER:", query)
    print("=" * 70)
    print("\n--- Response ---\n")
    print(result.get("response", "(no response produced)"))
    print("\n--- Intents ---")
    print(result.get("intents", []))
    print("\n--- Tool log ---")
    for entry in result.get("tool_log", []):
        print(_json.dumps(entry, default=str)[:200])
    if result.get("error"):
        print("\n--- Error ---")
        print(result["error"])
