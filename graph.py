"""LangGraph workflow assembly for the Healthcare Assistant.

Layout (parallel mode, the default):

                                START
                                  │
                                  ▼
                          classify_intent
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
   appointment_fanout      records_node            medical_search_node
        │                         │                         │
        ├─→ booking_node          │                         │
        ├─→ history_node          │                         │
        │   (parallel)            │                         │
        └────────┬────────────────┴─────────────────────────┘
                 ▼
          compose_response
                 │
                 ▼
                END

Multi-intent queries (e.g., "book a nephrologist AND summarize latest treatments")
fan out into parallel branches that converge on the composer. LangGraph waits
for all incoming edges before executing a node, so the merge is implicit.

Single-intent queries take just one branch and converge on compose. The graph
is the same shape either way; only the conditional edges differ.

Persistence: SqliteSaver keyed by thread_id (typically a patient_id) keeps
state across Streamlit reruns.
"""
from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import END, START, StateGraph

from config import Settings, load_settings
from nodes import (
    booking_node,
    classify_intent,
    compose_response_node,
    history_node,
    medical_search_node,
    records_node,
)
from state import HealthcareState

logger = logging.getLogger(__name__)


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


def build_workflow(*, settings: Optional[Settings] = None, with_checkpoint: bool = True):
    """Build and compile the Healthcare Assistant LangGraph workflow."""
    settings = settings or load_settings()

    workflow = StateGraph(HealthcareState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("booking_node", booking_node)
    workflow.add_node("records_node", records_node)
    workflow.add_node("history_node", history_node)
    workflow.add_node("medical_search_node", medical_search_node)
    workflow.add_node("compose_response", compose_response_node)

    workflow.add_edge(START, "classify_intent")

    # All branch sets — give LangGraph the full target universe
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

    # Every branch flows into compose
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
    import sys
    import json as _json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    query = sys.argv[1] if len(sys.argv) > 1 else (
        "My 70-year-old father has chronic kidney disease. "
        "Book a nephrologist for him and summarize the latest treatment methods."
    )

    graph = build_workflow(with_checkpoint=False)
    config = {"configurable": {"thread_id": "cli-demo"}}
    result = graph.invoke(
        {"user_input": query, "history": []},
        config=config,
    )

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
