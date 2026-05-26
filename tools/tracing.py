"""Workflow tracing — LangSmith if configured, JSONL otherwise.

Real observability has two audiences:

1. **The developer** debugging "why did this query take 12 seconds?" — wants
   a flame graph of every LLM and tool call. LangSmith is the right tool;
   set `LANGCHAIN_API_KEY` + `LANGCHAIN_TRACING_V2=true` (auto-picked up by
   LangChain/LangGraph) and traces show up at https://smith.langchain.com.

2. **A portfolio reviewer** without a LangSmith account who still wants to
   see the assistant's behavior over time. For them we keep a local
   append-only JSONL at `logs/runs.jsonl` — one row per workflow
   invocation, with timing, intents, error, backend choices. The Streamlit
   "Traces" page reads it for an at-a-glance recent-activity view.

Both paths are always-on (no per-call enable flag): the LangSmith
auto-tracing kicks in when the env vars are set, and the JSONL writer is
best-effort (failures never raise).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

# Single write lock so concurrent Streamlit threads can't tangle JSONL rows.
_write_lock = threading.Lock()


# ---------- LangSmith detection ----------

def langsmith_enabled() -> bool:
    """True when LangSmith env vars are present.

    LangChain/LangGraph honor `LANGCHAIN_API_KEY` and `LANGCHAIN_TRACING_V2`
    automatically — we don't have to wire anything; just report whether
    they're set so the UI can show a "View traces in LangSmith →" link.
    """
    return bool(os.getenv("LANGCHAIN_API_KEY")) and (
        os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes")
        # LangChain accepts both V2 and the newer "LANGSMITH_TRACING" alias.
        or os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")
    )


def langsmith_project() -> str | None:
    return os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT")


# ---------- JSONL run log ----------

def _trace_log_path() -> Path:
    from config import load_settings
    return Path(load_settings().trace_log_path)


@contextmanager
def trace_run(
    thread_id: str,
    user_input: str,
    actor: str = "patient_chat",
) -> Iterator[dict[str, Any]]:
    """Context manager that records one workflow invocation to the JSONL log.

    Yields a mutable dict the caller fills in (intents, error, state-derived
    fields, etc.). On exit, writes one row to logs/runs.jsonl.

    Usage:
        with trace_run(thread_id, user_input) as evt:
            state = workflow.invoke(...)
            evt["intents"] = state.get("intents")
            evt["is_emergency"] = state.get("is_emergency", False)
    """
    started = perf_counter()
    ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
    event: dict[str, Any] = {
        "ts": ts,
        "thread_id": thread_id,
        "actor": actor,
        "user_input": user_input[:500],  # cap PII bleed into logs
        "langsmith_enabled": langsmith_enabled(),
    }
    try:
        yield event
    except Exception as exc:
        event["error"] = str(exc)
        event["error_type"] = type(exc).__name__
        raise
    finally:
        event["latency_seconds"] = round(perf_counter() - started, 3)
        try:
            _append_jsonl(event)
        except Exception as exc:  # never let trace logging break the request
            logger.warning("Trace JSONL write failed: %s", exc)


def _append_jsonl(event: dict[str, Any]) -> None:
    path = _trace_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, default=str, separators=(",", ":"))
    with _write_lock, path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def tail_runs(limit: int = 50, path: Path | None = None) -> list[dict[str, Any]]:
    """Read recent run events, newest-first. Used by pages/4_Traces.py."""
    path = path or _trace_log_path()
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Could not read trace log %s: %s", path, exc)
        return []
    out: list[dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= limit:
            break
    return out


def run_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Rollup stats for the dashboard strip."""
    if not events:
        return {"total": 0, "p50_latency_seconds": None, "p95_latency_seconds": None,
                "error_rate": 0.0, "emergency_count": 0}
    latencies = sorted(e.get("latency_seconds") or 0 for e in events)
    errors = sum(1 for e in events if e.get("error"))
    emergencies = sum(1 for e in events if e.get("is_emergency"))
    n = len(latencies)
    p50 = latencies[n // 2]
    p95 = latencies[min(int(n * 0.95), n - 1)]
    return {
        "total": len(events),
        "p50_latency_seconds": round(p50, 3),
        "p95_latency_seconds": round(p95, 3),
        "error_rate": round(errors / len(events), 3),
        "emergency_count": emergencies,
    }
