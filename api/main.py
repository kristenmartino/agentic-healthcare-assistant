"""FastAPI service wrapping the LangGraph workflow.

Single-purpose: expose the existing graph, EHR tools, audit log, and trace
log to a Next.js frontend. Reuses every existing module — no logic is
duplicated here. The frontend talks SSE for chat (token streaming) and
plain JSON for the dashboard reads (patients, bookings, audit, traces).

Run locally:
    uvicorn api.main:app --reload --port 8000

Deploy to Fly.io:
    flyctl deploy
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Allow `python -m api.main` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_settings
from graph import build_workflow
from tools.appointments import (
    ensure_future_slots,
    get_specialty_stats,
    list_recent_bookings,
)
from tools.audit import audit_summary, query_audit
from tools.ehr import find_patient_by_name, get_patient_clinical_context, list_patients
from tools.medical_search import configured_backend
from tools.tracing import langsmith_enabled, langsmith_project, run_summary, tail_runs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------- app + middleware ----------

app = FastAPI(
    title="Agentic Healthcare Assistant API",
    version="1.0.0",
    description=(
        "FastAPI wrapper around the LangGraph workflow. Streams chat via SSE; "
        "exposes patients, bookings, audit log, and traces as plain JSON."
    ),
)

# CORS — the Next.js frontend on Vercel needs to call us cross-origin.
# Lock to known origins in production via the CORS_ALLOW_ORIGINS env var,
# default permissive for local dev.
_allow_origins = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------- shared singletons ----------

@app.on_event("startup")
def _bootstrap() -> None:
    """Build the workflow once at startup; pre-warm the slot table."""
    settings = load_settings()
    # Top up the appointments DB so the first booking call doesn't fail
    # against a stale slot table.
    try:
        ensure_future_slots(settings.appointments_db_path)
    except Exception as exc:
        logger.warning("ensure_future_slots failed at startup: %s", exc)
    # Eagerly build the LangGraph workflow so the first request doesn't pay
    # the import cost.
    app.state.workflow = build_workflow(with_checkpoint=True)
    app.state.settings = settings
    logger.info("API ready. LLM=%s EHR=%s Search=%s",
                settings.llm_provider, settings.ehr_backend,
                configured_backend(settings.tavily_api_key))


def _settings():
    return app.state.settings or load_settings()


# ---------- health + config ----------

@app.get("/")
def root() -> dict:
    """Friendly root — a bare-URL visitor sees a pointer, not a 404."""
    return {
        "name": "agentic-healthcare-api",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "/config", "/health",
            "/patients", "/patients/{id}", "/patients/search/by-name",
            "/bookings/recent", "/bookings/specialty-stats",
            "/audit", "/audit/summary",
            "/traces",
            "POST /chat (SSE)",
        ],
    }


@app.get("/health")
def health() -> dict:
    """Liveness probe. Used by Fly.io / Docker healthcheck."""
    s = _settings()
    return {
        "status": "ok",
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model,
        "ehr_backend": s.ehr_backend,
        "search_backend_intended": configured_backend(s.tavily_api_key),
        "langsmith_enabled": langsmith_enabled(),
    }


@app.get("/config")
def get_config() -> dict:
    """Frontend-visible runtime config. NO secrets — only labels.

    Drives the sidebar's status row (LLM / EHR / Search) and the LangSmith
    badge. Never returns the API keys themselves.
    """
    s = _settings()
    return {
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model,
        "ehr_backend": s.ehr_backend,
        "fhir_base_url": s.fhir_base_url if s.ehr_backend == "fhir" else None,
        "search_backend_intended": configured_backend(s.tavily_api_key),
        "tavily_configured": bool(s.tavily_api_key),
        "langsmith_enabled": langsmith_enabled(),
        "langsmith_project": langsmith_project(),
        "prompt_caching_enabled": s.enable_prompt_caching,
    }


# ---------- patients ----------

@app.get("/patients")
def get_patients() -> list[dict]:
    """List all patients (id, name, age, gender, summary)."""
    return list_patients(_settings(), actor="api")


@app.get("/patients/{patient_id}")
def get_patient(patient_id: str) -> dict:
    """One patient by patient_id, with clinical context (Conditions + Observations)
    when the EHR backend has them.
    """
    # Search by id by listing and filtering — small dataset; cheap.
    patients = list_patients(_settings(), actor="api")
    match = next((p for p in patients if p.get("patient_id") == patient_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    ctx = get_patient_clinical_context(patient_id, _settings(), actor="api")
    return {**match, "clinical": ctx}


@app.get("/patients/search/by-name")
def search_patient_by_name(name: str = Query(..., min_length=1)) -> dict | None:
    return find_patient_by_name(name, _settings(), actor="api")


# ---------- bookings ----------

@app.get("/bookings/recent")
def get_recent_bookings(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    return list_recent_bookings(_settings().appointments_db_path, limit=limit)


@app.get("/bookings/specialty-stats")
def specialty_stats() -> list[dict]:
    """Per-specialty booked/total/utilization. Drives the Doctor View KPIs."""
    return get_specialty_stats(_settings().appointments_db_path)


# ---------- audit log ----------

@app.get("/audit")
def get_audit(
    patient_id: str | None = None,
    action_prefix: str | None = None,
    actor: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict]:
    return query_audit(
        patient_id=patient_id,
        action_prefix=action_prefix,
        actor=actor,
        limit=limit,
        db_path=_settings().audit_db_path,
    )


@app.get("/audit/summary")
def get_audit_summary() -> dict:
    return audit_summary(_settings().audit_db_path)


# ---------- workflow traces ----------

@app.get("/traces")
def get_traces(limit: int = Query(100, ge=1, le=1000)) -> dict:
    events = tail_runs(limit=limit)
    return {"events": events, "summary": run_summary(events)}


# ---------- chat (SSE) ----------

class ChatRequest(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = Field(None, description="Stable id for cross-turn memory")
    patient_id: str | None = None
    patient_name: str | None = None


@app.post("/chat")
async def chat_stream(req: ChatRequest):
    """Stream the workflow's node-progress and final-response tokens as SSE.

    Event types emitted:
      - event: status   data: {node, label, summary}
      - event: token    data: {text}      — composer tokens only
      - event: done     data: {state}     — final accumulated state
      - event: error    data: {message}

    The frontend's EventSource consumer renders status as a progress strip
    and tokens into the chat bubble; `done` carries the full state for the
    right-hand panel (sources, audit, etc.).
    """
    workflow = app.state.workflow
    initial_state: dict = {"user_input": req.user_input, "history": []}
    if req.patient_id:
        initial_state["patient_id"] = req.patient_id
    if req.patient_name:
        initial_state["patient_name"] = req.patient_name
    thread_id = req.thread_id or "anonymous"
    config = {"configurable": {"thread_id": thread_id}}

    async def event_stream() -> AsyncIterator[bytes]:
        accumulated: dict[str, Any] = {}
        streamed_text = ""
        from tools.tracing import trace_run

        with trace_run(thread_id, req.user_input, actor="web") as trace_event:
            try:
                # LangGraph's sync .stream() needs an executor — use a thread.
                loop = asyncio.get_event_loop()
                queue: asyncio.Queue = asyncio.Queue()

                def _producer():
                    try:
                        for stream_mode, chunk in workflow.stream(
                            initial_state, config=config,
                            stream_mode=["updates", "messages"],
                        ):
                            loop.call_soon_threadsafe(queue.put_nowait,
                                                      (stream_mode, chunk))
                    except Exception as exc:
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, ("__done__", None))

                fut = loop.run_in_executor(None, _producer)

                while True:
                    stream_mode, chunk = await queue.get()
                    if stream_mode == "__done__":
                        break
                    if stream_mode == "error":
                        yield _sse("error", {"message": str(chunk)})
                        return

                    if stream_mode == "updates":
                        # chunk is {node_name: partial_state}
                        for node_name, partial in chunk.items():
                            yield _sse("status", {
                                "node": node_name,
                                "label": _node_label(node_name),
                                "summary": _summarize_partial(node_name, partial or {}),
                            })
                            if isinstance(partial, dict):
                                for k, v in partial.items():
                                    if isinstance(v, list) and isinstance(
                                            accumulated.get(k), list):
                                        accumulated[k] = accumulated[k] + v
                                    else:
                                        accumulated[k] = v

                    elif stream_mode == "messages":
                        # chunk is (AIMessageChunk, metadata)
                        try:
                            msg, metadata = chunk
                        except (TypeError, ValueError):
                            continue
                        if metadata.get("langgraph_node") != "compose_response":
                            continue
                        token = _content_to_text(getattr(msg, "content", "") or "")
                        if token:
                            streamed_text += token
                            yield _sse("token", {"text": token})

                await fut

                response = streamed_text or accumulated.get("response", "")
                # Annotate the trace event with state-derived fields.
                _raw_info = [r for r in (accumulated.get("medical_info") or [])
                             if "synthesis" not in r]
                from tools.medical_search import effective_backend
                trace_event.update({
                    "intents": accumulated.get("intents"),
                    "is_emergency": accumulated.get("is_emergency", False),
                    "patient_id": accumulated.get("patient_id"),
                    "node_count": len(accumulated.get("tool_log") or []),
                    "search_backend": effective_backend(_raw_info) if _raw_info else None,
                    "had_error": bool(accumulated.get("error")),
                })
                yield _sse("done", {
                    "response": response,
                    "intents": accumulated.get("intents"),
                    "is_emergency": accumulated.get("is_emergency", False),
                    "emergency_categories": accumulated.get("emergency_categories"),
                    "appointment": accumulated.get("appointment"),
                    "record_change": accumulated.get("record_change"),
                    "history_summary": accumulated.get("history_summary"),
                    "medical_info": accumulated.get("medical_info"),
                    "sources": accumulated.get("sources"),
                    "tool_log": accumulated.get("tool_log"),
                    "error": accumulated.get("error"),
                })
            except Exception as exc:
                logger.exception("chat stream failed")
                yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx response buffering
            "Connection": "keep-alive",
        },
    )


# ---------- helpers ----------

def _sse(event: str, data: dict) -> bytes:
    """Format one SSE event with explicit event type + JSON payload."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode()


def _node_label(node_name: str) -> str:
    return {
        "safety": "Safety check",
        "classify_intent": "Classifying intent",
        "booking_node": "Booking",
        "records_node": "Updating record",
        "history_node": "Retrieving history",
        "medical_search_node": "Medical search",
        "compose_response": "Composing response",
    }.get(node_name, node_name)


def _summarize_partial(node_name: str, partial: dict) -> str:
    if node_name == "classify_intent" and partial.get("intents"):
        return ", ".join(partial["intents"])
    if node_name == "safety" and partial.get("is_emergency"):
        return "🚨 emergency — " + ", ".join(partial.get("emergency_categories") or [])
    if node_name == "booking_node" and partial.get("appointment"):
        appt = partial["appointment"]
        return f"{appt.get('doctor_name', '?')} ({appt.get('specialty', '?')})"
    if node_name == "medical_search_node" and partial.get("medical_info"):
        return f"{len(partial['medical_info']) - 1} sources"  # -1 for synthesis pseudo-entry
    return ""


_ChannelType = Literal["text", "image_url", "tool_use", "tool_result"]


def _content_to_text(content: Any) -> str:
    """Flatten an AIMessageChunk content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)
