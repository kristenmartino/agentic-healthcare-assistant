"""Traces — recent workflow runs.

Reads the append-only JSONL written by `tools.tracing.trace_run`. One row
per user query, with timing, intents, error, and backend choices captured
when the workflow finished. When LangSmith is configured, links out to
the full flame-graph traces there.

Filters: actor (patient_chat / cli / doctor_view), is_emergency, has-error,
search backend used. CSV export.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta

import streamlit as st

from config import load_settings
from tools.tracing import langsmith_enabled, langsmith_project, run_summary, tail_runs

st.set_page_config(
    page_title="Traces — Healthcare Assistant",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #6ee7b7;
        border-left: 4px solid #047857;
        padding: 8px 14px;
        border-radius: 6px;
        margin-bottom: 12px;
        font-size: 14px;
        color: #064e3b;
    ">
        📊 <strong>Workflow traces.</strong> Every query is logged here so you
        can see latency, intent routing, and backend choices over time.
    </div>
    """,
    unsafe_allow_html=True,
)

settings = load_settings()
st.title("Traces")
st.caption(f"Source: `{settings.trace_log_path}`")

# ---------- LangSmith link ----------

if langsmith_enabled():
    project = langsmith_project() or "(default project)"
    st.success(
        f"✅ LangSmith tracing is enabled for project **{project}**. "
        "Open https://smith.langchain.com/ for the full per-call flame graphs."
    )
else:
    st.info(
        "ℹ️ LangSmith tracing is **not** configured. Set `LANGCHAIN_API_KEY` "
        "and `LANGCHAIN_TRACING_V2=true` in `.env` to enable per-LLM-call "
        "flame graphs in addition to this local JSONL log."
    )


# ---------- Summary strip ----------

events = tail_runs(limit=500)
summary = run_summary(events)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total runs", summary["total"])
c2.metric("p50 latency (s)", summary["p50_latency_seconds"] or 0)
c3.metric("p95 latency (s)", summary["p95_latency_seconds"] or 0)
c4.metric("Error rate", f"{summary['error_rate'] * 100:.1f}%")
c5.metric("Emergencies", summary["emergency_count"])


# ---------- Filters ----------

st.subheader("Filters")
fcol1, fcol2, fcol3, fcol4 = st.columns([2, 2, 2, 1])

with fcol1:
    actors = sorted({e.get("actor", "(unknown)") for e in events})
    actor_pick = st.selectbox("Actor", ["(all)", *actors], key="trace_actor")

with fcol2:
    backends = sorted({e.get("search_backend") for e in events if e.get("search_backend")})
    backend_pick = st.selectbox(
        "Search backend used",
        ["(all)", *backends, "(none / no search branch)"],
        key="trace_backend",
    )

with fcol3:
    emergency_pick = st.selectbox(
        "Emergency filter",
        ["(all)", "Emergencies only", "Non-emergencies only"],
        key="trace_emergency",
    )

with fcol4:
    error_only = st.checkbox("Errors only", value=False)
    days_back = st.number_input("Days back", value=7, min_value=1, max_value=365, step=1)
    since_iso = (datetime.utcnow() - timedelta(days=int(days_back))).isoformat() + "Z"


def _matches(e: dict) -> bool:
    if e.get("ts", "") < since_iso:
        return False
    if actor_pick != "(all)" and e.get("actor") != actor_pick:
        return False
    if backend_pick == "(none / no search branch)":
        if e.get("search_backend"):
            return False
    elif backend_pick != "(all)" and e.get("search_backend") != backend_pick:
        return False
    if emergency_pick == "Emergencies only" and not e.get("is_emergency"):
        return False
    if emergency_pick == "Non-emergencies only" and e.get("is_emergency"):
        return False
    if error_only and not e.get("error") and not e.get("had_error"):
        return False
    return True


filtered = [e for e in events if _matches(e)]


# ---------- Events table ----------

st.subheader(f"Events ({len(filtered)})")

if not filtered:
    st.info("No events match these filters. Run a query in the patient chat "
            "(or `python graph.py 'your question'` from the CLI) to populate "
            "this log.")
else:
    table = [
        {
            "ts": e["ts"],
            "actor": e.get("actor", ""),
            "user_input": (e.get("user_input") or "")[:80],
            "intents": ", ".join(e.get("intents") or []),
            "latency_s": e.get("latency_seconds"),
            "search": e.get("search_backend") or "—",
            "emergency": "🚨" if e.get("is_emergency") else "",
            "error": "❌" if (e.get("error") or e.get("had_error")) else "",
        }
        for e in filtered
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)

    with st.expander("Event details (JSON)", expanded=False):
        for e in filtered[:50]:
            st.code(json.dumps(e, default=str, indent=2), language="json")

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(table[0].keys()))
    writer.writeheader()
    writer.writerows(table)
    st.download_button(
        "Download CSV",
        data=buf.getvalue(),
        file_name=f"traces_{datetime.utcnow():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
    )
