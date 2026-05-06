"""Streamlit UI for the Healthcare Assistant.

Layout:
- Sidebar: patient selector (from EHR), provider info, quick links.
- Main: chat interface with conversation history.
- Right column (expanders): tool log, state trace, recent bookings.

Memory: SqliteSaver checkpointer keyed by `thread_id = patient_id`. Switching
patients in the sidebar starts a fresh conversation; switching back resumes.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime

import streamlit as st

from config import load_settings
from graph import build_workflow
from tools.appointments import list_recent_bookings
from tools.ehr_db import list_patients
from utils import (
    format_appointment_time,
    node_label,
    summarize_node_update,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------- Persistent disclaimer banner (top of every page) --------

st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #fff8e1 0%, #fff3cd 100%);
        border: 1px solid #ffe082;
        border-left: 4px solid #ff8f00;
        padding: 8px 14px;
        border-radius: 6px;
        margin-bottom: 12px;
        font-size: 14px;
        color: #5d4037;
    ">
        ⚠️ <strong>Informational only — not a substitute for clinical care.</strong>
        Synthetic data; no real PHI.
    </div>
    """,
    unsafe_allow_html=True,
)


# -------- One-time initialization --------

@st.cache_resource(show_spinner="Loading workflow ...")
def get_workflow():
    """Build and cache the LangGraph workflow."""
    return build_workflow(with_checkpoint=True)


@st.cache_resource(show_spinner=False)
def get_settings():
    return load_settings()


@st.cache_data(ttl=10, show_spinner=False)
def get_patients():
    return list_patients(get_settings().ehr_db_path)


@st.cache_data(ttl=10, show_spinner=False)
def get_recent_bookings():
    return list_recent_bookings(get_settings().appointments_db_path, limit=10)


# -------- Sidebar --------

settings = get_settings()
workflow = get_workflow()

with st.sidebar:
    st.header("🏥 Healthcare Assistant")
    st.caption(f"LLM: **{settings.llm_provider}** ({settings.llm_model})")
    if settings.llm_provider == "stub":
        st.warning(
            "Running in **stub mode** — LLM responses are templated. "
            "Set `GROQ_API_KEY` (free) or `OPENAI_API_KEY` in `.env` for real responses."
        )

    st.divider()
    st.subheader("Patient")

    patients = get_patients()
    patient_options = ["(walk-in / no patient)"] + [p["name"] for p in patients]
    selected_idx = st.selectbox(
        "Select a patient (sets thread_id for memory)",
        range(len(patient_options)),
        format_func=lambda i: patient_options[i],
        key="patient_selector",
    )

    if selected_idx == 0:
        thread_id = "walk-in"
        patient_id = None
        patient_name = None
    else:
        patient = patients[selected_idx - 1]
        thread_id = patient["patient_id"]
        patient_id = patient["patient_id"]
        patient_name = patient["name"]
        with st.expander("Selected patient details", expanded=False):
            st.json(patient)

    if st.button("🔄 Start new conversation"):
        thread_id = thread_id + "-" + str(int(time.time()))
        st.session_state["forced_thread_id"] = thread_id
        st.session_state["chat_history"] = []
        st.rerun()

    if "forced_thread_id" in st.session_state:
        thread_id = st.session_state["forced_thread_id"]

    st.divider()
    st.subheader("Quick examples")
    examples = [
        "Show me Anjali Mehra's medical history",
        "Book a cardiologist for next week",
        "What are the symptoms of pneumonia?",
        "Add a new patient: John Doe, age 45, male, hypertension",
        "My 70-year-old father has chronic kidney disease. Book a nephrologist and summarize the latest treatments.",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state["queued_query"] = ex
            st.rerun()


# -------- Main column --------

col_main, col_side = st.columns([3, 2])

with col_main:
    st.title("Conversation")
    st.caption(f"Thread: `{thread_id}`")

    # Chat history (Streamlit-native messages)
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for entry in st.session_state["chat_history"]:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    # Input — accept either typed input or a queued example click
    queued = st.session_state.pop("queued_query", None)
    user_input = st.chat_input("Ask a healthcare question...")
    if queued and not user_input:
        user_input = queued

    if user_input:
        # Show user message immediately
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Stream the workflow with per-node progress, then render the final response.
        with st.chat_message("assistant"):
            initial_state = {
                "user_input": user_input,
                "history": st.session_state["chat_history"],
            }
            if patient_id:
                initial_state["patient_id"] = patient_id
            if patient_name:
                initial_state["patient_name"] = patient_name

            config = {"configurable": {"thread_id": thread_id}}
            accumulated: dict = {}
            try:
                with st.status("Working...", expanded=True) as status:
                    for chunk in workflow.stream(
                        initial_state, config=config, stream_mode="updates"
                    ):
                        # chunk is {node_name: partial_state}
                        for node_name, partial in chunk.items():
                            label = node_label(node_name)
                            summary = summarize_node_update(node_name, partial or {})
                            st.write(f"✅ **{label}** — {summary}" if summary else f"✅ **{label}**")
                            # Merge partial into accumulated for final render
                            if isinstance(partial, dict):
                                for k, v in partial.items():
                                    if isinstance(v, list) and isinstance(accumulated.get(k), list):
                                        accumulated[k] = accumulated[k] + v
                                    else:
                                        accumulated[k] = v
                            status.update(label=f"Working — {label} done")
                    status.update(label="Done", state="complete", expanded=False)

                response = accumulated.get("response", "(no response produced)")
                st.markdown(response)
                st.session_state["chat_history"].append({"role": "assistant", "content": response})
                st.session_state["last_state"] = accumulated
                # Invalidate caches so dashboard updates
                get_patients.clear()
                get_recent_bookings.clear()
            except Exception as exc:
                st.error(f"Workflow error: {exc}")
                logging.exception("Workflow invocation failed")


with col_side:
    st.title("State & logs")

    last_state = st.session_state.get("last_state")
    if last_state is None:
        # Empty-state preview — show what the panel will look like once a query runs.
        with st.expander("👁 Preview — example state (replaced after first query)", expanded=True):
            st.caption("Sample data. Run a query to see real state, intents, and tool logs.")
            st.markdown("**Intents (example)**")
            st.code("['booking', 'medical_search']", language="python")
            st.markdown("**Appointment (example)**")
            st.code(
                json.dumps({
                    "doctor_name": "Dr. Arjun Kapoor",
                    "specialty": "nephrology",
                    "start_time": "Tue, May 5 · 9:00 AM",
                    "confirmation_no": "AGS-XXXXXX",
                }, indent=2),
                language="json",
            )
            st.markdown("**Tool log (example)**")
            st.code(
                "[\n"
                "  {\"node\": \"classify_intent\", \"parsed_intents\": [\"booking\", \"medical_search\"]},\n"
                "  {\"node\": \"booking\", \"result\": \"success\", \"confirmation_no\": \"AGS-XXXXXX\"},\n"
                "  {\"node\": \"medical_search\", \"backend\": \"duckduckgo\", \"results_count\": 4}\n"
                "]",
                language="json",
            )
    else:
        st.subheader("Intents")
        st.write(last_state.get("intents", []) or [last_state.get("intent", "?")])

        appt = last_state.get("appointment")
        if appt:
            st.subheader("Appointment")
            st.json(appt)

        rec = last_state.get("record_change")
        if rec:
            st.subheader("Record change")
            st.json(rec)

        hist = last_state.get("history_summary")
        if hist:
            st.subheader("History summary")
            st.markdown(hist)

        info = last_state.get("medical_info") or []
        if info:
            with st.expander(f"Medical search results ({len(info)} entries)", expanded=False):
                for item in info:
                    if "synthesis" in item:
                        st.markdown("**Synthesis**")
                        st.markdown(item["synthesis"])
                    else:
                        st.markdown(f"**{item.get('title')}**  \n{item.get('snippet', '')[:200]}…  \n[link]({item.get('url')})")

        sources = last_state.get("sources") or []
        if sources:
            with st.expander(f"Sources ({len(sources)})", expanded=False):
                for s in sources:
                    st.markdown(f"[{s['index']}] [{s.get('title','?')}]({s.get('url','')})  \n*{s.get('source','?')}*")

        with st.expander("Tool log", expanded=False):
            for entry in last_state.get("tool_log", []):
                st.code(json.dumps(entry, default=str, indent=2), language="json")

        if last_state.get("error"):
            st.subheader("⚠️ Errors")
            st.warning(last_state["error"])

    st.divider()
    st.subheader("Recent bookings")
    recent = get_recent_bookings()
    if recent:
        for b in recent:
            specialty = (b.get("specialty") or "").replace("_", " ")
            st.markdown(
                f"- **{b['doctor_name']}** ({specialty}) — "
                f"{format_appointment_time(b.get('start_time'))} — "
                f"`{b.get('confirmation_no','?')}`"
            )
    else:
        st.caption("No bookings yet.")

    st.divider()
    with st.expander("System info", expanded=False):
        st.json({
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "enable_persistence": settings.enable_persistence,
            "enable_parallel_fanout": settings.enable_parallel_fanout,
            "ehr_db_path": settings.ehr_db_path,
            "appointments_db_path": settings.appointments_db_path,
            "faiss_index_path": settings.faiss_index_path,
        })
