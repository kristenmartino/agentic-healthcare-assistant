"""Audit Log — PHI access viewer.

Surfaces every patient-identifiable read/write recorded by `tools.audit`.
HIPAA 45 CFR 164.312(b) requires the ability to examine activity in systems
containing ePHI; this page is the human-facing view of that audit table.

Filters: patient_id, action prefix, actor, free-text in details. CSV export.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta

import streamlit as st

from config import load_settings
from tools.audit import audit_summary, query_audit
from tools.ehr import list_patients


st.set_page_config(
    page_title="Audit Log — Healthcare Assistant",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #eef2ff 0%, #e0e7ff 100%);
        border: 1px solid #c7d2fe;
        border-left: 4px solid #4338ca;
        padding: 8px 14px;
        border-radius: 6px;
        margin-bottom: 12px;
        font-size: 14px;
        color: #1e1b4b;
    ">
        🔍 <strong>PHI access audit log.</strong>
        Every record read, write, and appointment booking is logged here.
        HIPAA Security Rule (45 CFR 164.312(b)).
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _settings():
    return load_settings()


settings = _settings()
st.title("Audit Log")
st.caption(f"Source: `{settings.audit_db_path}`")


# ---------- Summary strip ----------

summary = audit_summary(settings.audit_db_path)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total events", summary["total"])
col2.metric("Distinct actions", len(summary["by_action"]))
col3.metric("Distinct actors", len(summary["by_actor"]))
top_action = next(iter(summary["by_action"]), "—") if summary["by_action"] else "—"
col4.metric("Top action", top_action)

with st.expander("Counts by action", expanded=False):
    if summary["by_action"]:
        st.dataframe(
            [{"action": a, "count": n} for a, n in summary["by_action"].items()],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No events recorded yet — try a query in the patient chat first.")

with st.expander("Counts by actor", expanded=False):
    if summary["by_actor"]:
        st.dataframe(
            [{"actor": a, "count": n} for a, n in summary["by_actor"].items()],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No events recorded yet.")


# ---------- Filters ----------

st.subheader("Filters")
fcol1, fcol2, fcol3, fcol4 = st.columns([2, 2, 2, 1])

with fcol1:
    patients = list_patients(settings, actor="audit_view")
    patient_choices = ["(all)"] + [
        f"{p['name']} — {p['patient_id']}" for p in patients
    ]
    patient_pick = st.selectbox("Patient", patient_choices, key="audit_patient")
    patient_filter = None
    if patient_pick != "(all)":
        patient_filter = patient_pick.split(" — ", 1)[1]

with fcol2:
    action_choices = ["(all)"] + sorted(summary["by_action"].keys())
    action_pick = st.selectbox("Action", action_choices, key="audit_action")
    action_filter = None if action_pick == "(all)" else action_pick

with fcol3:
    actor_choices = ["(all)"] + sorted(summary["by_actor"].keys())
    actor_pick = st.selectbox("Actor", actor_choices, key="audit_actor")
    actor_filter = None if actor_pick == "(all)" else actor_pick

with fcol4:
    days_back = st.number_input("Days back", value=7, min_value=1, max_value=365, step=1)
    since_iso = (datetime.utcnow() - timedelta(days=int(days_back))).isoformat() + "Z"


# ---------- Events table ----------

events = query_audit(
    patient_id=patient_filter,
    action_prefix=action_filter,
    actor=actor_filter,
    since=since_iso,
    limit=500,
    db_path=settings.audit_db_path,
)

st.subheader(f"Events ({len(events)})")

if not events:
    st.info("No events match these filters. Run a query in the patient chat or the Doctor View to generate audit entries.")
else:
    # Flat table for the eye-scan
    rows = [
        {
            "ts": e["ts"],
            "actor": e["actor"],
            "action": e["action"],
            "resource": f"{e.get('resource_type') or ''}/{e.get('resource_id') or ''}".strip("/"),
            "patient_id": e.get("patient_id") or "",
            "outcome": e["outcome"],
        }
        for e in events
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Expandable details
    with st.expander("Event details (JSON)", expanded=False):
        for e in events[:50]:
            st.code(json.dumps(e, default=str, indent=2), language="json")

    # CSV download
    buf = io.StringIO()
    fieldnames = ["ts", "actor", "action", "resource_type", "resource_id",
                  "patient_id", "outcome", "details"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for e in events:
        row = {k: e.get(k, "") for k in fieldnames}
        if isinstance(row["details"], dict):
            row["details"] = json.dumps(row["details"], default=str)
        writer.writerow(row)

    st.download_button(
        "Download CSV",
        data=buf.getvalue(),
        file_name=f"audit_log_{datetime.utcnow():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
    )
