"""Doctor-facing dashboard.

This page is auto-discovered by Streamlit (via the `pages/` directory
convention) and appears in the sidebar nav. It's a read-mostly admin view
intended for clinicians and clinic staff — not for patients.

Sections:
1. KPI strip: total bookings, today's bookings, slots utilization, top specialty
2. Today's schedule: chronological list of today's bookings
3. Per-doctor schedule: pick a doctor, see their next 7 days
4. Specialty stats: utilization % per specialty
5. Patient roster: full EHR list with summaries
6. Slot manager: cancel a booking by slot_id (would be expanded with auth in production)
"""
from __future__ import annotations

# Allow imports from project root
import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_settings
from tools.appointments import (
    cancel_booking,
    get_doctor_schedule,
    get_doctors_for_dashboard,
    get_specialty_stats,
    list_all_bookings,
)
from tools.ehr import list_patients
from utils import format_appointment_time

st.set_page_config(
    page_title="Doctor View — Healthcare Assistant",
    page_icon="🩺",
    layout="wide",
)


# Persistent disclaimer banner (matches app.py)
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


@st.cache_resource(show_spinner=False)
def _settings():
    return load_settings()


@st.cache_data(ttl=10, show_spinner=False)
def _all_bookings(upcoming_only=False):
    return list_all_bookings(_settings().appointments_db_path, upcoming_only=upcoming_only)


@st.cache_data(ttl=10, show_spinner=False)
def _stats():
    return get_specialty_stats(_settings().appointments_db_path)


@st.cache_data(ttl=10, show_spinner=False)
def _doctors():
    return get_doctors_for_dashboard(_settings().appointments_db_path)


@st.cache_data(ttl=10, show_spinner=False)
def _patients():
    return list_patients(_settings(), actor="doctor_view")


def _refresh_all_caches():
    _all_bookings.clear()
    _stats.clear()
    _doctors.clear()
    _patients.clear()


# ----- HEADER -----

st.title("🩺 Doctor View")
st.caption("Admin dashboard for clinicians and clinic staff. All data is synthetic.")

if st.button("🔄 Refresh data", help="Re-read SQLite — useful after the patient agent makes a booking"):
    _refresh_all_caches()
    st.rerun()


# ----- KPIS -----

bookings_all = _all_bookings(upcoming_only=False)
bookings_upcoming = _all_bookings(upcoming_only=True)
stats = _stats()
doctors = _doctors()
patients = _patients()

today_str = datetime.now().date().isoformat()
today_bookings = [b for b in bookings_all if (b.get("start_time") or "").startswith(today_str)]

total_slots = sum(s["total_slots"] for s in stats)
booked_slots = sum(s["booked_slots"] for s in stats)
utilization = (100 * booked_slots / total_slots) if total_slots else 0
top_specialty = max(stats, key=lambda s: s["booked_slots"])["specialty"] if stats else "—"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total bookings", len(bookings_all))
c2.metric("Today", len(today_bookings))
c3.metric("Slot utilization", f"{utilization:.1f}%", help=f"{booked_slots} booked / {total_slots} total slots")
c4.metric("Top specialty", top_specialty.replace("_", " ").title())


# ----- TODAY'S SCHEDULE -----

st.divider()
st.subheader("📅 Today's schedule")
if today_bookings:
    rows = []
    for b in today_bookings:
        rows.append({
            "Time": format_appointment_time(b.get("start_time")).split(" · ")[-1],
            "Doctor": b.get("doctor_name", "?"),
            "Specialty": (b.get("specialty") or "").replace("_", " ").title(),
            "Patient ID": b.get("booked_by_patient_id", "?"),
            "Confirmation": b.get("confirmation_no", "?"),
            "Slot ID": b.get("slot_id", "?"),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.info("No bookings for today.")


# ----- UPCOMING (next 7 days) -----

st.divider()
st.subheader("📆 Upcoming bookings (next 7 days)")
cutoff = datetime.now() + timedelta(days=7)
upcoming = [
    b for b in bookings_upcoming
    if datetime.fromisoformat((b.get("start_time") or "").replace("Z", "")) <= cutoff
]
if upcoming:
    rows = []
    for b in upcoming[:50]:
        pretty = format_appointment_time(b.get("start_time"))
        date_part, _, time_part = pretty.partition(" · ")
        rows.append({
            "Date": date_part,
            "Time": time_part,
            "Doctor": b.get("doctor_name", "?"),
            "Specialty": (b.get("specialty") or "").replace("_", " ").title(),
            "Patient ID": b.get("booked_by_patient_id", "?"),
            "Confirmation": b.get("confirmation_no", "?"),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True, height=300)
else:
    st.info("No upcoming bookings in the next 7 days.")


# ----- PER-DOCTOR SCHEDULE + SPECIALTY STATS -----

st.divider()
left, right = st.columns([2, 1])

with left:
    st.subheader("👨‍⚕️ Per-doctor schedule (next 7 days)")
    doctor_options = [
        f"{d['name']} — {d['specialty'].replace('_',' ').title()} ({d['booked_count']} booked)"
        for d in doctors
    ]
    if doctor_options:
        selected = st.selectbox("Select a doctor", range(len(doctor_options)), format_func=lambda i: doctor_options[i])
        doc = doctors[selected]
        sched = get_doctor_schedule(_settings().appointments_db_path, doc["doctor_id"], days_ahead=7)

        booked_count = sum(1 for s in sched if s["booked"])
        st.caption(f"{booked_count} booked / {len(sched)} total slots in next 7 days")

        rows = []
        for s in sched[:30]:
            pretty = format_appointment_time(s.get("start_time"))
            date_part, _, time_part = pretty.partition(" · ")
            rows.append({
                "Date": date_part,
                "Time": time_part,
                "Status": "🟢 booked" if s["booked"] else "⚪ open",
                "Patient ID": s.get("booked_by_patient_id") or "—",
                "Confirmation": s.get("confirmation_no") or "—",
                "Slot ID": s["slot_id"],
            })
        st.dataframe(rows, use_container_width=True, hide_index=True, height=400)

with right:
    st.subheader("📊 Specialty utilization")
    if stats:
        chart_data = {
            "Specialty": [s["specialty"].replace("_", " ").title() for s in stats],
            "Utilization %": [s["utilization_pct"] for s in stats],
        }
        st.bar_chart(chart_data, x="Specialty", y="Utilization %", height=300)
        with st.expander("Raw stats", expanded=False):
            st.dataframe(stats, use_container_width=True, hide_index=True)


# ----- PATIENT ROSTER -----

st.divider()
st.subheader(f"👥 Patient roster ({len(patients)} patients)")
if patients:
    rows = []
    for p in patients:
        rows.append({
            "Name": p.get("name", "?"),
            "Age": p.get("age") or "—",
            "Gender": p.get("gender") or "—",
            "Patient ID": p.get("patient_id", "?"),
            "Summary": (p.get("summary") or "—")[:100],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ----- SLOT MANAGER -----

st.divider()
with st.expander("🛠️ Slot manager (cancel a booking)", expanded=False):
    st.caption(
        "In production this would require auth + a confirmation step. Here it's a debug control "
        "for clinicians to free up a slot if a patient cancels."
    )
    slot_id_input = st.number_input(
        "Slot ID to cancel",
        min_value=1,
        step=1,
        key="cancel_slot_id",
    )
    if st.button("Cancel this slot", type="primary"):
        result = cancel_booking(_settings().appointments_db_path, int(slot_id_input))
        st.json(result)
        if result["status"] == "cancelled":
            st.success(f"Slot {slot_id_input} freed (was confirmation {result['freed_confirmation']}).")
            _refresh_all_caches()
        elif result["status"] == "already_open":
            st.warning(f"Slot {slot_id_input} was already open.")
        else:
            st.error(f"Slot {slot_id_input} not found.")


# ----- FOOTER / SYSTEM INFO -----

st.divider()
with st.expander("ℹ️ System info", expanded=False):
    s = _settings()
    st.json({
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model,
        "ehr_db_path": s.ehr_db_path,
        "appointments_db_path": s.appointments_db_path,
        "total_doctors": len(doctors),
        "total_slots": total_slots,
        "booked_slots": booked_slots,
    })
