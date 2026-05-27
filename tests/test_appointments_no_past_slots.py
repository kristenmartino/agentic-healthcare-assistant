"""Tests for the stale-slot lower-bound on booking + doctor schedule.

Direct response to the PR #6 review: both `book_appointment` and
`get_doctor_schedule` previously allowed past slots to surface because
the SQL queries only had an UPPER bound on start_time. That meant:

  - book_appointment with no preferred_date could return yesterday's 4pm
  - get_doctor_schedule could return slots from last week as "upcoming"

Both are real bugs that ensure_future_slots only masks rather than fixes;
they also leak data (a past booked slot's booked_by_patient_id / confirmation_no
ended up in agent-loop responses).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tools.appointments import (
    book_appointment,
    get_doctor_schedule,
    initialize_appointments,
)


@pytest.fixture
def appts_db(tmp_path: Path) -> str:
    db = str(tmp_path / "appts.sqlite")
    initialize_appointments(db)
    return db


def _push_some_slots_into_past(db_path: str, hours_back: int = 6) -> int:
    """Move the EARLIEST N slots into the past so booking would pick one.
    Returns the number rewritten."""
    yesterday = datetime.now() - timedelta(hours=hours_back)
    with sqlite3.connect(db_path) as c:
        rows = c.execute(
            "SELECT slot_id FROM slots ORDER BY start_time LIMIT 20"
        ).fetchall()
        for i, (sid,) in enumerate(rows):
            t = yesterday - timedelta(minutes=30 * i)
            c.execute(
                "UPDATE slots SET start_time=?, end_time=? WHERE slot_id=?",
                (t.isoformat(), (t + timedelta(minutes=30)).isoformat(), sid),
            )
        c.commit()
    return len(rows)


def test_booking_skips_past_slots(appts_db):
    rewritten = _push_some_slots_into_past(appts_db)
    assert rewritten > 0
    appt = book_appointment(appts_db, patient_id="p1",
                            patient_name="Test", specialty="general_practice")
    start = datetime.fromisoformat(appt["start_time"])
    assert start >= datetime.now() - timedelta(minutes=1), (
        f"book_appointment returned a past slot ({start.isoformat()}); "
        "the start_time >= datetime('now') filter was missing."
    )


def test_doctor_schedule_skips_past_slots(appts_db):
    """Even when stale slots exist for the doctor, get_doctor_schedule
    must not return them. This prevents the agent_loop from quoting last
    week's bookings as 'upcoming' and leaking the prior booker's identifiers."""
    _push_some_slots_into_past(appts_db)
    with sqlite3.connect(appts_db) as c:
        doctor_id = c.execute("SELECT doctor_id FROM doctors LIMIT 1").fetchone()[0]
    schedule = get_doctor_schedule(appts_db, doctor_id, days_ahead=7)
    now = datetime.now()
    for slot in schedule:
        assert datetime.fromisoformat(slot["start_time"]) >= now - timedelta(minutes=1), (
            f"get_doctor_schedule returned past slot {slot['start_time']}"
        )


def test_booking_still_works_with_explicit_future_date(appts_db):
    """The preferred_date path should still work (and also exclude past
    slots automatically)."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    appt = book_appointment(appts_db, patient_id="p2", patient_name="X",
                            specialty="cardiology", preferred_date=tomorrow)
    assert datetime.fromisoformat(appt["start_time"]) >= datetime.now()
