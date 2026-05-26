"""Tests for tools/appointments.ensure_future_slots — the slot freshness top-up.

The function exists to handle the "seeded weeks ago, every slot is in the
past, every booking fails" case (which actually shows up in our eval logs
when an existing appointments.sqlite is reused across runs). Behavior must:

- Bootstrap from scratch if the DB doesn't exist yet.
- Skip work when plenty of open future slots already exist.
- Top up with new slots when the existing ones are all in the past, without
  wiping or duplicating anything.
- Preserve existing booked rows (a real production constraint).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tools.appointments import (
    book_appointment,
    ensure_future_slots,
    initialize_appointments,
)


@pytest.fixture
def appts_db(tmp_path: Path) -> str:
    return str(tmp_path / "appointments.sqlite")


def test_bootstraps_when_db_missing(appts_db):
    """First call against a missing DB should just initialize_appointments."""
    assert not Path(appts_db).exists()
    result = ensure_future_slots(appts_db)
    assert result.get("bootstrapped") is True
    # Now we should have a full doctor + slot population.
    with sqlite3.connect(appts_db) as conn:
        doctors = conn.execute("SELECT COUNT(*) FROM doctors").fetchone()[0]
        slots = conn.execute("SELECT COUNT(*) FROM slots").fetchone()[0]
    assert doctors > 0 and slots > 0


def test_skips_when_already_fresh(appts_db):
    initialize_appointments(appts_db)
    result = ensure_future_slots(appts_db)
    assert result["added"] == 0
    assert result["open_future_before"] == result["open_future_after"]


def test_tops_up_when_all_slots_in_the_past(appts_db):
    """The eval-day-after-seed scenario: rewrite every slot to be a year ago
    and confirm ensure_future_slots restores forward coverage."""
    initialize_appointments(appts_db)
    one_year_ago = datetime.now() - timedelta(days=365)
    with sqlite3.connect(appts_db) as conn:
        # Move every slot a full year into the past.
        conn.execute(
            "UPDATE slots SET start_time = ?, end_time = ?",
            (one_year_ago.isoformat(), (one_year_ago + timedelta(minutes=30)).isoformat()),
        )
        conn.commit()
        before = conn.execute(
            "SELECT COUNT(*) FROM slots WHERE start_time >= ?",
            (datetime.now().isoformat(),),
        ).fetchone()[0]
    assert before == 0

    result = ensure_future_slots(appts_db)
    assert result["added"] > 0
    # "added" counts inserts across the whole window (including today's
    # earlier-than-now slots, which don't show up in `open_future_after`).
    # So only assert we ended with at least some future capacity.
    assert result["open_future_after"] > 0

    # The newly added rows should actually be bookable.
    appt = book_appointment(
        appts_db,
        patient_id="p1",
        patient_name="Test Patient",
        specialty="nephrology",
    )
    assert appt["status"] == "confirmed"


def test_preserves_existing_bookings(appts_db):
    """Top-up must not delete or modify booked slots."""
    initialize_appointments(appts_db)
    appt = book_appointment(
        appts_db,
        patient_id="p2",
        patient_name="Booked Patient",
        specialty="cardiology",
    )
    conf = appt["confirmation_no"]

    ensure_future_slots(appts_db)  # call again

    with sqlite3.connect(appts_db) as conn:
        row = conn.execute(
            "SELECT booked, confirmation_no FROM slots WHERE slot_id = ?",
            (appt["slot_id"],),
        ).fetchone()
    assert row is not None
    assert row[0] == 1
    assert row[1] == conf


def test_no_duplicate_slots_on_repeated_calls(appts_db):
    initialize_appointments(appts_db)
    # Wipe all slots so the top-up will refill — but keep the doctors row.
    with sqlite3.connect(appts_db) as conn:
        conn.execute("DELETE FROM slots")
        conn.commit()

    r1 = ensure_future_slots(appts_db)
    r2 = ensure_future_slots(appts_db)
    assert r1["added"] > 0
    # Second call should add nothing new — every candidate already exists.
    assert r2["added"] == 0


def test_recovers_when_doctors_table_was_emptied(appts_db):
    """Some operational mishap leaves doctors empty; the helper should
    re-populate before generating slots."""
    initialize_appointments(appts_db)
    with sqlite3.connect(appts_db) as conn:
        conn.execute("DELETE FROM doctors")
        conn.execute("DELETE FROM slots")
        conn.commit()
    result = ensure_future_slots(appts_db)
    assert result["added"] > 0
    with sqlite3.connect(appts_db) as conn:
        n_doctors = conn.execute("SELECT COUNT(*) FROM doctors").fetchone()[0]
    assert n_doctors > 0
