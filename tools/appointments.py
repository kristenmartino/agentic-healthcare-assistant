"""Mock Doctor Schedule API.

Maintains a SQLite-backed catalog of doctors and an appointment slot table.
Slots are pre-generated for the next 14 days × 9am-5pm × 30 min increments.

Booking is deterministic: given a (patient_id, specialty, target_date), pick
the earliest available matching slot. This makes the system testable without
external state.
"""
from __future__ import annotations

import logging
import random
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


# Synthetic doctor roster — populated once at init time.
_DOCTORS: list[tuple[str, str, str]] = [
    ("DOC001", "Dr. Anjali Sharma", "general_practice"),
    ("DOC002", "Dr. Raj Mehta", "general_practice"),
    ("DOC003", "Dr. Priya Nair", "cardiology"),
    ("DOC004", "Dr. Vikram Singh", "cardiology"),
    ("DOC005", "Dr. Sunita Rao", "endocrinology"),
    ("DOC006", "Dr. Arjun Kapoor", "nephrology"),
    ("DOC007", "Dr. Meera Iyer", "nephrology"),
    ("DOC008", "Dr. Ramesh Pillai", "neurology"),
    ("DOC009", "Dr. Sarah Cohen", "pulmonology"),
    ("DOC010", "Dr. David Park", "oncology"),
    ("DOC011", "Dr. Lisa Wong", "psychiatry"),
    ("DOC012", "Dr. Faisal Khan", "dermatology"),
]


SCHEMA = """
CREATE TABLE IF NOT EXISTS doctors (
    doctor_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    specialty TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slots (
    slot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    booked INTEGER NOT NULL DEFAULT 0,
    booked_by_patient_id TEXT,
    confirmation_no TEXT,
    booked_at TIMESTAMP,
    FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
);

CREATE INDEX IF NOT EXISTS idx_slots_doctor_time ON slots(doctor_id, start_time);
CREATE INDEX IF NOT EXISTS idx_slots_specialty_open
    ON slots(doctor_id, booked, start_time);
"""


@contextmanager
def _connect(db_path: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize_appointments(db_path: str, days_ahead: int = 14) -> dict[str, int]:
    """Create the schema, populate doctors, and pre-generate slots."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)

        # Wipe + reload (idempotent)
        conn.execute("DELETE FROM slots")
        conn.execute("DELETE FROM doctors")

        conn.executemany(
            "INSERT INTO doctors (doctor_id, name, specialty) VALUES (?, ?, ?)",
            _DOCTORS,
        )

        # Generate slots: today + days_ahead, 9am-5pm, 30-min increments, weekdays only
        slots_to_insert: list[tuple[str, str, str]] = []
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        for doctor_id, _, _ in _DOCTORS:
            for d in range(days_ahead):
                day = today + timedelta(days=d)
                if day.weekday() >= 5:
                    continue  # skip weekends
                for hour in range(9, 17):
                    for minute in (0, 30):
                        start = day.replace(hour=hour, minute=minute)
                        end = start + timedelta(minutes=30)
                        slots_to_insert.append((doctor_id, start.isoformat(), end.isoformat()))

        conn.executemany(
            "INSERT INTO slots (doctor_id, start_time, end_time) VALUES (?, ?, ?)",
            slots_to_insert,
        )

    logger.info(
        "Appointments seeded: %d doctors, %d slots over next %d days",
        len(_DOCTORS), len(slots_to_insert), days_ahead,
    )
    return {"doctors": len(_DOCTORS), "slots": len(slots_to_insert)}


def ensure_future_slots(db_path: str, *, days_ahead: int = 14, min_open_slots: int = 50) -> dict[str, int]:
    """Idempotent slot freshness check — call at every startup path.

    The problem this solves: `initialize_appointments` pre-generates slots
    for today + N days at seed time. If that seed runs weeks before the next
    `book_appointment` call, every slot is already in the past, and bookings
    fail with "no available slots" — silently breaking the eval and any
    grader's first interaction.

    This function appends slots forward without touching existing rows.
    Safe to call from seed.py, the Streamlit app's startup, and eval setup
    without disturbing already-booked appointments.

    Args:
        days_ahead: how far forward to extend the window.
        min_open_slots: if fewer open future slots exist than this, top up.

    Returns: {"added": int, "open_future_before": int, "open_future_after": int}.
    """
    # If the doctor / slot tables don't exist, defer to initialize_appointments.
    if not Path(db_path).exists():
        initialize_appointments(db_path, days_ahead=days_ahead)
        return {"added": 0, "bootstrapped": True}

    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)

        # Make sure the doctors table is populated. If a downstream user
        # somehow ended up with an empty doctors table, we can't generate
        # slots — fall back to a full re-init.
        doctor_rows = conn.execute("SELECT doctor_id FROM doctors").fetchall()
        if not doctor_rows:
            conn.executemany(
                "INSERT INTO doctors (doctor_id, name, specialty) VALUES (?, ?, ?)",
                _DOCTORS,
            )
            doctor_rows = conn.execute("SELECT doctor_id FROM doctors").fetchall()

        open_before = conn.execute(
            "SELECT COUNT(*) FROM slots WHERE booked = 0 AND start_time >= ?",
            (datetime.now().isoformat(),),
        ).fetchone()[0]

        if open_before >= min_open_slots:
            return {"added": 0, "open_future_before": open_before,
                    "open_future_after": open_before}

        # Generate forward starting from today, skipping (doctor, start_time)
        # pairs that already exist so we never duplicate.
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        candidate: list[tuple[str, str, str]] = []
        for row in doctor_rows:
            for d in range(days_ahead):
                day = today + timedelta(days=d)
                if day.weekday() >= 5:
                    continue
                for hour in range(9, 17):
                    for minute in (0, 30):
                        start = day.replace(hour=hour, minute=minute)
                        end = start + timedelta(minutes=30)
                        candidate.append((row["doctor_id"], start.isoformat(), end.isoformat()))

        # Bulk dedup against the existing table.
        added = 0
        for doctor_id, start, end in candidate:
            existing = conn.execute(
                "SELECT 1 FROM slots WHERE doctor_id = ? AND start_time = ?",
                (doctor_id, start),
            ).fetchone()
            if existing:
                continue
            conn.execute(
                "INSERT INTO slots (doctor_id, start_time, end_time) VALUES (?, ?, ?)",
                (doctor_id, start, end),
            )
            added += 1

        open_after = conn.execute(
            "SELECT COUNT(*) FROM slots WHERE booked = 0 AND start_time >= ?",
            (datetime.now().isoformat(),),
        ).fetchone()[0]

    logger.info(
        "ensure_future_slots: open_before=%d, open_after=%d, added=%d (window=%dd)",
        open_before, open_after, added, days_ahead,
    )
    return {"added": added, "open_future_before": open_before,
            "open_future_after": open_after}


def list_doctors_for_specialty(db_path: str, specialty: str) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT doctor_id, name, specialty FROM doctors WHERE specialty = ?",
            (specialty,),
        ).fetchall()
    return [dict(r) for r in rows]


def book_appointment(
    db_path: str,
    *,
    patient_id: str,
    patient_name: str,
    specialty: str,
    preferred_date: str | None = None,
) -> dict:
    """Book the earliest available slot matching specialty (and date if given).

    Returns: {doctor_id, doctor_name, start_time, end_time, slot_id, confirmation_no, status}.
    Raises ValueError if no slot is available.
    """
    with _connect(db_path) as conn:
        query = """
            SELECT s.slot_id, s.doctor_id, d.name AS doctor_name, s.start_time, s.end_time
              FROM slots s
              JOIN doctors d ON d.doctor_id = s.doctor_id
             WHERE s.booked = 0
               AND d.specialty = ?
        """
        params: list = [specialty]

        if preferred_date:
            query += " AND date(s.start_time) >= date(?)"
            params.append(preferred_date)

        query += " ORDER BY s.start_time LIMIT 1"

        row = conn.execute(query, params).fetchone()

        if row is None:
            raise ValueError(
                f"No available slots for specialty '{specialty}'"
                + (f" on or after {preferred_date}" if preferred_date else "")
            )

        confirmation = f"AGS-{random.randint(100000, 999999)}"
        conn.execute(
            """
            UPDATE slots
               SET booked = 1,
                   booked_by_patient_id = ?,
                   confirmation_no = ?,
                   booked_at = CURRENT_TIMESTAMP
             WHERE slot_id = ?
            """,
            (patient_id, confirmation, row["slot_id"]),
        )

    return {
        "doctor_id": row["doctor_id"],
        "doctor_name": row["doctor_name"],
        "specialty": specialty,
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "slot_id": row["slot_id"],
        "confirmation_no": confirmation,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "status": "confirmed",
    }


def list_recent_bookings(db_path: str, limit: int = 20) -> list[dict]:
    """For the Streamlit dashboard."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.slot_id, s.start_time, s.confirmation_no, s.booked_by_patient_id,
                   d.name AS doctor_name, d.specialty, s.booked_at
              FROM slots s
              JOIN doctors d ON d.doctor_id = s.doctor_id
             WHERE s.booked = 1
             ORDER BY s.booked_at DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_all_bookings(db_path: str, *, upcoming_only: bool = False) -> list[dict]:
    """All bookings, optionally filtered to future appointments only."""
    where = "WHERE s.booked = 1"
    if upcoming_only:
        where += " AND s.start_time >= datetime('now')"
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT s.slot_id, s.start_time, s.end_time, s.confirmation_no,
                   s.booked_by_patient_id, s.booked_at,
                   d.doctor_id, d.name AS doctor_name, d.specialty
              FROM slots s
              JOIN doctors d ON d.doctor_id = s.doctor_id
             {where}
             ORDER BY s.start_time ASC
            """,
        ).fetchall()
    return [dict(r) for r in rows]


def get_doctor_schedule(db_path: str, doctor_id: str, *, days_ahead: int = 7) -> list[dict]:
    """Return all slots (booked + open) for a specific doctor."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.slot_id, s.start_time, s.end_time, s.booked,
                   s.booked_by_patient_id, s.confirmation_no
              FROM slots s
             WHERE s.doctor_id = ?
               AND date(s.start_time) <= date('now', ? || ' days')
             ORDER BY s.start_time ASC
            """,
            (doctor_id, f"+{days_ahead}"),
        ).fetchall()
    return [dict(r) for r in rows]


def cancel_booking(db_path: str, slot_id: int) -> dict:
    """Free a previously-booked slot. Idempotent."""
    with _connect(db_path) as conn:
        before = conn.execute(
            "SELECT booked, confirmation_no, booked_by_patient_id FROM slots WHERE slot_id = ?",
            (slot_id,),
        ).fetchone()
        if not before:
            return {"status": "not_found", "slot_id": slot_id}
        if not before["booked"]:
            return {"status": "already_open", "slot_id": slot_id}
        conn.execute(
            """
            UPDATE slots
               SET booked = 0,
                   booked_by_patient_id = NULL,
                   confirmation_no = NULL,
                   booked_at = NULL
             WHERE slot_id = ?
            """,
            (slot_id,),
        )
    return {
        "status": "cancelled",
        "slot_id": slot_id,
        "freed_confirmation": before["confirmation_no"],
    }


def get_specialty_stats(db_path: str) -> list[dict]:
    """Per-specialty booking counts + utilization rate."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT d.specialty,
                   COUNT(s.slot_id) AS total_slots,
                   SUM(CASE WHEN s.booked = 1 THEN 1 ELSE 0 END) AS booked_slots
              FROM doctors d
              JOIN slots s ON s.doctor_id = d.doctor_id
             GROUP BY d.specialty
             ORDER BY booked_slots DESC, d.specialty
            """,
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        total = d["total_slots"] or 0
        booked = d["booked_slots"] or 0
        d["utilization_pct"] = round(100 * booked / total, 1) if total else 0.0
        out.append(d)
    return out


def get_doctors_for_dashboard(db_path: str) -> list[dict]:
    """All doctors with their booking counts. For doctor-selector dropdown."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT d.doctor_id, d.name, d.specialty,
                   SUM(CASE WHEN s.booked = 1 THEN 1 ELSE 0 END) AS booked_count
              FROM doctors d
              LEFT JOIN slots s ON s.doctor_id = d.doctor_id
             GROUP BY d.doctor_id, d.name, d.specialty
             ORDER BY d.specialty, d.name
            """,
        ).fetchall()
    return [dict(r) for r in rows]


# CLI: python -m tools.appointments init <db_path>
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) >= 3 and sys.argv[1] == "init":
        result = initialize_appointments(sys.argv[2])
        print(result)
    else:
        print("Usage: python -m tools.appointments init <db_path>")
        sys.exit(1)
