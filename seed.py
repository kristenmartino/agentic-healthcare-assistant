"""One-shot seed script: initialize EHR + appointments + FAISS index.

Run once before the first launch:
    python seed.py

Idempotent — running again rebuilds everything from source.
"""
from __future__ import annotations

import logging
from pathlib import Path

from config import load_settings
from tools.appointments import initialize_appointments
from tools.ehr_db import initialize_ehr
from tools.vector_index import build_index


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = load_settings()

    print("=" * 60)
    print(" Healthcare Assistant — Seed Script")
    print("=" * 60)

    # 1. EHR
    print("\n[1/3] Loading records.xlsx → EHR SQLite ...")
    ehr_result = initialize_ehr(settings.records_xlsx_path, settings.ehr_db_path)
    print(f"      {ehr_result}")

    # 2. Appointments
    print("\n[2/3] Pre-generating appointment slots ...")
    appt_result = initialize_appointments(settings.appointments_db_path)
    print(f"      {appt_result}")

    # 3. FAISS index over patient PDFs
    print("\n[3/3] Building FAISS index over patient PDFs ...")
    pdf_dir = Path(settings.patient_pdf_dir)
    pdf_paths = sorted(str(p) for p in pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"      WARNING: No PDFs found in {pdf_dir} — index will be empty.")
        return
    index_result = build_index(pdf_paths, settings.faiss_index_path, settings.faiss_chunks_path)
    print(f"      Indexed {len(pdf_paths)} PDFs: {index_result}")

    print("\n" + "=" * 60)
    print("Seed complete.")
    print("Next: python graph.py \"your query\"  OR  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
