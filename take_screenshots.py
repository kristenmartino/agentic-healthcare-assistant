"""Take submission screenshots via playwright.

Captures 7 PNGs to screenshots/:
1. 01_patient_view_empty.png       — initial chat page
2. 02_multi_intent_response.png    — canonical CKD multi-intent query result
3. 03_state_trace_expanded.png     — same query, tool log expanded
4. 04_history_query.png            — Anjali Mehra history query
5. 05_doctor_view_dashboard.png    — Doctor View page (KPIs, today's schedule)
6. 06_doctor_per_doctor.png        — Doctor View per-doctor schedule + utilization chart
7. 07_doctor_patient_roster.png    — Doctor View patient roster + slot manager
"""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8501"
OUT = Path(__file__).parent / "screenshots"
OUT.mkdir(exist_ok=True)


def wait_for_idle(page, ms: int = 1500):
    """Streamlit re-runs on input. Wait for the spinner to disappear."""
    page.wait_for_load_state("networkidle", timeout=15000)
    time.sleep(ms / 1000)


def shoot(page, name: str, full_page: bool = True):
    path = OUT / name
    page.screenshot(path=str(path), full_page=full_page)
    print(f"  ✓ {name} ({path.stat().st_size // 1024} KB)")
    return path


def click_chat_input_and_send(page, text: str):
    """Send a message via the chat input."""
    chat = page.locator('[data-testid="stChatInput"] textarea').first
    chat.fill(text)
    chat.press("Enter")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()

        # ----- 1. Patient view, empty -----
        print("[1/7] Patient view (empty)")
        page.goto(BASE, wait_until="networkidle", timeout=20000)
        wait_for_idle(page)
        shoot(page, "01_patient_view_empty.png")

        # ----- 2. Multi-intent canonical -----
        print("[2/7] Multi-intent canonical query")
        click_chat_input_and_send(
            page,
            "My 70-year-old father has chronic kidney disease. "
            "Book a nephrologist for him and summarize the latest treatment methods.",
        )
        # Wait until the "Thinking..." placeholder is replaced by real text
        # (the assistant chat-message bubble loses the hourglass).
        try:
            page.wait_for_function(
                "() => !document.body.innerText.includes('⏳ Thinking') && document.body.innerText.includes('Confirmation')",
                timeout=45000,
            )
        except Exception:
            page.wait_for_timeout(30000)
        wait_for_idle(page, ms=2000)
        shoot(page, "02_multi_intent_response.png")

        # ----- 3. Tool log expanded -----
        print("[3/7] Tool log expanded")
        # Streamlit expanders are <details> elements with summary "Tool log"
        try:
            tool_log = page.locator("summary:has-text('Tool log')").first
            tool_log.click(timeout=5000)
            page.wait_for_timeout(800)
        except Exception as e:
            print(f"  (could not expand tool log: {e})")
        shoot(page, "03_state_trace_expanded.png")

        # ----- 4. History query (new conversation) -----
        print("[4/7] History query — Anjali Mehra")
        # Click "Start new conversation" button to clear chat
        try:
            new_conv = page.locator("button:has-text('Start new conversation')").first
            new_conv.click(timeout=5000)
            page.wait_for_timeout(2000)
            wait_for_idle(page)
        except Exception:
            page.goto(BASE, wait_until="networkidle")
            wait_for_idle(page)

        click_chat_input_and_send(page, "Show me Anjali Mehra's medical history")
        try:
            page.wait_for_function(
                "() => !document.body.innerText.includes('⏳ Thinking') && document.body.innerText.includes('Anjali')",
                timeout=45000,
            )
        except Exception:
            page.wait_for_timeout(20000)
        wait_for_idle(page, ms=2000)
        shoot(page, "04_history_query.png")

        # ----- 5. Doctor view: top of page -----
        print("[5/7] Doctor View — KPIs + today's schedule")
        # Streamlit multi-page nav: pages live at /Doctor%20View
        # Streamlit auto-generates URLs from the filename: 2_Doctor_View.py → "Doctor View"
        page.goto(f"{BASE}/Doctor_View", wait_until="networkidle", timeout=20000)
        wait_for_idle(page, ms=2000)
        shoot(page, "05_doctor_view_dashboard.png", full_page=False)

        # ----- 6. Doctor view: per-doctor schedule -----
        print("[6/7] Doctor View — per-doctor schedule")
        # Scroll to the per-doctor section
        page.evaluate("window.scrollTo(0, 800)")
        page.wait_for_timeout(800)
        shoot(page, "06_doctor_per_doctor.png", full_page=False)

        # ----- 7. Doctor view: patient roster + slot manager -----
        print("[7/7] Doctor View — patient roster")
        page.evaluate("window.scrollTo(0, 1800)")
        page.wait_for_timeout(800)
        shoot(page, "07_doctor_patient_roster.png", full_page=False)

        browser.close()

    print(f"\nAll screenshots written to: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
