# 2-minute UI walkthrough script

Recording target: 100-130 seconds. Show, in order, the 5 things a portfolio
reviewer most wants to see: the safety classifier firing, the multi-intent
fan-out producing real cited medical content, the streaming response, the
PHI audit log, and the trace dashboard.

## Setup before recording

1. `pip install -r requirements.txt`
2. `python seed.py` (one-time)
3. Set `.env` with `ANTHROPIC_API_KEY` (so streaming + good prose) and
   `TAVILY_API_KEY` (so real MedlinePlus citations).
4. `streamlit run app.py`
5. Open at 1280×800 or wider so the right-hand state panel is visible.
6. Have a second tab open for the **🔍 Audit Log** and **📊 Traces** pages.

## Beat sheet (with talking points)

**0:00-0:10 — Open with the safety classifier**

Sidebar shows `LLM: anthropic (claude-sonnet-4-6)`, `EHR backend: sqlite`,
`Search: tavily`. Type:

> *"Help, I have crushing chest pain right now and I can't breathe."*

> Say: "Before any LLM gets a chance to soften this, a deterministic safety
> classifier fires — sub-millisecond — and returns the hardcoded 911 / 988
> / 112 / 108 template. The model never sees emergency queries."

Show: `intents: ["emergency"]` and the 🚨 cardiac template in the chat.

**0:10-0:35 — Multi-intent fan-out with real citations**

Click "Start new conversation". Select patient "Anjali Mehra" in the
sidebar. Type:

> *"My 70-year-old father has chronic kidney disease. Book a nephrologist
> for him and summarize the latest treatment methods."*

> Say: "Two intents — booking AND medical search — fan out in parallel.
> The booking branch books the earliest nephrology slot. The medical-
> search branch sends a *stripped* query to Tavily — without 'book a
> nephrologist for him' — so it actually finds MedlinePlus content."

Watch the status panel tick through `safety`, `classify_intent`,
`booking_node`, `medical_search_node`, `compose_response`. The response
streams in token-by-token.

**0:35-1:00 — Show the cited sources panel**

Expand the right-side state panel. Point at:
- `intents: ["booking", "medical_search"]`
- `Appointment` JSON with doctor + confirmation #
- `Sources` panel — actual MedlinePlus / WHO / CDC URLs
- `Tool log` with `extraction_method: "llm"` on the medical_search node

> Say: "Every cited source has a real URL. The state panel is the audit
> trail of what the agent did this turn — what intents fired, what
> backend each branch hit, what tools were called."

**1:00-1:20 — Audit Log page**

Click the **🔍 Audit Log** page in the sidebar.

> Say: "HIPAA's Security Rule requires every PHI access to be logged in
> a way that survives the rest of the system being compromised. The
> audit log lives in its own SQLite DB. Every patient lookup, every
> booking, every record write produces one row tagged with actor,
> action, resource, and JSON details."

Filter to actor=`patient_chat` and show the 4-5 rows from the previous
query.

**1:20-1:45 — Traces page**

Click the **📊 Traces** page.

> Say: "Every workflow invocation is logged here, separate from PHI
> access — this is the engineering observability layer. P50 and P95
> latency, error rate, emergency count. Filterable by actor (chat /
> CLI / eval), search backend used, days back. CSV export for
> spreadsheet review."

> Say (if applicable): "LangSmith auto-engages when the env vars are
> set, so per-LLM-call flame graphs are one click away."

**1:45-2:00 — Wrap**

Back to the chat page.

> Say: "FHIR-backed EHR, separate audit log for PHI access, hardcoded
> safety triage that pre-empts the LLM, Tavily-cited medical content,
> token-streamed responses, exposed via MCP for any client to drive.
> 124 unit tests, two eval suites, GitHub Actions CI."

End frame: sidebar visible showing the stack labels + a screenshot of
the multi-intent CKD response.

## Recording tips

- **Loom 1080p, 30fps** is plenty.
- Disable Streamlit's "Always rerun" auto-rerun so the page doesn't
  flash mid-clip.
- Keep the cursor moving but slowly — no flicks.
- If a query takes more than 10s, edit the dead air out — Loom's
  trimmer is fine. Don't speed up the streaming response itself; the
  typing animation is the point.
- Embed the Loom in the README's "Live demo" section right under
  the Streamlit URL.

## What to NOT show

- Your actual API keys (the `.env` file). Sidebar shows the provider
  name, never the key.
- The HuggingFace token warning from sentence-transformers (run a
  query before starting the recording so the warning is already
  out of the way).
- Any non-fixture patient data — every name on screen should be one
  of the 5 synthetic patients.
