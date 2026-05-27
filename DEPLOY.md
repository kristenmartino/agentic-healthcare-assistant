# Deploying to Streamlit Cloud

Five-minute path from the merged `main` branch to a live URL.

## Prerequisites

- This repo, public on GitHub (it already is)
- A Streamlit Cloud account (free; sign in with GitHub at https://share.streamlit.io)
- An LLM key — Anthropic (recommended), Groq, or OpenAI
- Optional: a Tavily key for real medical citations

## One-time setup

1. **Open https://share.streamlit.io** → "New app".
2. **Repository**: `kristenmartino/agentic-healthcare-assistant`. **Branch**: `main`. **Main file path**: `app.py`.
3. **Advanced settings → Secrets** — paste the TOML below (filling in real keys). These map to environment variables in the deployed container.

   ```toml
   # LLM (pick one; Anthropic is the default-best)
   ANTHROPIC_API_KEY = "sk-ant-..."
   ANTHROPIC_MODEL = "claude-sonnet-4-6"

   # Optional alternatives — only used if ANTHROPIC_API_KEY is absent
   # OPENAI_API_KEY = "sk-..."
   # GROQ_API_KEY   = "gsk_..."

   # Optional: real medical citations from MedlinePlus/WHO/CDC
   TAVILY_API_KEY = "tvly-..."

   # EHR backend — use the bundled FHIR fixtures so the demo works
   # without a separate HAPI FHIR server. To swap, set EHR_BACKEND="fhir"
   # and FHIR_BASE_URL.
   EHR_BACKEND = "fhir_fixture"

   # Disable cross-session persistence on a shared demo (anyone using the
   # app would otherwise resume the previous user's thread).
   ENABLE_PERSISTENCE = "false"

   # Observability — optional. Set both to enable LangSmith.
   # LANGCHAIN_API_KEY = "ls_..."
   # LANGCHAIN_TRACING_V2 = "true"
   # LANGCHAIN_PROJECT = "agentic-healthcare-assistant"
   ```

4. **Python version**: 3.11 or 3.12 (Streamlit Cloud's default is fine).
5. **Deploy**. First build takes 3-5 minutes (pulls langgraph, langchain, sentence-transformers).

## After deploy

- Public URL will be `https://<your-app-slug>.streamlit.app`.
- Hit the URL, try the canonical multi-intent CKD query, then open the **📊 Traces** page to see your own query logged.
- Update the README's "Live demo" section with the URL.

## Live-demo housekeeping

- **No real PHI**: the bundled FHIR fixtures are 5 synthetic patients (Anjali Mehra, David Thompson, Ramesh Kulkarni, Rebeca Nagle, Priya Narayan). The disclaimer banner at the top of every page makes this explicit.
- **Shared session**: `ENABLE_PERSISTENCE=false` is the right call for a public URL — otherwise the SqliteSaver checkpointer would let any visitor resume the previous visitor's thread.
- **Cost ceiling**: prompt caching (T8) is on by default — system prompts get cached at ~10% of input cost for repeat calls. On Streamlit Cloud's free tier the limiter is more about LLM API spend than compute; an ANTHROPIC_API_KEY with a $10 monthly cap is plenty for portfolio traffic.
- **Restart on secrets change**: Streamlit Cloud picks up `[secrets]` edits automatically but does NOT restart the running container — go to "Manage app" → "Reboot" after editing.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `LLM provider: stub` in the sidebar | No API key set in secrets | Re-check `[secrets]` block; reboot the app |
| "no available slots" on every booking | `ensure_future_slots` couldn't write to `/mount/src/.../data/appointments.sqlite` | Streamlit Cloud's filesystem is ephemeral but writable — should self-heal on reboot |
| Audit Log page empty | No queries yet → nothing logged | Run one query first |
| `WARNING: FAISS lookup failed: ... size 384 is different from 512` | Stale index from a previous deploy | Trigger a rebuild (push a no-op commit) |
