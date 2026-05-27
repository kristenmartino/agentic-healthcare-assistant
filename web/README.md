# Healthcare Assistant — Next.js frontend

Next.js 15 + shadcn/ui + Tailwind. Talks to the FastAPI service at `../api/`
over SSE (chat) + plain JSON (dashboards). The Streamlit app under `../app.py`
still works — this is the polished portfolio frontend, deployed alongside.

## Develop locally

```bash
# 1. Backend (in a separate terminal, from the repo root)
pip install -r requirements.txt -r api/requirements.txt
EHR_BACKEND=fhir_fixture uvicorn api.main:app --reload --port 8000

# 2. Frontend
cd web
npm install
cp .env.example .env.local                # default points at localhost:8000
npm run dev
```

App at http://localhost:3000.

## Deploy

```bash
# Backend on Fly.io (one-time)
flyctl auth login
flyctl launch --no-deploy --config api/fly.toml --dockerfile api/Dockerfile
flyctl secrets set ANTHROPIC_API_KEY=sk-ant-... TAVILY_API_KEY=tvly-...
flyctl secrets set EHR_BACKEND=fhir_fixture ENABLE_PERSISTENCE=false
flyctl secrets set CORS_ALLOW_ORIGINS=https://<your-vercel-domain>
flyctl deploy --config api/fly.toml --dockerfile api/Dockerfile

# Frontend on Vercel
# 1. "Import Project" → kristenmartino/agentic-healthcare-assistant
# 2. Set Root Directory: web/
# 3. Environment variable: NEXT_PUBLIC_API_BASE = https://<your-fly-app>.fly.dev
# 4. Deploy
```

## Routes

| Route | What it shows |
|---|---|
| `/` | Streaming chat, sidebar patient picker, status strip, per-message state artifacts (intents / sources / appointment / errors) |
| `/audit` | PHI access log table with patient/actor/action filters + CSV export |
| `/traces` | Workflow trace dashboard — total runs, p50/p95 latency, error rate, emergency count, filterable event list |
| `/about` | Architecture overview, GitHub link, stack badges |

## Tech notes

- **SSE** is hand-rolled in `lib/chat.ts` because EventSource doesn't support POST; we use fetch + ReadableStream to keep the patient_id off the URL.
- **State** is Zustand (`lib/store.ts`) — per-slice subscriptions so the patient picker doesn't re-render on every streamed token.
- **Theme** uses next-themes with system-preference detection; the toggle is in the sidebar.
- **Type-safe**: every API response is typed in `lib/api.ts` matching the FastAPI Pydantic shapes.
