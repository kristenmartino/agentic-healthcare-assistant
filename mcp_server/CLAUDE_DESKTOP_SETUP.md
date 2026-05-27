# Using the Healthcare Assistant from Claude Desktop (MCP)

The Healthcare Assistant's MCP server exposes its 8 tools to any MCP client
— most notably Claude Desktop, where you can ask Claude to book appointments,
look up patient history, run medical searches, or inspect the audit log,
and it'll call the right tool with the right arguments.

## What you'll demo

A 90-second flow you can record as a Loom for the portfolio README:

1. *"Find Anjali Mehra's record and tell me her active conditions."*
   → Claude calls `find_patient` then `get_history`. Shows the FHIR-backed
   record + condition list inline in the chat.
2. *"Book her a nephrologist for next Tuesday."*
   → Claude calls `book_appointment`. The booking shows up in your local
   `data/appointments.sqlite` and in the Doctor View dashboard.
3. *"What are the latest treatment options for chronic kidney disease?"*
   → Claude calls `medical_search` against MedlinePlus/WHO via Tavily.
4. *"Show me the last 5 audit log events for Anjali."*
   → Claude calls `get_audit_log(patient_id="fhir:anjali-mehra")`, you see
   one row per tool call from steps 1–2.

## Prerequisites

- Python 3.11+ available on PATH
- Claude Desktop installed (https://claude.ai/download)
- This repo cloned locally with `pip install -r requirements.txt` plus
  `pip install 'mcp[cli]'` (the MCP runtime)
- Optionally an `.env` set up with `ANTHROPIC_API_KEY` and `TAVILY_API_KEY`
  (the MCP tools work without an LLM key — they expose raw data — but
  `medical_search` returns richer content with Tavily configured)

## Configuration

Edit Claude Desktop's MCP config:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

Add the `healthcare-assistant` entry (preserve any other `mcpServers` you
already have):

```json
{
  "mcpServers": {
    "healthcare-assistant": {
      "command": "python",
      "args": [
        "-m",
        "mcp_server.healthcare_mcp"
      ],
      "cwd": "/ABSOLUTE/PATH/TO/agentic-healthcare-assistant",
      "env": {
        "EHR_BACKEND": "fhir_fixture",
        "TAVILY_API_KEY": "tvly-..."
      }
    }
  }
}
```

Replace `/ABSOLUTE/PATH/TO/...` with the absolute path to your local clone.
Set `EHR_BACKEND=sqlite` if you've seeded the original records.xlsx, or
`EHR_BACKEND=fhir` if you have a HAPI FHIR server reachable (see README).

## Restart Claude Desktop

Quit completely (⌘Q on Mac) and reopen. In a new conversation, click the
🔌 icon next to the message input — `healthcare-assistant` should be
listed under "Available tools" with all 8 tools.

## Tool list

| Tool | Use |
|---|---|
| `book_appointment(patient_name, specialty, preferred_date?)` | Books the earliest slot for a specialty |
| `list_doctors(specialty?)` | Lists doctors (filter by specialty) |
| `find_patient(name)` | Returns one patient record by name |
| `list_patients()` | Lists all patients |
| `upsert_patient(...)` | Adds or updates a patient |
| `get_history(patient_name)` | Returns the record + matching PDF chunks |
| `medical_search(query, top_k=4)` | Trusted-domain web search |
| `get_audit_log(patient_id?, action_prefix?, limit=50)` | PHI access log |

## Verifying without Claude Desktop

If you just want to confirm the server starts cleanly:

```bash
# Stdio mode — Claude Desktop talks to it this way
python -m mcp_server.healthcare_mcp

# HTTP mode — for curl or custom MCP clients
python -m mcp_server.healthcare_mcp --http
# then: curl http://127.0.0.1:8000/mcp/...

# Dry-run — calls every tool directly without an MCP runtime
python -m mcp_server.healthcare_mcp --dry-run
```

The dry-run prints one summary line per tool and exits 0 when everything
wired up correctly.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Claude Desktop shows "0 tools available" for healthcare-assistant | Server didn't start | Check Claude Desktop logs at `~/Library/Logs/Claude/mcp-server-healthcare-assistant.log` |
| `ModuleNotFoundError: No module named 'mcp_server'` | `cwd` is wrong | Use the absolute path; pwd inside the repo and copy it |
| Tools listed but every call returns "no records" | `EHR_BACKEND` not set / fixtures missing | Confirm `data/fhir_fixtures/patients.json` exists; or set `EHR_BACKEND=sqlite` after running `python seed.py` |
| `medical_search` returns `source: "stub"` | No Tavily key + DDG rate-limited | Set `TAVILY_API_KEY` in the `env` block above |

## Loom recording script

Suggested script for the 90-second video:

> "I'm going to use Claude Desktop to drive my Healthcare Assistant via
> MCP — no separate UI, just natural language. *(Open Claude Desktop,
> click the 🔌 icon to show the 8 tools listed.)* I'll ask it to look up
> Anjali Mehra. *(Type the query; Claude calls `find_patient` then
> `get_history`.)* Now book her a nephrologist next Tuesday — Claude
> calls `book_appointment` and gets back a confirmation number. Finally,
> show me the audit trail. *(Claude calls `get_audit_log`; the response
> shows three rows for the three tool calls I just made — every PHI
> access is recorded.)*  Same backend the Streamlit UI uses, exposed via
> MCP so any client can drive it."

Length target: 70-100 seconds. Embed the resulting Loom in the README's
"Live demo" section right under the Streamlit URL.
