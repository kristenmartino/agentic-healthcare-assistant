import { ExternalLink, GitBranch, Github } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const features = [
  {
    title: "FHIR R4 EHR backend",
    body: "Pluggable EHR layer — SQLite for the course demo, FHIR R4 against any HAPI / AWS HealthLake / Microsoft FHIR server in production. Conditions are SNOMED-coded; Observations are LOINC-coded. History summaries cite these directly.",
  },
  {
    title: "PHI access audit log",
    body: "Every patient-identifiable read or write produces one row in a separate SQLite DB (HIPAA 45 CFR 164.312(b) shape). The audit DB is intentionally distinct from the EHR DB — separation-of-duties so an EHR compromise can't silently erase the trail.",
  },
  {
    title: "Clinical safety classifier",
    body: "Pre-classifier between START and intent routing. Deterministic regex catches cardiac / stroke / suicide / anaphylaxis / severe-bleeding / altered-mental-status; emits a hardcoded 911/988/112/108 template and skips the LLM entirely. Informational guards suppress the obvious false positives.",
  },
  {
    title: "Multi-intent fan-out",
    body: "The classifier may return multiple intents — LangGraph fans out to all relevant branches in parallel, then converges on the composer. State fields with multiple writers use Annotated reducers to avoid silent overwrites.",
  },
  {
    title: "Token-streamed responses",
    body: "Mixed stream_mode=[updates, messages] surfaces node-progress AND composer tokens in the same loop. The chat bubble fills in real-time; node-progress badges animate in as each branch completes.",
  },
  {
    title: "MCP server",
    body: "The 8 tools the LangGraph agent uses internally are also exposed via Anthropic's Model Context Protocol, so Claude Desktop (or any MCP client) can drive the same booking / records / history / search / audit operations.",
  },
  {
    title: "Two-tier evaluation",
    body: "Routing eval (14 cases — plumbing + safety classifier) plus adversarial eval (20 cases — jailbreaks, PHI isolation, dangerous advice, prompt injection, refusal quality). LLM-as-judge plus deterministic substring backstop; CI red-lines on safety < 3.",
  },
  {
    title: "Observability",
    body: "Every workflow invocation produces one JSONL trace row with timing, intents, backends, errors. Optional LangSmith auto-engages when LANGCHAIN_API_KEY is set for per-LLM-call flame graphs.",
  },
];

export default function AboutPage() {
  return (
    <AppShell>
      <div className="flex h-full flex-col overflow-y-auto">
        <header className="border-b px-6 py-6">
          <div className="text-xs text-muted-foreground">About</div>
          <h1 className="mt-1 text-2xl font-semibold">Healthcare Assistant</h1>
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
            LangGraph-backed agentic assistant: FHIR-grounded patient
            lookups, PHI-audit-traceable tool calls, deterministic clinical
            safety triage, multi-intent fan-out, exposed via MCP for
            external clients.
          </p>
          <div className="mt-3 flex flex-wrap gap-1.5">
            <Badge variant="secondary">LangGraph</Badge>
            <Badge variant="secondary">FastAPI</Badge>
            <Badge variant="secondary">Next.js 15</Badge>
            <Badge variant="secondary">shadcn/ui</Badge>
            <Badge variant="secondary">Anthropic Claude</Badge>
            <Badge variant="secondary">FHIR R4</Badge>
            <Badge variant="secondary">MCP</Badge>
            <Badge variant="secondary">Tavily</Badge>
          </div>
          <div className="mt-4 flex flex-wrap gap-2 text-sm">
            <a
              href="https://github.com/kristenmartino/agentic-healthcare-assistant"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 hover:bg-secondary/40"
            >
              <Github className="h-4 w-4" /> GitHub
              <ExternalLink className="h-3 w-3 text-muted-foreground" />
            </a>
            <a
              href="https://github.com/kristenmartino/agentic-healthcare-assistant/blob/main/README.md"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 hover:bg-secondary/40"
            >
              <GitBranch className="h-4 w-4" /> Architecture docs
              <ExternalLink className="h-3 w-3 text-muted-foreground" />
            </a>
          </div>
        </header>

        <div className="mx-auto grid w-full max-w-5xl grid-cols-1 gap-4 p-6 md:grid-cols-2">
          {features.map((f) => (
            <Card key={f.title}>
              <CardHeader>
                <CardTitle className="text-base">{f.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>{f.body}</CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>

        <footer className="border-t px-6 py-4 text-xs text-muted-foreground">
          Synthetic data; no real PHI. Informational only — not a substitute
          for clinical care.
        </footer>
      </div>
    </AppShell>
  );
}
