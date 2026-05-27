/**
 * API client for the FastAPI backend.
 *
 * All routes here are typed and return Promises. SSE chat is in chat.ts —
 * separate so the bundler can tree-shake it from the dashboard pages that
 * don't need streaming support.
 */

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ||
  "http://localhost:8000";

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { Accept: "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!r.ok) {
    const body = await r.text();
    throw new Error(`${r.status} ${r.statusText}: ${body.slice(0, 200)}`);
  }
  return r.json();
}

// ---------- shared types ----------

export type LLMProvider = "anthropic" | "groq" | "openai" | "stub";

export type Config = {
  llm_provider: LLMProvider;
  llm_model: string;
  ehr_backend: "sqlite" | "fhir" | "fhir_fixture";
  fhir_base_url: string | null;
  search_backend_intended: "tavily" | "duckduckgo";
  tavily_configured: boolean;
  langsmith_enabled: boolean;
  langsmith_project: string | null;
  prompt_caching_enabled: boolean;
};

export type Patient = {
  patient_id: string;
  name: string;
  age: number | null;
  gender: string | null;
  phone_raw?: string | null;
  email?: string | null;
  address?: string | null;
  summary: string | null;
  fhir_id?: string;
};

export type Condition = { code?: { text?: string; coding?: { display?: string }[] } };
export type Observation = {
  name: string;
  value: number | null;
  unit: string | null;
  date: string | null;
};

export type PatientDetail = Patient & {
  clinical: {
    conditions: Condition[];
    observations: Observation[];
  };
};

export type Booking = {
  slot_id: number;
  doctor_name: string;
  specialty: string;
  start_time: string;
  end_time: string;
  confirmation_no: string;
  booked_by_patient_id: string;
  booked_at: string;
};

export type SpecialtyStat = {
  specialty: string;
  total_slots: number;
  booked_slots: number;
  utilization_pct: number;
};

export type AuditEvent = {
  id: number;
  ts: string;
  actor: string;
  action: string;
  resource_type: string | null;
  resource_id: string | null;
  patient_id: string | null;
  outcome: string;
  details: Record<string, unknown>;
};

export type TraceEvent = {
  ts: string;
  thread_id: string;
  actor: string;
  user_input: string;
  intents?: string[] | null;
  is_emergency?: boolean;
  patient_id?: string | null;
  node_count?: number;
  search_backend?: string | null;
  had_error?: boolean;
  langsmith_enabled?: boolean;
  latency_seconds?: number;
  error?: string;
  error_type?: string;
};

export type TraceSummary = {
  total: number;
  p50_latency_seconds: number | null;
  p95_latency_seconds: number | null;
  error_rate: number;
  emergency_count: number;
};

// ---------- routes ----------

export const api = {
  base: API_BASE,
  config: () => get<Config>("/config"),
  health: () => get<{ status: string }>("/health"),
  patients: () => get<Patient[]>("/patients"),
  patient: (id: string) => get<PatientDetail>(`/patients/${encodeURIComponent(id)}`),
  searchPatient: (name: string) =>
    get<Patient | null>(
      `/patients/search/by-name?name=${encodeURIComponent(name)}`,
    ),
  recentBookings: (limit = 20) =>
    get<Booking[]>(`/bookings/recent?limit=${limit}`),
  specialtyStats: () => get<SpecialtyStat[]>("/bookings/specialty-stats"),
  audit: (params: {
    patient_id?: string;
    action_prefix?: string;
    actor?: string;
    limit?: number;
  } = {}) => {
    const qs = new URLSearchParams();
    if (params.patient_id) qs.set("patient_id", params.patient_id);
    if (params.action_prefix) qs.set("action_prefix", params.action_prefix);
    if (params.actor) qs.set("actor", params.actor);
    qs.set("limit", String(params.limit ?? 100));
    return get<AuditEvent[]>(`/audit?${qs}`);
  },
  auditSummary: () =>
    get<{
      total: number;
      by_action: Record<string, number>;
      by_actor: Record<string, number>;
    }>("/audit/summary"),
  traces: (limit = 100) =>
    get<{ events: TraceEvent[]; summary: TraceSummary }>(`/traces?limit=${limit}`),
};
