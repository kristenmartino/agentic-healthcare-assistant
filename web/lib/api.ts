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

// The Fly.io backend scales to zero and `suspend`-resumes; a genuine cold boot
// (fresh deploy / long idle) can still briefly reset or time-out connections
// while the machine wakes. Retry transient failures (network errors +
// 502/503/504) with backoff so a cold load self-heals instead of surfacing as
// a hard "Backend unreachable".
//
// Budget: the whole retry loop is bounded by a TRUE overall deadline of ~60s
// (TOTAL_RETRY_BUDGET_MS), measured in wall-clock time from the first attempt.
// Every per-attempt timeout and every backoff sleep is clamped to whatever
// remains of that budget, so a slow/hanging backend can never push total
// wall-clock time past ~60s — regardless of how many attempts or how long any
// single attempt hangs. RETRY_BACKOFF_MS is the *intended* backoff schedule;
// the deadline, not the schedule length, is what ultimately bounds the work.
const TOTAL_RETRY_BUDGET_MS = 60000;
const RETRY_BACKOFF_MS = [1000, 2000, 3000, 5000, 8000, 8000, 8000, 8000, 8000];
const RETRY_STATUSES = new Set([502, 503, 504]);
const ATTEMPT_TIMEOUT_MS = 15000;

// Resolve a signal's abort reason: prefer the caller's explicit reason, else a
// standard AbortError for runtimes that leave `reason` undefined.
function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException("Aborted", "AbortError");
}

// Sleep that rejects immediately if `signal` aborts, so a cancelled caller
// doesn't have to wait out a backoff interval before the loop notices. With no
// signal it's a plain setTimeout.
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortReason(signal));
      return;
    }
    const timer = setTimeout(resolve, ms);
    if (signal) {
      const sig = signal; // captured non-null for the closure
      sig.addEventListener(
        "abort",
        () => {
          clearTimeout(timer);
          reject(abortReason(sig));
        },
        { once: true },
      );
    }
  });
}

export async function fetchWithRetry(
  url: string,
  init: RequestInit,
): Promise<Response> {
  const startedAt = Date.now();
  // Caller-provided cancellation signal (e.g. a component unmounting mid-load).
  // Normalize null → undefined so the optional-chaining guards read cleanly.
  const callerSignal = init.signal ?? undefined;
  let lastErr: unknown;

  for (let attempt = 0; attempt <= RETRY_BACKOFF_MS.length; attempt++) {
    // Caller cancellation is TERMINAL — never start, or continue, work the
    // caller has abandoned. Checked at the top of every iteration so an abort
    // before the first attempt (or during a backoff sleep) short-circuits
    // immediately instead of being retried like a transient failure.
    if (callerSignal?.aborted) throw abortReason(callerSignal);

    const remaining = TOTAL_RETRY_BUDGET_MS - (Date.now() - startedAt);
    if (remaining <= 0) break;

    const ctrl = new AbortController();
    // Clamp the per-attempt timeout to the remaining budget so a single
    // hanging attempt can never push total wall-clock past TOTAL_RETRY_BUDGET_MS.
    const timer = setTimeout(
      () => ctrl.abort(),
      Math.min(ATTEMPT_TIMEOUT_MS, remaining),
    );

    // Compose the caller's signal onto this attempt's controller — don't just
    // spread `signal: ctrl.signal`, which would silently drop the caller's.
    // Manual addEventListener (not AbortSignal.any) for broad browser support;
    // cleaned up in `finally`.
    let onCallerAbort: (() => void) | undefined;
    if (callerSignal && !callerSignal.aborted) {
      onCallerAbort = () => ctrl.abort(callerSignal.reason);
      callerSignal.addEventListener("abort", onCallerAbort, { once: true });
    }

    try {
      const r = await fetch(url, { ...init, signal: ctrl.signal });
      // Gateway errors are emitted by the proxy while the machine wakes — retry.
      if (RETRY_STATUSES.has(r.status) && attempt < RETRY_BACKOFF_MS.length) {
        const sleepMs = Math.min(
          RETRY_BACKOFF_MS[attempt],
          TOTAL_RETRY_BUDGET_MS - (Date.now() - startedAt),
        );
        if (sleepMs <= 0) return r; // budget exhausted — surface what we have
        await sleep(sleepMs, callerSignal);
        continue;
      }
      return r;
    } catch (e) {
      // Caller cancellation is terminal — don't spend retries on it. This also
      // catches an abort that surfaced as the fetch rejecting: we tell it apart
      // from our own per-attempt timeout abort (a transient failure worth
      // retrying) by checking the caller signal directly.
      if (callerSignal?.aborted) throw abortReason(callerSignal);
      // Network error or per-attempt timeout (AbortError) — retry if budget left.
      lastErr = e;
      if (attempt < RETRY_BACKOFF_MS.length) {
        const sleepMs = Math.min(
          RETRY_BACKOFF_MS[attempt],
          TOTAL_RETRY_BUDGET_MS - (Date.now() - startedAt),
        );
        if (sleepMs <= 0) break; // budget exhausted — stop retrying
        await sleep(sleepMs, callerSignal);
        continue;
      }
    } finally {
      clearTimeout(timer);
      if (callerSignal && onCallerAbort) {
        callerSignal.removeEventListener("abort", onCallerAbort);
      }
    }
  }

  if (lastErr instanceof Error) throw lastErr;
  throw new Error("Backend did not respond before retry budget expired");
}

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetchWithRetry(`${API_BASE}${path}`, {
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
  // Latency breakdown (issue #12): per-node wall-clock durations (ms) plus
  // cold-start markers written by the API. Lets you tell a Fly boot apart
  // from slow in-request LLM hops.
  node_timings?: Record<string, number> | null;
  cold_start?: boolean;
  seconds_since_boot?: number;
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
