/**
 * SSE chat client.
 *
 * The browser's native EventSource only supports GET — we use a POST so the
 * body can carry the patient_id / thread_id without leaking to the URL. So
 * we hand-roll the SSE parse: fetch with ReadableStream, decode UTF-8,
 * split on blank lines, dispatch typed events to the supplied handlers.
 *
 * Handlers are best-effort: errors are caught and reported via onError so a
 * malformed event doesn't kill the stream.
 */
import { api } from "@/lib/api";
import type { AuditEvent, Booking } from "@/lib/api";

export type StatusEvent = {
  node: string;
  label: string;
  summary: string;
};

export type DoneEvent = {
  response: string;
  intents?: string[] | null;
  is_emergency?: boolean;
  emergency_categories?: string[] | null;
  appointment?: Booking | null;
  record_change?: { operation: string; patient_id: string; after?: Record<string, unknown> } | null;
  history_summary?: string | null;
  medical_info?: ({ synthesis?: string } & Record<string, unknown>)[] | null;
  sources?: { index: number; title: string; url: string; source: string }[] | null;
  tool_log?: Record<string, unknown>[] | null;
  error?: string | null;
};

export type ChatHandlers = {
  onStatus?: (e: StatusEvent) => void;
  onToken?: (text: string) => void;
  onDone?: (e: DoneEvent) => void;
  onError?: (msg: string) => void;
};

export type ChatRequest = {
  user_input: string;
  thread_id?: string;
  patient_id?: string | null;
  patient_name?: string | null;
};

export async function chatStream(
  req: ChatRequest,
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch(`${api.base}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(req),
    signal,
  });

  if (!r.ok || !r.body) {
    const body = await r.text().catch(() => "");
    handlers.onError?.(`${r.status} ${r.statusText}: ${body.slice(0, 200)}`);
    return;
  }

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE messages are separated by a blank line ("\n\n").
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      dispatch(raw, handlers);
    }
  }
  // Flush any trailing message without a blank-line terminator.
  if (buffer.trim()) dispatch(buffer, handlers);
}

function dispatch(raw: string, handlers: ChatHandlers): void {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith(":")) continue; // comment
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return;
  let payload: unknown;
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch (e) {
    handlers.onError?.(`malformed SSE payload: ${(e as Error).message}`);
    return;
  }

  try {
    if (event === "status" && handlers.onStatus) {
      handlers.onStatus(payload as StatusEvent);
    } else if (event === "token" && handlers.onToken) {
      const t = (payload as { text?: string }).text;
      if (t) handlers.onToken(t);
    } else if (event === "done" && handlers.onDone) {
      handlers.onDone(payload as DoneEvent);
    } else if (event === "error" && handlers.onError) {
      handlers.onError((payload as { message?: string }).message ?? "unknown error");
    }
  } catch (e) {
    handlers.onError?.(`handler threw: ${(e as Error).message}`);
  }
}

/** Helper for ad-hoc audit fetches scoped to a single patient. */
export async function fetchAuditForPatient(patient_id: string): Promise<AuditEvent[]> {
  return api.audit({ patient_id });
}
