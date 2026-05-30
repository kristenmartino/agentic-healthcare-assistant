import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchWithRetry } from "./api";

/**
 * Tests for fetchWithRetry's deadline-bounded retry behavior.
 *
 * We fake timers so the backoff sleeps resolve instantly and Date.now()
 * advances only when we advance fake time — that lets us drive the ~60s
 * retry budget deterministically without any real waiting.
 *
 * `pump()` settles a fetchWithRetry promise by repeatedly flushing pending
 * timers (the per-attempt abort timer + each backoff sleep). Advancing time
 * also moves the faked Date.now(), so the overall deadline is honored.
 */
async function pump<T>(p: Promise<T>): Promise<T> {
  let done = false;
  // Attach a handler to `p` *synchronously* so it can never be flagged as an
  // unhandled rejection while we drive timers (the consumer attaches its own
  // handler later, after pump returns the already-settled promise — attaching a
  // second handler to an already-rejected promise is fine). The swallowing
  // mirror only tracks settlement via `done`; the real value/error is observed
  // by returning `p` itself.
  void p.then(
    () => {
      done = true;
    },
    () => {
      done = true;
    },
  );
  // Drain pending timers (per-attempt abort timers + backoff sleeps) and flush
  // microtasks between each, so promises settle deterministically. Advancing
  // fake time also moves Date.now(), so the overall deadline is honored.
  for (let i = 0; i < 100 && !done; i++) {
    await vi.runAllTimersAsync();
  }
  return p;
}

function jsonResponse(status: number): Response {
  return new Response(status === 204 ? null : "{}", {
    status,
    headers: { "content-type": "application/json" },
  });
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("fetchWithRetry", () => {
  it("retries on a 503 then succeeds", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(503))
      .mockResolvedValueOnce(jsonResponse(200));
    vi.stubGlobal("fetch", fetchMock);

    const res = await pump(fetchWithRetry("https://x.test/health", {}));

    expect(res.status).toBe(200);
    expect(fetchMock.mock.calls.length).toBeGreaterThan(1);
  });

  it("retries on a network error (TypeError) then succeeds", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(jsonResponse(200));
    vi.stubGlobal("fetch", fetchMock);

    const res = await pump(fetchWithRetry("https://x.test/health", {}));

    expect(res.status).toBe(200);
    expect(fetchMock.mock.calls.length).toBeGreaterThan(1);
  });

  it("honors the total retry budget when the backend hangs", async () => {
    // Simulate the exact footgun the reviewer found: a backend that hangs on
    // every attempt. Each attempt only settles when its per-attempt timeout
    // aborts the signal, so it burns real (fake) wall-clock time. Combined with
    // backoff sleeps, the ~60s deadline MUST stop the loop well before the
    // schedule's 10 attempts (each at full 15s) could ever run.
    const abortError = () => new DOMException("Aborted", "AbortError");
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => {
      // Hang until the per-attempt AbortController fires, then reject as a real
      // aborted fetch would. This makes each attempt consume (fake) wall-clock
      // time, so the overall deadline — not the schedule length — bounds the loop.
      return new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal;
        if (!signal) return; // hang forever (should never happen here)
        if (signal.aborted) {
          reject(abortError());
          return;
        }
        signal.addEventListener("abort", () => reject(abortError()), {
          once: true,
        });
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      pump(fetchWithRetry("https://x.test/health", {})),
    ).rejects.toThrow();

    // The intended schedule allows up to 10 attempts; the deadline must bound
    // it. A hanging attempt is aborted at ~15s, so 10 full attempts would take
    // ~150s + ~51s backoff (~201s) — proving the deadline (not the schedule)
    // is what ends the loop, far fewer attempts run.
    const calls = fetchMock.mock.calls.length;
    expect(calls).toBeGreaterThan(1);
    expect(calls).toBeLessThan(10);
  });

  it("does not retry a non-retryable status (404)", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(404));
    vi.stubGlobal("fetch", fetchMock);

    const res = await pump(fetchWithRetry("https://x.test/missing", {}));

    expect(res.status).toBe(404);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("stops within budget and returns the last 503 when every attempt is a gateway error", async () => {
    // Persistent gateway error: the loop must TERMINATE (never retry forever)
    // and hand back the last 503 within the retry budget — get() then surfaces
    // it as a thrown error. Proves the all-503 path is bounded too, not just
    // the hanging-connection path above.
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(503));
    vi.stubGlobal("fetch", fetchMock);

    const res = await pump(fetchWithRetry("https://x.test/health", {}));

    expect(res.status).toBe(503);
    expect(fetchMock.mock.calls.length).toBeGreaterThan(1);
    // Bounded by the attempt schedule / deadline — never unbounded.
    expect(fetchMock.mock.calls.length).toBeLessThanOrEqual(10);
  });

  it("rejects immediately with the caller's reason when the signal is already aborted (no fetch)", async () => {
    // A caller that has already given up (e.g. component unmounted before the
    // request started) must short-circuit before any network call.
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200));
    vi.stubGlobal("fetch", fetchMock);

    const reason = new DOMException("cancelled before start", "AbortError");
    const ctrl = new AbortController();
    ctrl.abort(reason);

    await expect(
      pump(fetchWithRetry("https://x.test/health", { signal: ctrl.signal })),
    ).rejects.toBe(reason);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("stops promptly with the caller's reason when aborted during an in-flight attempt", async () => {
    // The footgun the review flagged: a caller abort must be terminal, not
    // retried like a transient failure. Here the attempt hangs until its
    // signal aborts; we trip the CALLER signal mid-flight and expect a prompt
    // rejection with the caller's reason and NO further attempts.
    const abortError = () => new DOMException("Aborted", "AbortError");
    const reason = new DOMException("user navigated away", "AbortError");
    const ctrl = new AbortController();
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => {
      return new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal;
        if (signal?.aborted) return reject(abortError());
        signal?.addEventListener("abort", () => reject(abortError()), {
          once: true,
        });
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    // Abort the caller well before the 15s per-attempt timeout would fire.
    setTimeout(() => ctrl.abort(reason), 100);

    await expect(
      pump(fetchWithRetry("https://x.test/health", { signal: ctrl.signal })),
    ).rejects.toBe(reason);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("stops retrying when the caller aborts during a backoff sleep", async () => {
    // Abort lands BETWEEN attempts, while we're waiting out the backoff. The
    // abortable sleep must reject promptly with the caller's reason so the loop
    // never makes the next attempt (the 200 mock is never reached).
    const reason = new DOMException("cancelled mid-backoff", "AbortError");
    const ctrl = new AbortController();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(503)) // attempt 0 → triggers backoff
      .mockResolvedValue(jsonResponse(200)); // would succeed if we retried
    vi.stubGlobal("fetch", fetchMock);

    // Fire during the ~1s backoff after the first 503.
    setTimeout(() => ctrl.abort(reason), 500);

    await expect(
      pump(fetchWithRetry("https://x.test/health", { signal: ctrl.signal })),
    ).rejects.toBe(reason);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
