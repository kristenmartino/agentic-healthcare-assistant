"use client";
import * as React from "react";
import { Sidebar } from "@/components/sidebar";
import { api } from "@/lib/api";
import type { Config, Patient } from "@/lib/api";

/**
 * Shell context: every page reads from here for the patient list + config
 * rather than receiving them via cloneElement (which fights TypeScript and
 * makes children types invariant).
 */
type ShellContextValue = {
  config: Config | null;
  patients: Patient[];
  error: string | null;
};
const ShellContext = React.createContext<ShellContextValue>({
  config: null,
  patients: [],
  error: null,
});

export function useShell() {
  return React.useContext(ShellContext);
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const [config, setConfig] = React.useState<Config | null>(null);
  const [patients, setPatients] = React.useState<Patient[]>([]);
  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [waking, setWaking] = React.useState(false);

  React.useEffect(() => {
    let cancelled = false;
    // A warm backend answers in well under a second; if we're still waiting
    // after 2.5s the scale-to-zero server is almost certainly cold-booting,
    // so show a friendly "waking up" note instead of nothing. api.ts retries
    // the request underneath for up to ~50s before this ever rejects.
    const wakeTimer = setTimeout(() => {
      if (!cancelled) setWaking(true);
    }, 2500);
    Promise.all([api.config(), api.patients()])
      .then(([c, p]) => {
        if (cancelled) return;
        setConfig(c);
        setPatients(p);
        setError(null);
      })
      .catch((e: Error) => {
        if (cancelled) return;
        setError(e.message);
      })
      .finally(() => {
        if (cancelled) return;
        clearTimeout(wakeTimer);
        setWaking(false);
        setLoading(false);
      });
    return () => {
      cancelled = true;
      clearTimeout(wakeTimer);
    };
  }, []);

  return (
    <ShellContext.Provider value={{ config, patients, error }}>
      <div className="flex h-screen">
        <Sidebar config={config} patients={patients} />
        <main className="flex flex-1 flex-col overflow-hidden">
          {loading && waking && (
            <div className="border-b bg-muted px-4 py-2 text-xs text-muted-foreground">
              ⏳ Waking up the backend… the free-tier server sleeps when idle and
              can take up to a minute to start. Retrying automatically.
            </div>
          )}
          {!loading && error && (
            <div className="border-b bg-destructive/10 px-4 py-2 text-xs text-destructive">
              ⚠️ Backend unreachable: {error}. Verify {api.base} is up; check
              NEXT_PUBLIC_API_BASE.
            </div>
          )}
          {children}
        </main>
      </div>
    </ShellContext.Provider>
  );
}
