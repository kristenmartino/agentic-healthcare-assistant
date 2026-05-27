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

  React.useEffect(() => {
    let cancelled = false;
    Promise.all([api.config(), api.patients()])
      .then(([c, p]) => {
        if (cancelled) return;
        setConfig(c);
        setPatients(p);
      })
      .catch((e: Error) => {
        if (cancelled) return;
        setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <ShellContext.Provider value={{ config, patients, error }}>
      <div className="flex h-screen">
        <Sidebar config={config} patients={patients} />
        <main className="flex flex-1 flex-col overflow-hidden">
          {error && (
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
