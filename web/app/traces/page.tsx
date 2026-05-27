"use client";
import * as React from "react";
import { Activity, AlertTriangle, BarChart3, Timer } from "lucide-react";

import { AppShell } from "@/components/app-shell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import type { TraceEvent, TraceSummary } from "@/lib/api";
import { formatTimestamp } from "@/lib/utils";

export default function TracesPage() {
  return (
    <AppShell>
      <TracesView />
    </AppShell>
  );
}

function TracesView() {
  const [events, setEvents] = React.useState<TraceEvent[]>([]);
  const [summary, setSummary] = React.useState<TraceSummary | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    api
      .traces(200)
      .then(({ events, summary }) => {
        if (cancelled) return;
        setEvents(events);
        setSummary(summary);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="border-b px-6 py-4">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Activity className="h-4 w-4" />
          <span>Workflow traces · every chat invocation, every node</span>
        </div>
        <h1 className="mt-1 text-2xl font-semibold">Traces</h1>
      </header>

      <div className="grid grid-cols-2 gap-3 border-b p-4 md:grid-cols-5">
        <MetricCard
          label="Total runs"
          value={summary?.total ?? "—"}
          icon={<BarChart3 className="h-4 w-4" />}
        />
        <MetricCard
          label="p50 latency"
          value={summary?.p50_latency_seconds != null ? `${summary.p50_latency_seconds}s` : "—"}
          icon={<Timer className="h-4 w-4" />}
        />
        <MetricCard
          label="p95 latency"
          value={summary?.p95_latency_seconds != null ? `${summary.p95_latency_seconds}s` : "—"}
          icon={<Timer className="h-4 w-4" />}
        />
        <MetricCard
          label="Error rate"
          value={summary ? `${(summary.error_rate * 100).toFixed(1)}%` : "—"}
          icon={<AlertTriangle className="h-4 w-4" />}
          variant={summary && summary.error_rate > 0.05 ? "destructive" : "default"}
        />
        <MetricCard
          label="Emergencies"
          value={summary?.emergency_count ?? "—"}
          icon={<AlertTriangle className="h-4 w-4" />}
          variant={summary && summary.emergency_count > 0 ? "warning" : "default"}
        />
      </div>

      <div className="flex-1 overflow-auto">
        {loading ? (
          <div className="p-8 text-center text-sm text-muted-foreground">Loading…</div>
        ) : !events.length ? (
          <div className="p-8 text-center text-sm text-muted-foreground">
            No traces yet — chat with the assistant and they&apos;ll show up here.
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="sticky top-0 border-b bg-muted/50 backdrop-blur">
              <tr>
                <Th>When</Th>
                <Th>Actor</Th>
                <Th>Query</Th>
                <Th>Intents</Th>
                <Th>Backend</Th>
                <Th>Latency</Th>
                <Th>Flags</Th>
              </tr>
            </thead>
            <tbody>
              {events.map((e, i) => (
                <tr
                  key={`${e.ts}-${i}`}
                  className="border-b transition-colors hover:bg-muted/30"
                >
                  <Td>
                    <span className="font-mono text-xs text-muted-foreground">
                      {formatTimestamp(e.ts)}
                    </span>
                  </Td>
                  <Td>
                    <Badge variant="secondary" className="font-mono text-[10px]">
                      {e.actor}
                    </Badge>
                  </Td>
                  <Td className="max-w-[24rem]">
                    <span className="line-clamp-1 text-xs">{e.user_input}</span>
                  </Td>
                  <Td>
                    <div className="flex flex-wrap gap-1">
                      {(e.intents || []).map((i) => (
                        <Badge
                          key={i}
                          variant={i === "emergency" ? "destructive" : "outline"}
                          className="text-[9px] uppercase"
                        >
                          {i.replace("_", " ")}
                        </Badge>
                      ))}
                    </div>
                  </Td>
                  <Td>
                    {e.search_backend && (
                      <Badge
                        variant={e.search_backend === "stub" ? "warning" : "outline"}
                        className="font-mono text-[10px]"
                      >
                        {e.search_backend}
                      </Badge>
                    )}
                  </Td>
                  <Td>
                    <span className="font-mono text-xs text-muted-foreground">
                      {e.latency_seconds != null ? `${e.latency_seconds}s` : "—"}
                    </span>
                  </Td>
                  <Td>
                    <div className="flex gap-1">
                      {e.is_emergency && (
                        <Badge variant="destructive" className="text-[9px]">
                          🚨
                        </Badge>
                      )}
                      {(e.had_error || e.error) && (
                        <Badge variant="destructive" className="text-[9px]">
                          err
                        </Badge>
                      )}
                    </div>
                  </Td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  icon,
  variant = "default",
}: {
  label: string;
  value: string | number;
  icon?: React.ReactNode;
  variant?: "default" | "destructive" | "warning";
}) {
  return (
    <Card
      className={
        variant === "destructive"
          ? "border-destructive/40"
          : variant === "warning"
            ? "border-warning/40"
            : ""
      }
    >
      <CardHeader className="pb-1">
        <CardTitle className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
          {icon}
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-semibold">{value}</div>
      </CardContent>
    </Card>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th className="px-4 py-2 text-left text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
      {children}
    </th>
  );
}
function Td({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <td className={`px-4 py-2 align-middle ${className ?? ""}`}>{children}</td>;
}
