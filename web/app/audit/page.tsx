"use client";
import * as React from "react";
import { Download, ShieldCheck } from "lucide-react";

import { useShell } from "@/components/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api } from "@/lib/api";
import type { AuditEvent } from "@/lib/api";
import { formatTimestamp } from "@/lib/utils";

export default function AuditPage() {
  return <AuditView />;
}

function AuditView() {
  const { patients } = useShell();
  const [events, setEvents] = React.useState<AuditEvent[]>([]);
  const [summary, setSummary] = React.useState<{
    total: number;
    by_action: Record<string, number>;
    by_actor: Record<string, number>;
  } | null>(null);
  const [patientFilter, setPatientFilter] = React.useState("all");
  const [actorFilter, setActorFilter] = React.useState("all");
  const [actionFilter, setActionFilter] = React.useState("all");
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      api.audit({
        patient_id: patientFilter === "all" ? undefined : patientFilter,
        actor: actorFilter === "all" ? undefined : actorFilter,
        action_prefix: actionFilter === "all" ? undefined : actionFilter,
        limit: 500,
      }),
      api.auditSummary(),
    ])
      .then(([rows, s]) => {
        if (cancelled) return;
        setEvents(rows);
        setSummary(s);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [patientFilter, actorFilter, actionFilter]);

  const downloadCsv = () => {
    if (!events.length) return;
    const cols = ["ts", "actor", "action", "resource_type", "resource_id", "patient_id", "outcome", "details"];
    const lines = [
      cols.join(","),
      ...events.map((e) =>
        cols
          .map((c) => {
            const v = (e as unknown as Record<string, unknown>)[c];
            const s = typeof v === "object" ? JSON.stringify(v) : String(v ?? "");
            return `"${s.replace(/"/g, '""')}"`;
          })
          .join(","),
      ),
    ].join("\n");
    const blob = new Blob([lines], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `audit-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
  };

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="border-b px-6 py-4">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <ShieldCheck className="h-4 w-4" />
          <span>PHI access audit log · HIPAA 45 CFR 164.312(b)</span>
        </div>
        <h1 className="mt-1 text-2xl font-semibold">Audit Log</h1>
      </header>

      <div className="grid grid-cols-2 gap-3 border-b p-4 md:grid-cols-4">
        <MetricCard label="Total events" value={summary?.total ?? "—"} />
        <MetricCard
          label="Distinct actions"
          value={summary ? Object.keys(summary.by_action).length : "—"}
        />
        <MetricCard
          label="Distinct actors"
          value={summary ? Object.keys(summary.by_actor).length : "—"}
        />
        <MetricCard
          label="Top action"
          value={
            summary ? Object.keys(summary.by_action)[0] ?? "—" : "—"
          }
        />
      </div>

      <div className="border-b p-4">
        <div className="flex flex-wrap items-center gap-2">
          <Select value={patientFilter} onValueChange={setPatientFilter}>
            <SelectTrigger className="w-[220px]">
              <SelectValue placeholder="All patients" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All patients</SelectItem>
              {patients?.map((p) => (
                <SelectItem key={p.patient_id} value={p.patient_id}>
                  {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={actorFilter} onValueChange={setActorFilter}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="All actors" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All actors</SelectItem>
              {summary &&
                Object.keys(summary.by_actor).map((a) => (
                  <SelectItem key={a} value={a}>
                    {a}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>

          <Select value={actionFilter} onValueChange={setActionFilter}>
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="All actions" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All actions</SelectItem>
              {summary &&
                Object.keys(summary.by_action).map((a) => (
                  <SelectItem key={a} value={a}>
                    {a}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>

          <div className="ml-auto flex items-center gap-2 text-sm text-muted-foreground">
            <span>{events.length} events</span>
            <Button
              variant="outline"
              size="sm"
              onClick={downloadCsv}
              disabled={!events.length}
            >
              <Download className="mr-1.5 h-3.5 w-3.5" /> CSV
            </Button>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        {loading ? (
          <div className="p-8 text-center text-sm text-muted-foreground">Loading…</div>
        ) : !events.length ? (
          <div className="p-8 text-center text-sm text-muted-foreground">
            No audit events match these filters. Run a query in the chat first.
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="sticky top-0 border-b bg-muted/50 backdrop-blur">
              <tr>
                <Th>Timestamp</Th>
                <Th>Actor</Th>
                <Th>Action</Th>
                <Th>Resource</Th>
                <Th>Patient</Th>
                <Th>Outcome</Th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => (
                <tr
                  key={e.id}
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
                  <Td>
                    <span className="font-mono text-xs">{e.action}</span>
                  </Td>
                  <Td>
                    <span className="text-xs">
                      {e.resource_type && (
                        <Badge variant="outline" className="mr-1 font-mono text-[10px]">
                          {e.resource_type}
                        </Badge>
                      )}
                      <span className="font-mono text-muted-foreground">
                        {e.resource_id || "—"}
                      </span>
                    </span>
                  </Td>
                  <Td>
                    <span className="font-mono text-xs text-muted-foreground">
                      {e.patient_id || "—"}
                    </span>
                  </Td>
                  <Td>
                    <Badge
                      variant={
                        e.outcome === "success"
                          ? "success"
                          : e.outcome === "error"
                            ? "destructive"
                            : "secondary"
                      }
                      className="text-[10px]"
                    >
                      {e.outcome}
                    </Badge>
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

function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <Card>
      <CardHeader className="pb-1">
        <CardTitle className="text-xs font-medium text-muted-foreground">
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
function Td({ children }: { children: React.ReactNode }) {
  return <td className="px-4 py-2 align-middle">{children}</td>;
}
