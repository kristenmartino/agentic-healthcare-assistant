"use client";
import * as React from "react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AlertCircle, AlertTriangle, CalendarCheck, FileText, Search, Stethoscope, User } from "lucide-react";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { cn, formatAppointmentTime } from "@/lib/utils";
import type { ChatMessage } from "@/lib/store";

// Keep in sync with state.Intent on the backend. `schedule` and `audit`
// are agent_loop-mode-only (the classifier graph never emits them).
const intentBadgeVariant: Record<string, "default" | "secondary" | "warning" | "destructive" | "success"> = {
  booking: "default",
  records: "secondary",
  history: "secondary",
  medical_search: "secondary",
  schedule: "default",
  audit: "warning",
  general: "secondary",
  emergency: "destructive",
};

export function Message({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className={cn("flex gap-3", isUser ? "flex-row-reverse" : "flex-row")}
    >
      <Avatar
        className={cn(
          "h-8 w-8 shrink-0",
          isUser ? "border border-border" : "bg-primary/10",
        )}
      >
        <AvatarFallback
          className={cn(
            "text-[10px]",
            isUser ? "bg-secondary" : "bg-primary/10 text-primary",
          )}
        >
          {isUser ? <User className="h-4 w-4" /> : <Stethoscope className="h-4 w-4" />}
        </AvatarFallback>
      </Avatar>
      <div
        className={cn(
          "flex max-w-[80%] flex-col gap-2",
          isUser ? "items-end" : "items-start",
        )}
      >
        <Card
          className={cn(
            "px-4 py-3 prose-chat text-sm",
            isUser
              ? "border-primary/20 bg-primary/5"
              : message.state?.is_emergency
                ? "border-destructive/40 bg-destructive/5"
                : "",
          )}
        >
          {message.streaming && !message.content ? (
            <ThinkingIndicator />
          ) : (
            <div className={cn(message.streaming && "streaming-cursor")}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content || (message.streaming ? "" : "(no response)")}
              </ReactMarkdown>
            </div>
          )}
        </Card>

        {!isUser && message.state && <AssistantArtifacts state={message.state} />}
      </div>
    </motion.div>
  );
}

function ThinkingIndicator() {
  // A bouncing-dots animation alone loops silently and can still feel hung
  // on a slow turn. A ticking elapsed counter (shown once we cross ~2s) is
  // unambiguous proof the request is still alive, not stuck. The timer
  // starts on mount — i.e. the moment the streaming bubble first renders —
  // and clears when content arrives and this component unmounts.
  const [elapsed, setElapsed] = React.useState(0);
  React.useEffect(() => {
    const start = Date.now();
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex items-center gap-2 py-1 text-muted-foreground">
      <div className="flex items-center gap-1" aria-hidden>
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current [animation-delay:-0.3s]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current [animation-delay:-0.15s]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current" />
      </div>
      <span className="text-xs" role="status" aria-live="polite">
        Working…{elapsed >= 2 ? ` ${elapsed}s` : ""}
      </span>
    </div>
  );
}

function AssistantArtifacts({ state }: { state: NonNullable<ChatMessage["state"]> }) {
  const items: React.ReactNode[] = [];

  if (state.intents?.length) {
    items.push(
      <div key="intents" className="flex flex-wrap items-center gap-1">
        {state.intents.map((i) => (
          <Badge
            key={i}
            variant={intentBadgeVariant[i] ?? "secondary"}
            className="text-[10px] uppercase tracking-wide"
          >
            {i.replace("_", " ")}
          </Badge>
        ))}
      </div>,
    );
  }

  if (state.is_emergency) {
    items.push(
      <div
        key="emergency"
        className="flex items-center gap-1.5 text-xs font-medium text-destructive"
      >
        <AlertTriangle className="h-3.5 w-3.5" />
        Safety classifier fired — LLM bypassed
      </div>,
    );
  }

  if (state.appointment) {
    const a = state.appointment;
    const cancelled = a.action === "cancelled";
    items.push(
      <ArtifactRow
        key="appointment"
        icon={<CalendarCheck className="h-3.5 w-3.5" />}
        label={
          cancelled
            ? `Cancelled appointment (slot ${a.slot_id})`
            : `Booked ${a.doctor_name} (${a.specialty?.replace("_", " ")})`
        }
        meta={
          cancelled
            ? a.confirmation_no
            : `${formatAppointmentTime(a.start_time)} · ${a.confirmation_no}`
        }
      />,
    );
  }

  if (state.record_change) {
    items.push(
      <ArtifactRow
        key="record"
        icon={<FileText className="h-3.5 w-3.5" />}
        label={`Record ${state.record_change.operation}d`}
        meta={state.record_change.patient_id}
      />,
    );
  }

  // Agent-mode-only structured artifacts. Rendered as a single summary
  // row each — the detailed JSON lives in the tool_log if reviewers want it.
  if (state.schedule_results?.schedule) {
    const open = state.schedule_results.schedule.filter(
      (s) => !(s as { booked?: number | boolean }).booked,
    ).length;
    const total = state.schedule_results.schedule.length;
    const doctor =
      (state.schedule_results.doctor as { name?: string })?.name ?? "Doctor";
    items.push(
      <ArtifactRow
        key="schedule"
        icon={<CalendarCheck className="h-3.5 w-3.5" />}
        label={`${doctor}'s schedule`}
        meta={`${open} open / ${total} total slots`}
      />,
    );
  }

  if (state.bookings_results?.length) {
    items.push(
      <ArtifactRow
        key="bookings"
        icon={<CalendarCheck className="h-3.5 w-3.5" />}
        label="Upcoming appointments"
        meta={`${state.bookings_results.length} booking(s)`}
      />,
    );
  }

  if (state.audit_results?.length) {
    items.push(
      <ArtifactRow
        key="audit"
        icon={<FileText className="h-3.5 w-3.5" />}
        label="Audit events"
        meta={`${state.audit_results.length} PHI access event(s)`}
      />,
    );
  }

  if (state.sources?.length) {
    const realSources = state.sources.filter(
      (s) => (s as { source?: string }).source !== "stub",
    );
    items.push(
      <details key="sources" className="group text-xs">
        <summary className="flex cursor-pointer items-center gap-1.5 text-muted-foreground hover:text-foreground">
          <Search className="h-3.5 w-3.5" />
          {realSources.length || state.sources.length} sources
          {realSources.length === 0 && (
            <Badge variant="warning" className="ml-1 text-[9px]">stub</Badge>
          )}
        </summary>
        <ul className="mt-1.5 space-y-1 pl-5">
          {state.sources.map((s) => (
            <li key={s.index}>
              <a
                href={s.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                [{s.index}] {s.title || s.url}
              </a>
            </li>
          ))}
        </ul>
      </details>,
    );
  }

  if (state.error) {
    items.push(
      <div key="err" className="flex items-center gap-1.5 text-xs text-destructive">
        <AlertCircle className="h-3.5 w-3.5" />
        {state.error}
      </div>,
    );
  }

  if (!items.length) return null;

  return <div className="flex flex-col gap-1.5">{items}</div>;
}

function ArtifactRow({
  icon,
  label,
  meta,
}: {
  icon: React.ReactNode;
  label: string;
  meta: string | null | undefined;
}) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-success">{icon}</span>
      <span className="font-medium">{label}</span>
      {meta && <span className="font-mono text-muted-foreground">{meta}</span>}
    </div>
  );
}
