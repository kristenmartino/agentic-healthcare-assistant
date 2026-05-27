"use client";
import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Eye, MessagesSquare, ShieldCheck, Stethoscope } from "lucide-react";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { ThemeToggle } from "@/components/theme-toggle";
import { useChatStore } from "@/lib/store";
import { cn, formatTimestamp } from "@/lib/utils";
import type { Config, Patient } from "@/lib/api";

const navItems = [
  { href: "/", label: "Chat", icon: MessagesSquare },
  { href: "/audit", label: "Audit Log", icon: ShieldCheck },
  { href: "/traces", label: "Traces", icon: Activity },
  { href: "/about", label: "About", icon: Eye },
];

type SidebarProps = {
  config: Config | null;
  patients: Patient[];
};

export function Sidebar({ config, patients }: SidebarProps) {
  const pathname = usePathname();
  const selectedPatientId = useChatStore((s) => s.selectedPatientId);
  const setSelectedPatientId = useChatStore((s) => s.setSelectedPatientId);
  const resetThread = useChatStore((s) => s.resetThread);

  return (
    <aside className="hidden h-screen w-72 shrink-0 flex-col border-r bg-card lg:flex">
      <div className="flex h-14 items-center gap-2 border-b px-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
          <Stethoscope className="h-4 w-4" />
        </div>
        <div className="flex-1">
          <div className="text-sm font-semibold leading-tight">Healthcare</div>
          <div className="text-xs text-muted-foreground leading-tight">Assistant</div>
        </div>
        <ThemeToggle />
      </div>

      <ScrollArea className="flex-1">
        <div className="p-3">
          <nav className="space-y-1">
            {navItems.map((item) => {
              const active = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                    active
                      ? "bg-secondary text-secondary-foreground"
                      : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground",
                  )}
                >
                  <item.icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>

        <Separator />

        <div className="p-3">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Patient
            </h3>
            <Button
              size="sm"
              variant="ghost"
              className="h-6 px-2 text-xs"
              onClick={() => resetThread()}
            >
              New thread
            </Button>
          </div>
          <button
            onClick={() => {
              setSelectedPatientId(null);
              resetThread();
            }}
            className={cn(
              "mb-1 flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors",
              !selectedPatientId
                ? "bg-secondary text-secondary-foreground"
                : "hover:bg-secondary/60",
            )}
          >
            <Avatar className="h-7 w-7">
              <AvatarFallback className="bg-muted text-[10px]">--</AvatarFallback>
            </Avatar>
            <div className="flex-1">
              <div className="text-sm font-medium leading-tight">Walk-in</div>
              <div className="text-xs text-muted-foreground leading-tight">
                no patient context
              </div>
            </div>
          </button>
          {patients.map((p) => {
            const active = selectedPatientId === p.patient_id;
            const initials = p.name
              .split(" ")
              .slice(0, 2)
              .map((n) => n[0])
              .join("")
              .toUpperCase();
            return (
              <button
                key={p.patient_id}
                onClick={() => {
                  setSelectedPatientId(p.patient_id);
                  resetThread();
                }}
                className={cn(
                  "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors",
                  active
                    ? "bg-secondary text-secondary-foreground"
                    : "hover:bg-secondary/60",
                )}
              >
                <Avatar className="h-7 w-7">
                  <AvatarFallback className="bg-primary/10 text-[10px] text-primary">
                    {initials}
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1 overflow-hidden">
                  <div className="text-sm font-medium leading-tight truncate">
                    {p.name}
                  </div>
                  <div className="text-xs text-muted-foreground leading-tight truncate">
                    {p.summary || `${p.age ?? "?"} · ${p.gender ?? "?"}`}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </ScrollArea>

      {config && (
        <div className="border-t p-3 space-y-1.5">
          <StatusRow
            label="LLM"
            value={`${config.llm_provider}${config.llm_provider !== "stub" ? "" : ""}`}
            variant={config.llm_provider === "stub" ? "warning" : "default"}
          />
          <StatusRow
            label="EHR"
            value={config.ehr_backend}
            variant="secondary"
          />
          <StatusRow
            label="Search"
            value={config.search_backend_intended}
            variant={config.tavily_configured ? "default" : "secondary"}
          />
          {config.langsmith_enabled && (
            <StatusRow label="Tracing" value="LangSmith" variant="success" />
          )}
          <div className="pt-2 text-[10px] text-muted-foreground">
            Built {formatTimestamp(new Date().toISOString()).split(",")[0]}
          </div>
        </div>
      )}
    </aside>
  );
}

function StatusRow({
  label,
  value,
  variant,
}: {
  label: string;
  value: string;
  variant: "default" | "secondary" | "success" | "warning";
}) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-muted-foreground">{label}</span>
      <Badge variant={variant} className="font-mono text-[10px]">
        {value}
      </Badge>
    </div>
  );
}
