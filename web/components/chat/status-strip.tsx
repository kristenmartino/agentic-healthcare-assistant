"use client";
import { motion } from "framer-motion";
import { CheckCircle2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { StatusEvent } from "@/lib/chat";

/**
 * Per-message status strip — the node-progress trail shown above the
 * assistant's response while the workflow is running, and collapsed to
 * a one-line summary after it completes.
 */
export function StatusStrip({
  events,
  streaming,
}: {
  events: StatusEvent[];
  streaming?: boolean;
}) {
  if (!events.length) return null;
  return (
    <div className="ml-11 mb-1 flex flex-wrap items-center gap-1.5">
      {events.map((e, i) => (
        <motion.div
          key={`${e.node}-${i}`}
          initial={{ opacity: 0, y: -2 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.15 }}
        >
          <Badge variant="secondary" className="gap-1 font-normal">
            <CheckCircle2 className="h-3 w-3 text-success" />
            <span className="text-[11px]">{e.label}</span>
            {e.summary && (
              <span className="text-[11px] text-muted-foreground">
                — {e.summary}
              </span>
            )}
          </Badge>
        </motion.div>
      ))}
      {streaming && (
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-[11px] text-muted-foreground"
        >
          …working
        </motion.span>
      )}
    </div>
  );
}
