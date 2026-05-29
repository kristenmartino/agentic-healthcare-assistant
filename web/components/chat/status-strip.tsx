"use client";
import { motion } from "framer-motion";
import { CheckCircle2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { StatusEvent } from "@/lib/chat";

/**
 * Per-message status strip — the trail of completed workflow steps shown
 * above the assistant's response. Each badge is a node that has finished
 * (the backend emits a step as the node completes). The live "still
 * working" signal is owned by the bubble's ThinkingIndicator (which shows
 * a ticking elapsed timer), so the strip stays a clean done-step trail.
 */
export function StatusStrip({ events }: { events: StatusEvent[] }) {
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
    </div>
  );
}
