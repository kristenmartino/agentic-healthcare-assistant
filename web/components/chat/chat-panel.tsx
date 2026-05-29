"use client";
import * as React from "react";
import { ArrowUp, Loader2, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Message } from "@/components/chat/message";
import { StatusStrip } from "@/components/chat/status-strip";
import { useChatStore } from "@/lib/store";
import { chatStream } from "@/lib/chat";
import type { Patient } from "@/lib/api";

const EXAMPLE_QUERIES = [
  "Show me Anjali Mehra's medical history",
  "Book a cardiologist for next week",
  "What are the symptoms of pneumonia?",
  "My 70-year-old father has chronic kidney disease. Book a nephrologist and summarize the latest treatments.",
];

export function ChatPanel({ patients }: { patients: Patient[] }) {
  const [input, setInput] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const scrollRef = React.useRef<HTMLDivElement>(null);

  const messages = useChatStore((s) => s.messages);
  const threadId = useChatStore((s) => s.threadId);
  const selectedPatientId = useChatStore((s) => s.selectedPatientId);
  const appendUserMessage = useChatStore((s) => s.appendUserMessage);
  const appendAssistantMessage = useChatStore((s) => s.appendAssistantMessage);
  const appendAssistantToken = useChatStore((s) => s.appendAssistantToken);
  const recordStatusEvent = useChatStore((s) => s.recordStatusEvent);
  const finalizeAssistantMessage = useChatStore((s) => s.finalizeAssistantMessage);
  const markAssistantError = useChatStore((s) => s.markAssistantError);

  const selectedPatient = patients.find((p) => p.patient_id === selectedPatientId) || null;

  // Auto-scroll to bottom on every new message
  React.useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const send = React.useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || busy) return;
      setBusy(true);
      setInput("");

      // Snapshot the conversation BEFORE appending the new turn, so the
      // backend gets prior turns as memory. Read fresh from the store to
      // avoid a stale closure. Skip empty / streaming / error bubbles, and
      // cap to the last 16 messages (~8 turns) to bound prompt tokens.
      const history = useChatStore
        .getState()
        .messages.filter(
          (m) => !m.streaming && m.content.trim() && !m.content.startsWith("⚠️"),
        )
        .slice(-16)
        .map((m) => ({ role: m.role, content: m.content }));

      appendUserMessage(trimmed);
      const aId = appendAssistantMessage();

      try {
        await chatStream(
          {
            user_input: trimmed,
            thread_id: threadId,
            patient_id: selectedPatient?.patient_id ?? null,
            patient_name: selectedPatient?.name ?? null,
            history,
          },
          {
            onStatus: (e) => recordStatusEvent(aId, e),
            onToken: (token) => appendAssistantToken(aId, token),
            onDone: (state) => finalizeAssistantMessage(aId, state),
            onError: (msg) => markAssistantError(aId, msg),
          },
        );
      } catch (e) {
        markAssistantError(aId, (e as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [
      appendAssistantMessage,
      appendAssistantToken,
      appendUserMessage,
      busy,
      finalizeAssistantMessage,
      markAssistantError,
      recordStatusEvent,
      selectedPatient,
      threadId,
    ],
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    send(input);
  };

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Disclaimer banner */}
      <div className="border-b bg-warning/10 px-4 py-2 text-xs text-warning-foreground">
        <strong>Informational only</strong> — not a substitute for clinical care. Synthetic data; no real PHI.
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto flex max-w-3xl flex-col gap-5">
          {messages.length === 0 && <EmptyState onPick={send} />}
          {messages.map((m, i) => {
            const prev = messages[i - 1];
            const isFirstAssistant = m.role === "assistant" && prev?.role !== "assistant";
            return (
              <div key={m.id}>
                {m.role === "assistant" && m.statusEvents?.length ? (
                  <StatusStrip events={m.statusEvents} streaming={m.streaming} />
                ) : null}
                <Message message={m} />
                {isFirstAssistant && null}
              </div>
            );
          })}
        </div>
      </div>

      {/* Composer */}
      <div className="border-t bg-background px-4 py-3">
        <form
          onSubmit={handleSubmit}
          className="mx-auto flex max-w-3xl items-end gap-2"
        >
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder={
              selectedPatient
                ? `Ask about ${selectedPatient.name}, book an appointment, or look something up…`
                : "Ask a healthcare question, book an appointment, or look something up…"
            }
            rows={1}
            className="min-h-[40px] resize-none"
            disabled={busy}
          />
          <Button type="submit" size="icon" disabled={busy || !input.trim()}>
            {busy ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <ArrowUp className="h-4 w-4" />
            )}
          </Button>
        </form>
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="mx-auto mt-12 max-w-xl text-center">
      <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
        <Sparkles className="h-5 w-5" />
      </div>
      <h2 className="text-xl font-semibold">How can I help?</h2>
      <p className="mt-1 text-sm text-muted-foreground">
        Book appointments, look up patient history, or search trusted medical sources.
      </p>
      <div className="mt-5 grid gap-2 text-left text-sm">
        {EXAMPLE_QUERIES.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onPick(q)}
            className="rounded-md border bg-card px-4 py-3 transition-colors hover:border-primary/50 hover:bg-secondary/40"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
