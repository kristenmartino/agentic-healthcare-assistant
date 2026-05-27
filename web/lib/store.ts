/**
 * Client-side state for the chat page.
 *
 * Zustand because:
 *  - Multiple components (Sidebar, ChatInput, ResponsePanel) all touch
 *    `selectedPatientId` and `thread_id`. Prop-drilling those gets ugly.
 *  - We don't want every chat-history append to re-render the patient
 *    picker — Zustand's per-slice subscriptions are cleaner than Context
 *    for that.
 *
 * NOT persisted to localStorage — the audit log lives server-side; the
 * chat history is per-tab on purpose for the shared demo URL.
 */
import { create } from "zustand";
import type { DoneEvent, StatusEvent } from "@/lib/chat";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  state?: DoneEvent | null;          // populated on assistant messages after done
  statusEvents?: StatusEvent[];      // node-progress trail for assistant messages
  streaming?: boolean;
};

type ChatStore = {
  selectedPatientId: string | null;
  threadId: string;
  messages: ChatMessage[];

  // patient
  setSelectedPatientId: (id: string | null) => void;

  // thread lifecycle
  resetThread: () => void;

  // message lifecycle
  appendUserMessage: (text: string) => string;
  appendAssistantMessage: () => string;
  appendAssistantToken: (id: string, token: string) => void;
  recordStatusEvent: (id: string, e: StatusEvent) => void;
  finalizeAssistantMessage: (id: string, state: DoneEvent) => void;
  setAssistantContent: (id: string, content: string) => void;
  markAssistantError: (id: string, error: string) => void;
};

function nextThreadId(): string {
  return `thread-${Math.random().toString(36).slice(2, 10)}-${Date.now()}`;
}

function newMessageId(): string {
  return `m-${Math.random().toString(36).slice(2, 10)}-${Date.now()}`;
}

export const useChatStore = create<ChatStore>((set) => ({
  selectedPatientId: null,
  threadId: nextThreadId(),
  messages: [],

  setSelectedPatientId: (id) => set({ selectedPatientId: id }),

  resetThread: () => set({ threadId: nextThreadId(), messages: [] }),

  appendUserMessage: (text) => {
    const id = newMessageId();
    set((s) => ({
      messages: [...s.messages, { id, role: "user", content: text }],
    }));
    return id;
  },

  appendAssistantMessage: () => {
    const id = newMessageId();
    set((s) => ({
      messages: [
        ...s.messages,
        {
          id,
          role: "assistant",
          content: "",
          streaming: true,
          statusEvents: [],
        },
      ],
    }));
    return id;
  },

  appendAssistantToken: (id, token) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, content: m.content + token } : m,
      ),
    })),

  recordStatusEvent: (id, e) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id
          ? { ...m, statusEvents: [...(m.statusEvents || []), e] }
          : m,
      ),
    })),

  finalizeAssistantMessage: (id, state) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id
          ? {
              ...m,
              streaming: false,
              state,
              // Prefer streamed content if any; fall back to the done event's
              // response field.
              content: m.content || state.response || "",
            }
          : m,
      ),
    })),

  setAssistantContent: (id, content) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, content } : m,
      ),
    })),

  markAssistantError: (id, error) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id
          ? { ...m, streaming: false, content: `⚠️ ${error}` }
          : m,
      ),
    })),
}));
