"use client";
import { useShell } from "@/components/app-shell";
import { ChatPanel } from "@/components/chat/chat-panel";

export default function HomePage() {
  const { patients } = useShell();
  return <ChatPanel patients={patients} />;
}
