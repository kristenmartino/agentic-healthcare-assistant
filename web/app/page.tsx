"use client";
import { AppShell, useShell } from "@/components/app-shell";
import { ChatPanel } from "@/components/chat/chat-panel";

export default function HomePage() {
  return (
    <AppShell>
      <ChatPanelWithShellData />
    </AppShell>
  );
}

function ChatPanelWithShellData() {
  const { patients } = useShell();
  return <ChatPanel patients={patients} />;
}
