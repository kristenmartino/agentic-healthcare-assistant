import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";

import { AppShell } from "@/components/app-shell";
import { ThemeProvider } from "@/components/theme-provider";
import "./globals.css";

// Geist is the visual target but ships as its own npm package; Inter +
// JetBrains Mono via next/font is one less dep and visually adjacent.
// CSS variables are referenced from tailwind.config.ts (font-sans / font-mono).
const fontSans = Inter({
  subsets: ["latin"],
  variable: "--font-geist-sans",
  display: "swap",
});
const fontMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Healthcare Assistant",
  description:
    "LangGraph-backed agentic healthcare assistant — FHIR-grounded patient lookups, PHI audit log, clinical safety triage, MCP server.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${fontSans.variable} ${fontMono.variable}`}
    >
      <body className="min-h-screen bg-background font-sans antialiased">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {/* AppShell lives in the root layout so it mounts ONCE and persists
              across route changes — navigating to /audit, /traces, /about and
              back no longer remounts it (which used to refetch patients/config
              and flicker the patient selection). */}
          <AppShell>{children}</AppShell>
        </ThemeProvider>
      </body>
    </html>
  );
}
