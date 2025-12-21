import type { Metadata } from "next";

import "@/styles/globals.css";
import { Sidebar } from "@/components/navigation/sidebar";

export const metadata: Metadata = {
  title: "AI自动写小说系统",
  description: "AI驱动的小说自动生成系统"
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh">
      <body className="min-h-screen">
        <div className="flex min-h-screen">
          <Sidebar />
          <main className="flex-1 bg-background p-8">
            <div className="mx-auto max-w-6xl space-y-8">{children}</div>
          </main>
        </div>
      </body>
    </html>
  );
}
