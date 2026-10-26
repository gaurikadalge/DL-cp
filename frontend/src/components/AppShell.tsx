"use client";

import React, { useState, useEffect } from "react";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";
import { api } from "@/lib/api";

interface AppShellProps {
  title: string;
  description?: string;
  children: React.ReactNode;
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

export function AppShell({
  title,
  description,
  children,
  onRefresh,
  isRefreshing,
}: AppShellProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [backendHealthy, setBackendHealthy] = useState(false);

  useEffect(() => {
    let mounted = true;

    async function checkBackend() {
      try {
        const health = await api.getHealth();
        if (mounted) {
          setBackendHealthy(health.status === "ok");
        }
      } catch {
        if (mounted) {
          setBackendHealthy(false);
        }
      }
    }

    checkBackend();
    const interval = setInterval(checkBackend, 10000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen bg-slate-50/60 dark:bg-slate-950 flex text-slate-900 dark:text-slate-100 font-sans">
      <Sidebar
        collapsed={collapsed}
        onToggle={() => setCollapsed(!collapsed)}
        backendHealthy={backendHealthy}
      />

      <div
        className={`flex-1 flex flex-col min-w-0 transition-all duration-300 ${
          collapsed ? "pl-20" : "pl-64"
        }`}
      >
        <Header
          title={title}
          description={description}
          onRefresh={onRefresh}
          isRefreshing={isRefreshing}
        />

        <main className="flex-1 p-6 max-w-7xl w-full mx-auto">{children}</main>
      </div>
    </div>
  );
}
