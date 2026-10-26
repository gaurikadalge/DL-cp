"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  MessageSquareText,
  Files,
  BarChart3,
  Settings,
  ChevronLeft,
  ChevronRight,
  BookOpen,
} from "lucide-react";

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  backendHealthy: boolean;
}

export function Sidebar({ collapsed, onToggle, backendHealthy }: SidebarProps) {
  const pathname = usePathname();

  const navItems = [
    {
      name: "Dashboard",
      href: "/",
      icon: LayoutDashboard,
      description: "Overview & metrics",
    },
    {
      name: "Research Assistant",
      href: "/assistant",
      icon: MessageSquareText,
      description: "Ask questions with citations",
    },
    {
      name: "Document Library",
      href: "/documents",
      icon: Files,
      description: "Manage & upload PDFs",
    },
    {
      name: "Evaluation Dashboard",
      href: "/evaluation",
      icon: BarChart3,
      description: "BM25 vs FAISS vs RRF",
    },
    {
      name: "Settings & System",
      href: "/settings",
      icon: Settings,
      description: "API & dependency health",
    },
  ];

  return (
    <aside
      className={`fixed top-0 left-0 z-40 h-screen transition-all duration-300 flex flex-col bg-white dark:bg-slate-900 border-r border-slate-200/80 dark:border-slate-800 ${
        collapsed ? "w-20" : "w-64"
      }`}
    >
      {/* Brand Header */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-slate-100 dark:border-slate-800/80">
        {!collapsed && (
          <Link href="/" className="flex items-center gap-2.5 overflow-hidden">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20 shrink-0">
              <BookOpen className="w-4 h-4" />
            </div>
            <div className="min-w-0">
              <h1 className="font-bold text-sm text-slate-900 dark:text-slate-50 truncate tracking-tight">
                Multimodal RAG
              </h1>
              <p className="text-[11px] text-slate-400 dark:text-slate-500 font-mono truncate">
                Research Assistant
              </p>
            </div>
          </Link>
        )}

        {collapsed && (
          <div className="mx-auto w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
            <BookOpen className="w-4 h-4" />
          </div>
        )}

        <button
          type="button"
          onClick={onToggle}
          className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          title={collapsed ? "Expand Sidebar" : "Collapse Sidebar"}
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Navigation List */}
      <nav className="flex-1 px-3 py-4 space-y-1.5 overflow-y-auto">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-150 group relative ${
                isActive
                  ? "bg-blue-600 text-white font-medium shadow-sm shadow-blue-600/20"
                  : "text-slate-600 dark:text-slate-400 hover:bg-slate-100/80 dark:hover:bg-slate-800/80 hover:text-slate-900 dark:hover:text-slate-100"
              }`}
            >
              <Icon
                className={`w-5 h-5 shrink-0 transition-transform group-hover:scale-105 ${
                  isActive ? "text-white" : "text-slate-400 dark:text-slate-500 group-hover:text-slate-700 dark:group-hover:text-slate-300"
                }`}
              />

              {!collapsed && (
                <div className="min-w-0">
                  <div className="text-xs font-semibold tracking-wide truncate">
                    {item.name}
                  </div>
                  <div
                    className={`text-[10px] truncate ${
                      isActive ? "text-blue-100" : "text-slate-400 dark:text-slate-500"
                    }`}
                  >
                    {item.description}
                  </div>
                </div>
              )}

              {/* Collapsed Tooltip */}
              {collapsed && (
                <div className="absolute left-full ml-2 px-2.5 py-1.5 rounded-lg bg-slate-900 text-white text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity shadow-lg z-50">
                  {item.name}
                </div>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Backend Status Footer */}
      <div className="p-3 border-t border-slate-100 dark:border-slate-800/80">
        <div
          className={`p-2.5 rounded-xl border flex items-center gap-2.5 transition-colors ${
            backendHealthy
              ? "bg-emerald-500/5 border-emerald-500/20 text-emerald-700 dark:text-emerald-400"
              : "bg-rose-500/5 border-rose-500/20 text-rose-700 dark:text-rose-400"
          }`}
        >
          <div
            className={`w-2 h-2 rounded-full shrink-0 ${
              backendHealthy ? "bg-emerald-500 animate-pulse" : "bg-rose-500"
            }`}
          />
          {!collapsed && (
            <div className="min-w-0 flex-1">
              <p className="text-[11px] font-semibold truncate leading-tight">
                {backendHealthy ? "Backend Connected" : "Backend Offline"}
              </p>
              <p className="text-[10px] text-slate-400 dark:text-slate-500 truncate font-mono">
                :8000
              </p>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
