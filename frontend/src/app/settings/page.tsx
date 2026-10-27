"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import {
  Server,
  Activity,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ExternalLink,
  ShieldCheck,
  Terminal,
  Cpu,
  Layers,
  Database,
  Search,
  BookOpen,
} from "lucide-react";
import { api } from "@/lib/api";
import { HealthCheckResponse, AppInfo } from "@/types/api";
import { StatusBadge } from "@/components/StatusBadge";
import { AppShell } from "@/components/AppShell";

interface DependencyDetail {
  name: string;
  category: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
}

const DEPENDENCY_INFO: Record<string, DependencyDetail> = {
  database: {
    name: "SQLite Database",
    category: "Persistence",
    description: "ACID-compliant storage for document records, chunks, and metadata.",
    icon: Database,
  },
  vector_index: {
    name: "FAISS Vector Index",
    category: "Dense Retrieval",
    description: "In-memory IndexFlatIP storing normalized 384-d semantic embeddings.",
    icon: Layers,
  },
  keyword_retrieval: {
    name: "BM25 Keyword Index",
    category: "Lexical Retrieval",
    description: "BM25Okapi sparse index with Porter stemming for exact lexical matching.",
    icon: Search,
  },
  llm_provider: {
    name: "LLM Generation Provider",
    category: "Answer Synthesis",
    description: "Grounded prompt generation using configured provider (OpenAI / Mock fallback).",
    icon: Cpu,
  },
  ocr_engine: {
    name: "Tesseract OCR Engine",
    category: "Multimodal Ingestion",
    description: "Optical character recognition for scanned PDFs and diagram text recovery.",
    icon: BookOpen,
  },
  reranker: {
    name: "Cross-Encoder Reranker",
    category: "Post-Retrieval",
    description: "ms-marco-MiniLM-L-6-v2 cross-encoder scoring retrieved candidates.",
    icon: ShieldCheck,
  },
};

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthCheckResponse | null>(null);
  const [appInfo, setAppInfo] = useState<AppInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [pingLatency, setPingLatency] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const customApiUrl = api.getBaseUrl();

  const apiPort = useMemo(() => {
    try {
      const url = new URL(customApiUrl);
      return url.port || "8001";
    } catch {
      return "8001";
    }
  }, [customApiUrl]);

  const testConnection = useCallback(async () => {
    setLoading(true);
    setError(null);
    const start = performance.now();
    try {
      const [healthData, infoData] = await Promise.all([
        api.getHealth(),
        api.getAppInfo().catch(() => null),
      ]);
      const end = performance.now();
      setPingLatency(Math.round(end - start));
      setHealth(healthData);
      setAppInfo(infoData);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to connect to backend server";
      setError(msg);
      setHealth(null);
      setPingLatency(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    const start = performance.now();
    Promise.all([api.getHealth(), api.getAppInfo().catch(() => null)])
      .then(([healthData, infoData]) => {
        if (!active) return;
        const end = performance.now();
        setPingLatency(Math.round(end - start));
        setHealth(healthData);
        setAppInfo(infoData);
      })
      .catch((err: unknown) => {
        if (!active) return;
        const msg = err instanceof Error ? err.message : "Failed to connect to backend server";
        setError(msg);
        setHealth(null);
        setPingLatency(null);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  return (
    <AppShell
      title="System & Pipeline Settings"
      description="Validate backend connectivity, monitor subsystem health, and review RAG configuration"
      onRefresh={testConnection}
      isRefreshing={loading}
    >
      <div className="space-y-8 max-w-6xl">
        {/* Connectivity Banner */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-5 rounded-2xl border border-slate-200/80 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xs flex items-center gap-4">
            <div
              className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${
                health?.status === "ok"
                  ? "bg-emerald-500/10 text-emerald-500"
                  : health?.status === "degraded"
                  ? "bg-amber-500/10 text-amber-500"
                  : "bg-red-500/10 text-red-500"
              }`}
            >
              <Server className="w-6 h-6" />
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                API Status
              </div>
              <div className="flex items-center gap-2 mt-1">
                <span className="font-bold text-lg text-slate-900 dark:text-slate-100 capitalize">
                  {health ? health.status : "Offline / Unreachable"}
                </span>
                {health && <StatusBadge status={health.status} />}
              </div>
            </div>
          </div>

          <div className="p-5 rounded-2xl border border-slate-200/80 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xs flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-blue-500/10 text-blue-500 flex items-center justify-center shrink-0">
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Ping Latency
              </div>
              <div className="text-lg font-bold text-slate-900 dark:text-slate-100 mt-1">
                {pingLatency !== null ? `${pingLatency} ms` : "N/A"}
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400 truncate">
                Round-trip to {customApiUrl}
              </div>
            </div>
          </div>

          <div className="p-5 rounded-2xl border border-slate-200/80 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xs flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-indigo-500/10 text-indigo-500 flex items-center justify-center shrink-0">
              <Terminal className="w-6 h-6" />
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Backend Version
              </div>
              <div className="text-lg font-bold text-slate-900 dark:text-slate-100 mt-1">
                {appInfo?.version || health?.version || "0.1.0"}
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400 capitalize">
                {appInfo?.environment || health?.environment || "Development"} Environment
              </div>
            </div>
          </div>
        </div>

        {/* Error Alert if Offline */}
        {error && (
          <div className="p-4 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-600 dark:text-rose-400 flex items-start gap-3">
            <XCircle className="w-5 h-5 shrink-0 mt-0.5" />
            <div className="text-sm">
              <div className="font-semibold">Backend Unreachable</div>
              <div className="mt-0.5 text-slate-600 dark:text-slate-400">{error}</div>
              <div className="mt-2 text-xs font-mono bg-white dark:bg-slate-900 p-2 rounded border border-slate-200 dark:border-slate-800 text-slate-800 dark:text-slate-200">
                uvicorn app.main:app --host 127.0.0.1 --port {apiPort}
              </div>
            </div>
          </div>
        )}

        {/* Pipeline Dependencies Grid */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">
                Pipeline Subsystem Status
              </h2>
              <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400">
                Live diagnostics reported by the FastAPI backend health monitor (<code className="font-mono text-blue-600 dark:text-blue-400">GET /health</code>).
              </p>
            </div>
            <span className="text-xs font-mono text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 px-2.5 py-1 rounded-lg">
              Last Checked: {health ? new Date(health.timestamp).toLocaleTimeString() : "Never"}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(DEPENDENCY_INFO).map(([depKey, info]) => {
              const rawStatus = health?.dependencies?.[depKey] || (health ? "not_configured" : "unavailable");
              const IconComponent = info.icon;

              const isHealthy =
                rawStatus === "configured" || rawStatus === "available";
              const isDegraded =
                rawStatus === "not_loaded_scaffold" || rawStatus === "not_configured";

              return (
                <div
                  key={depKey}
                  className="p-5 rounded-2xl border border-slate-200/80 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xs hover:border-blue-500/40 transition flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-start justify-between gap-3 mb-3">
                      <div className="w-10 h-10 rounded-xl bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-slate-700 dark:text-slate-300 shrink-0">
                        <IconComponent className="w-5 h-5" />
                      </div>
                      <span
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
                          isHealthy
                            ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20"
                            : isDegraded
                            ? "bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20"
                            : "bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20"
                        }`}
                      >
                        {isHealthy ? (
                          <CheckCircle2 className="w-3.5 h-3.5" />
                        ) : isDegraded ? (
                          <AlertTriangle className="w-3.5 h-3.5" />
                        ) : (
                          <XCircle className="w-3.5 h-3.5" />
                        )}
                        <span className="capitalize">{rawStatus.replace(/_/g, " ")}</span>
                      </span>
                    </div>

                    <div className="text-xs font-semibold text-blue-600 dark:text-blue-400 uppercase tracking-wider">
                      {info.category}
                    </div>
                    <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mt-0.5">
                      {info.name}
                    </h3>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1.5 leading-relaxed">
                      {info.description}
                    </p>
                  </div>

                  <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800/80 flex items-center justify-between text-xs text-slate-400 dark:text-slate-500 font-mono">
                    <span>key: {depKey}</span>
                    <span className="text-[11px] capitalize">{rawStatus}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Configuration & Environment Details */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Connection Configuration */}
          <div className="p-6 rounded-2xl border border-slate-200/80 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xs space-y-4">
            <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Server className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              Backend Connection Config
            </h2>
            <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400">
              The frontend communicates with FastAPI via REST calls using typed contracts.
            </p>

            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium text-slate-600 dark:text-slate-300 uppercase tracking-wider block mb-1">
                  Active API Base URL
                </label>
                <input
                  type="text"
                  value={customApiUrl}
                  readOnly
                  className="w-full px-3.5 py-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 font-mono text-xs text-slate-900 dark:text-slate-100 cursor-not-allowed"
                />
                <span className="text-[11px] text-slate-400 dark:text-slate-500 mt-1 block">
                  Configured via <code className="text-blue-600 dark:text-blue-400 font-mono">NEXT_PUBLIC_API_BASE_URL</code> in <code className="text-slate-700 dark:text-slate-300 font-mono">frontend/.env.local</code>.
                </span>
              </div>

              <div className="pt-2 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between text-sm">
                <span className="text-slate-500 dark:text-slate-400 text-xs sm:text-sm">Interactive Swagger Docs</span>
                <a
                  href={`${api.getBaseUrl()}/docs`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 text-blue-600 dark:text-blue-400 hover:underline text-xs font-medium"
                >
                  /docs <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </div>

              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-500 dark:text-slate-400 text-xs sm:text-sm">Redoc API Specification</span>
                <a
                  href={`${api.getBaseUrl()}/redoc`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 text-blue-600 dark:text-blue-400 hover:underline text-xs font-medium"
                >
                  /redoc <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </div>

              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-500 dark:text-slate-400 text-xs sm:text-sm">Raw Health Endpoint</span>
                <a
                  href={`${api.getBaseUrl()}/health`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 text-blue-600 dark:text-blue-400 hover:underline text-xs font-medium"
                >
                  /health <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </div>
            </div>
          </div>

          {/* Environment Guidelines */}
          <div className="p-6 rounded-2xl border border-slate-200/80 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xs space-y-4">
            <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Terminal className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              Environment Setup Guidelines
            </h2>
            <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400">
              Configure system variables in the project root <code className="text-slate-800 dark:text-slate-200 font-mono">.env</code> file:
            </p>

            <div className="space-y-3 font-mono text-xs">
              <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200/80 dark:border-slate-800 space-y-1.5">
                <div className="text-slate-900 dark:text-slate-100 font-semibold"># LLM Provider</div>
                <div className="text-slate-500 dark:text-slate-400">OPENAI_API_KEY=sk-...</div>
                <div className="text-slate-500 dark:text-slate-400">LLM_PROVIDER=openai # or mock</div>
              </div>

              <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200/80 dark:border-slate-800 space-y-1.5">
                <div className="text-slate-900 dark:text-slate-100 font-semibold"># Embedding & Reranker Models</div>
                <div className="text-slate-500 dark:text-slate-400">EMBEDDING_MODEL=all-MiniLM-L6-v2</div>
                <div className="text-slate-500 dark:text-slate-400">RERANKER_MODEL=ms-marco-MiniLM-L-6-v2</div>
              </div>

              <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200/80 dark:border-slate-800 space-y-1.5">
                <div className="text-slate-900 dark:text-slate-100 font-semibold"># Backend Server Port</div>
                <div className="text-slate-500 dark:text-slate-400">PORT={apiPort}</div>
                <div className="text-slate-500 dark:text-slate-400">HOST=127.0.0.1</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
