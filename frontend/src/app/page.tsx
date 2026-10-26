"use client";

import React, { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { AppShell } from "@/components/AppShell";
import { MetricCard } from "@/components/MetricCard";
import { StatusBadge } from "@/components/StatusBadge";
import { api } from "@/lib/api";
import { DocumentMetadata, HealthCheckResponse } from "@/types/api";
import { formatDate } from "@/lib/utils";
import {
  Files,
  Layers,
  Sparkles,
  Zap,
  ArrowRight,
  Search,
  Server,
  BookOpen,
} from "lucide-react";

export default function DashboardPage() {
  const router = useRouter();
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [health, setHealth] = useState<HealthCheckResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [queryInput, setQueryInput] = useState("");

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [docsData, healthData] = await Promise.allSettled([
        api.listDocuments(0, 10),
        api.getHealth(),
      ]);

      if (docsData.status === "fulfilled") {
        setDocuments(docsData.value);
      }
      if (healthData.status === "fulfilled") {
        setHealth(healthData.value);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    Promise.allSettled([api.listDocuments(0, 10), api.getHealth()]).then(
      ([docsData, healthData]) => {
        if (!active) return;
        if (docsData.status === "fulfilled") {
          setDocuments(docsData.value);
        }
        if (healthData.status === "fulfilled") {
          setHealth(healthData.value);
        }
        setLoading(false);
      }
    );
    return () => {
      active = false;
    };
  }, []);

  const totalChunks = documents.reduce((acc, doc) => acc + (doc.chunk_count || 0), 0);
  const totalPages = documents.reduce((acc, doc) => acc + (doc.page_count || 0), 0);

  const handleQuickQuery = (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryInput.trim()) return;
    router.push(`/assistant?q=${encodeURIComponent(queryInput.trim())}`);
  };

  return (
    <AppShell
      title="Research Assistant Dashboard"
      description="Multimodal PDF ingestion, hybrid FAISS + BM25 retrieval, and grounded citation synthesis"
      onRefresh={fetchData}
      isRefreshing={loading}
    >
      <div className="space-y-6">
        {/* Quick Question Bar */}
        <div className="relative p-6 rounded-3xl bg-gradient-to-r from-blue-600 via-indigo-600 to-violet-600 text-white shadow-xl shadow-blue-500/10 overflow-hidden">
          <div className="absolute right-0 top-0 w-96 h-96 bg-white/10 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20" />

          <div className="relative z-10 max-w-2xl">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-white/20 backdrop-blur-md mb-3 text-white border border-white/20">
              <Sparkles className="w-3.5 h-3.5" />
              Grounded Question Answering
            </span>
            <h3 className="text-xl sm:text-2xl font-bold tracking-tight mb-2">
              Ask anything across your academic paper library
            </h3>
            <p className="text-blue-100 text-xs sm:text-sm mb-5 leading-relaxed">
              Synthesize factual answers grounded in dense vector search and BM25 keywords with verifiable page citations.
            </p>

            <form onSubmit={handleQuickQuery} className="flex gap-2">
              <div className="relative flex-1">
                <Search className="w-4 h-4 absolute left-3.5 top-3.5 text-slate-400" />
                <input
                  type="text"
                  value={queryInput}
                  onChange={(e) => setQueryInput(e.target.value)}
                  placeholder="e.g. How does Reciprocal Rank Fusion combine rankings?"
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-white text-slate-900 placeholder:text-slate-400 text-xs sm:text-sm focus:outline-hidden focus:ring-2 focus:ring-white/80 shadow-md font-medium"
                />
              </div>
              <button
                type="submit"
                className="px-5 py-2.5 rounded-xl bg-slate-900 text-white hover:bg-slate-800 text-xs sm:text-sm font-semibold transition-all shadow-md shrink-0 flex items-center gap-1.5"
              >
                <span>Ask</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </form>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Indexed Papers"
            value={documents.length}
            subtitle={`${totalPages} pages parsed & structured`}
            icon={Files}
            badge="SQLite Store"
            badgeColor="blue"
          />
          <MetricCard
            title="Evidence Chunks"
            value={totalChunks}
            subtitle="Indexed in FAISS & BM25"
            icon={Layers}
            badge="Cosine + Sparse"
            badgeColor="green"
          />
          <MetricCard
            title="Hybrid RRF Recall@5"
            value="0.9321"
            subtitle="Benchmark on 35 academic queries"
            icon={Zap}
            badge="+ Reranker"
            badgeColor="purple"
          />
          <MetricCard
            title="System Status"
            value={health?.status === "ok" ? "Operational" : "Offline"}
            subtitle={health?.app_name || "FastAPI Backend"}
            icon={Server}
            badge={health?.version || "v0.1.0"}
            badgeColor={health?.status === "ok" ? "green" : "amber"}
          />
        </div>

        {/* Two-Column Overview */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Ingested Documents */}
          <div className="lg:col-span-2 p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 shadow-xs">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  Document Library
                </h4>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Recently processed and indexed research publications
                </p>
              </div>
              <Link
                href="/documents"
                className="text-xs font-semibold text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-1"
              >
                <span>Manage Library</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>

            {loading ? (
              <div className="py-8 text-center text-xs text-slate-400">
                Loading indexed documents...
              </div>
            ) : documents.length === 0 ? (
              <div className="py-12 text-center border-2 border-dashed rounded-xl border-slate-200 dark:border-slate-800">
                <BookOpen className="w-8 h-8 mx-auto text-slate-300 dark:text-slate-600 mb-2" />
                <p className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                  No research papers indexed yet
                </p>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1 mb-3">
                  Upload an academic PDF to start answering questions with page citations.
                </p>
                <Link
                  href="/documents"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600 text-white text-xs font-semibold shadow-xs hover:bg-blue-700 transition-colors"
                >
                  Upload First Paper
                </Link>
              </div>
            ) : (
              <div className="divide-y divide-slate-100 dark:divide-slate-800 overflow-x-auto">
                {documents.map((doc) => (
                  <div
                    key={doc.document_id}
                    className="py-3 flex items-center justify-between gap-3 text-xs"
                  >
                    <div className="min-w-0 flex items-center gap-2.5">
                      <div className="w-8 h-8 rounded-lg bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400 flex items-center justify-center shrink-0">
                        <Files className="w-4 h-4" />
                      </div>
                      <div className="min-w-0">
                        <p className="font-semibold text-slate-800 dark:text-slate-200 truncate">
                          {doc.filename}
                        </p>
                        <p className="text-[11px] text-slate-400 font-mono">
                          {doc.page_count ?? 1} pages • {doc.chunk_count ?? 0} chunks •{" "}
                          {formatDate(doc.created_at)}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
                      <StatusBadge status={doc.status} />
                      <Link
                        href={`/assistant?doc=${doc.document_id}`}
                        className="px-2.5 py-1 rounded-lg border border-slate-200 dark:border-slate-800 text-[11px] font-medium hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 transition-colors"
                      >
                        Query
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Subsystem Topology Card */}
          <div className="p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 shadow-xs space-y-4">
            <div>
              <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                Retrieval Pipeline
              </h4>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Active pipeline architecture & models
              </p>
            </div>

            <div className="space-y-2.5 text-xs">
              <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800 flex items-center justify-between">
                <span className="font-medium text-slate-700 dark:text-slate-300">Dense Channel</span>
                <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400 font-semibold">
                  FAISS IndexFlatIP (384-d)
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800 flex items-center justify-between">
                <span className="font-medium text-slate-700 dark:text-slate-300">Sparse Channel</span>
                <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400 font-semibold">
                  Okapi BM25 (k1=1.5, b=0.75)
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800 flex items-center justify-between">
                <span className="font-medium text-slate-700 dark:text-slate-300">Fusion Strategy</span>
                <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-purple-50 dark:bg-purple-950/60 text-purple-600 dark:text-purple-400 font-semibold">
                  RRF (k=60)
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800 flex items-center justify-between">
                <span className="font-medium text-slate-700 dark:text-slate-300">Reranker</span>
                <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 font-semibold">
                  ms-marco-MiniLM-L6
                </span>
              </div>
            </div>

            <div className="pt-2 border-t border-slate-100 dark:border-slate-800">
              <Link
                href="/evaluation"
                className="w-full py-2 px-3 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 font-semibold text-xs flex items-center justify-center gap-1.5 transition-colors"
              >
                <span>View 35-Query Benchmark</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
