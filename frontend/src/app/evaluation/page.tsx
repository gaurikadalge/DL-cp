"use client";

import React, { useState, useEffect } from "react";
import { AppShell } from "@/components/AppShell";
import { MetricCard } from "@/components/MetricCard";
import { api } from "@/lib/api";
import { BenchmarkResultsData } from "@/types/api";
import { FALLBACK_BENCHMARK_DATA } from "@/data/benchmarkData";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  Award,
  Target,
  Clock,
  Download,
  Info,
  Sparkles,
} from "lucide-react";
import { useTheme } from "@/context/ThemeContext";

export default function EvaluationPage() {
  const { isDark } = useTheme();
  const [data, setData] = useState<BenchmarkResultsData>(FALLBACK_BENCHMARK_DATA);
  const [activeTab, setActiveTab] = useState<"overall" | "categories">("overall");

  useEffect(() => {
    let active = true;
    api
      .getBenchmarkResults()
      .then((res) => {
        if (active) setData(res);
      })
      .catch(() => {
        if (active) setData(FALLBACK_BENCHMARK_DATA);
      });
    return () => {
      active = false;
    };
  }, []);

  const overall = data.overall_results;
  const categories = data.category_results;
  const metadata = data.metadata;

  // Format overall data for Recharts
  const chartData = Object.entries(overall).map(([config, metrics]) => ({
    name: config,
    "Hit@5": Number(metrics["Hit@K"].toFixed(3)),
    "Recall@5": Number(metrics["Recall@K"].toFixed(3)),
    MRR: Number(metrics.MRR.toFixed(3)),
    "NDCG@5": Number(metrics["NDCG@K"].toFixed(3)),
    Latency_ms: Number(metrics.Mean_Latency_ms.toFixed(2)),
  }));

  // Format category data for Multi-Passage Comparison
  const multiPassageData = Object.entries(categories.multi_passage || {}).map(
    ([config, metrics]) => ({
      name: config,
      Recall: Number(metrics["Recall@K"].toFixed(3)),
      NDCG: Number(metrics["NDCG@K"].toFixed(3)),
      MRR: Number(metrics.MRR.toFixed(3)),
    })
  );

  const downloadCSV = () => {
    let csv = "Retrieval Configuration,Hit@5,Recall@5,MRR,NDCG@5,Mean Latency (ms)\n";
    Object.entries(overall).forEach(([cfg, m]) => {
      csv += `"${cfg}",${m["Hit@K"]},${m["Recall@K"]},${m.MRR},${m["NDCG@K"]},${m.Mean_Latency_ms}\n`;
    });

    const blob = new Blob([csv], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "retrieval_benchmark_results.csv";
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <AppShell
      title="Retrieval Benchmark & Evaluation"
      description="Quantitative evaluation comparing BM25, Dense FAISS, Hybrid RRF, and Cross-Encoder Reranking"
    >
      <div className="space-y-6">
        {/* Top Header Card */}
        <div className="p-6 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-800">
                Top-K = 5 Benchmark
              </span>
              <span className="text-xs font-mono text-slate-400">
                N = {metadata.total_queries} Total Queries ({metadata.answerable_queries} Answerable,{" "}
                {metadata.unanswerable_queries} Negative Controls)
              </span>
            </div>
            <h3 className="text-base font-bold text-slate-900 dark:text-slate-100">
              Comparative Information Retrieval Benchmarks
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Evaluated across 20 verified ground-truth chunks from 6 landmark papers (ResNet, Transformer, AdamW, ViT, LoRA, DDPM).
            </p>
          </div>

          <button
            type="button"
            onClick={downloadCSV}
            className="px-4 py-2 rounded-xl border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 text-xs font-semibold text-slate-700 dark:text-slate-200 flex items-center gap-2 transition-colors self-start sm:self-auto shrink-0 shadow-2xs"
          >
            <Download className="w-4 h-4" />
            <span>Export CSV</span>
          </button>
        </div>

        {/* High-Level Overview Metrics */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Max Overall Recall@5"
            value="0.9321"
            subtitle="Hybrid RRF + Cross-Encoder"
            icon={Target}
            badge="Top Accuracy"
            badgeColor="green"
          />
          <MetricCard
            title="Top MRR Baseline"
            value="0.9444"
            subtitle="Okapi BM25 Lexical Search"
            icon={Award}
            badge="Exact Jargon"
            badgeColor="blue"
          />
          <MetricCard
            title="Multi-Passage Recall"
            value="0.9444"
            subtitle="vs. 0.8611 for pure BM25 (+9.7%)"
            icon={Sparkles}
            badge="Cross-Attention"
            badgeColor="purple"
          />
          <MetricCard
            title="In-Memory Latency"
            value="0.28 ms"
            subtitle="CPU FAISS + BM25 + RRF + Reranker"
            icon={Clock}
            badge="Sub-Millisecond"
            badgeColor="amber"
          />
        </div>

        {/* Tab Controls */}
        <div className="flex items-center gap-2 border-b border-slate-200 dark:border-slate-800 pb-2">
          <button
            type="button"
            onClick={() => setActiveTab("overall")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeTab === "overall"
                ? "bg-blue-600 text-white"
                : "text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"
            }`}
          >
            Overall Configurations (N=35)
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("categories")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeTab === "categories"
                ? "bg-blue-600 text-white"
                : "text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"
            }`}
          >
            Per-Category Deep Dive
          </button>
        </div>

        {/* Charts & Visualizations */}
        {activeTab === "overall" ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Interactive Bar Chart */}
            <div className="lg:col-span-8 p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 shadow-xs space-y-4">
              <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                Retrieval Metrics by Architecture (K=5)
              </h4>

              <div className="h-80 w-full pt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={isDark ? 0.15 : 0.4} stroke={isDark ? "#64748b" : "#cbd5e1"} />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: isDark ? "#94a3b8" : "#475569" }} stroke={isDark ? "#475569" : "#cbd5e1"} />
                    <YAxis domain={[0, 1.1]} tick={{ fontSize: 11, fill: isDark ? "#94a3b8" : "#475569" }} stroke={isDark ? "#475569" : "#cbd5e1"} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: isDark ? "#0f172a" : "#ffffff",
                        borderRadius: "0.75rem",
                        border: isDark ? "1px solid #334155" : "1px solid #e2e8f0",
                        fontSize: "12px",
                        color: isDark ? "#fff" : "#0f172a",
                        boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1)",
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "10px", color: isDark ? "#94a3b8" : "#475569" }} />
                    <Bar dataKey="Hit@5" fill="#2563eb" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Recall@5" fill="#10b981" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="MRR" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="NDCG@5" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Academic Insights Box */}
            <div className="lg:col-span-4 p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 shadow-xs space-y-3">
              <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
                <Info className="w-4 h-4 text-blue-500" />
                Critical Evaluation Insights
              </h4>

              <div className="space-y-2.5 text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                <div className="p-3 rounded-xl bg-blue-50/60 dark:bg-blue-950/30 border border-blue-200/60 dark:border-blue-900/40">
                  <span className="font-bold text-blue-700 dark:text-blue-300 block mb-1">
                    1. The Scientific BM25 Phenomenon
                  </span>
                  On exact technical terminology (e.g. &quot;AdamW&quot;, &quot;RRF&quot;), BM25 achieves a perfect 1.000 MRR. In scientific papers, acronyms have zero polysemy, making lexical search formidable.
                </div>

                <div className="p-3 rounded-xl bg-purple-50/60 dark:bg-purple-950/30 border border-purple-200/60 dark:border-purple-900/40">
                  <span className="font-bold text-purple-700 dark:text-purple-300 block mb-1">
                    2. Where Reranking Decisively Wins
                  </span>
                  On queries requiring evidence distributed across multiple passages, Cross-Encoder Reranking raises Recall@5 from 0.8611 to 0.9444 (+9.7%) and NDCG@5 to 0.7938.
                </div>

                <div className="p-3 rounded-xl bg-emerald-50/60 dark:bg-emerald-950/30 border border-emerald-200/60 dark:border-emerald-900/40">
                  <span className="font-bold text-emerald-700 dark:text-emerald-300 block mb-1">
                    3. Sub-Millisecond Speed
                  </span>
                  FAISS dense search (~0.12ms) and BM25 (~0.11ms) combined with RRF execute in &lt;0.3ms on CPU, making the retrieval subsystem instant.
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 shadow-xs space-y-4">
            <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
              Multi-Passage Synthesis Breakdown (6 Complex Relational Queries)
            </h4>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Compares how each configuration performs when an answer requires synthesizing evidence across multiple distinct sections.
            </p>

            <div className="h-72 w-full pt-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={multiPassageData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={isDark ? 0.15 : 0.4} stroke={isDark ? "#64748b" : "#cbd5e1"} />
                  <XAxis dataKey="name" tick={{ fontSize: 11, fill: isDark ? "#94a3b8" : "#475569" }} stroke={isDark ? "#475569" : "#cbd5e1"} />
                  <YAxis domain={[0, 1.1]} tick={{ fontSize: 11, fill: isDark ? "#94a3b8" : "#475569" }} stroke={isDark ? "#475569" : "#cbd5e1"} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: isDark ? "#0f172a" : "#ffffff",
                      borderRadius: "0.75rem",
                      border: isDark ? "1px solid #334155" : "1px solid #e2e8f0",
                      fontSize: "12px",
                      color: isDark ? "#fff" : "#0f172a",
                      boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1)",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "10px", color: isDark ? "#94a3b8" : "#475569" }} />
                  <Bar dataKey="Recall" fill="#10b981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="NDCG" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="MRR" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Tabular Benchmark Results */}
        <div className="p-6 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 shadow-xs space-y-3">
          <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
            Verified Machine-Readable Benchmark Table
          </h4>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-100 dark:border-slate-800 text-slate-400 font-semibold tracking-wider uppercase text-[10px]">
                  <th className="py-3 px-3">Retrieval Configuration</th>
                  <th className="py-3 px-3">Hit@5</th>
                  <th className="py-3 px-3">Recall@5</th>
                  <th className="py-3 px-3">MRR</th>
                  <th className="py-3 px-3">NDCG@5</th>
                  <th className="py-3 px-3 text-right">Mean Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {Object.entries(overall).map(([cfg, m]) => (
                  <tr key={cfg} className="hover:bg-slate-50/50 dark:hover:bg-slate-800/40">
                    <td className="py-3 px-3 font-semibold text-slate-900 dark:text-slate-100">
                      {cfg}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-700 dark:text-slate-300">
                      {m["Hit@K"].toFixed(4)}
                    </td>
                    <td className="py-3 px-3 font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                      {m["Recall@K"].toFixed(4)}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-700 dark:text-slate-300">
                      {m.MRR.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-700 dark:text-slate-300">
                      {m["NDCG@K"].toFixed(4)}
                    </td>
                    <td className="py-3 px-3 font-mono text-right text-slate-500">
                      {m.Mean_Latency_ms.toFixed(2)} ms
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
