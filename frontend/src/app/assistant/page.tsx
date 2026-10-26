"use client";

import React, { useState, useEffect, useCallback, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { AppShell } from "@/components/AppShell";
import { CitationChip } from "@/components/CitationChip";
import { EvidenceCard } from "@/components/EvidenceCard";
import { api } from "@/lib/api";
import { DocumentMetadata, QueryResponse, Citation } from "@/types/api";
import {
  MessageSquare,
  Sparkles,
  Layers,
  Sliders,
  Send,
  Loader2,
  AlertTriangle,
  FileText,
  Clock,
} from "lucide-react";

const SAMPLE_PROMPTS = [
  "How does Reciprocal Rank Fusion combine rankings from dense and sparse retrievers?",
  "Why should weight decay be decoupled from the gradient update in AdamW?",
  "What is the degradation problem in deep neural networks and how does ResNet address it?",
  "How does the Vision Transformer project image patches into input tokens?",
  "What is the catalytic mechanism of carbonic anhydrase in human red blood cells?",
];

function AssistantContent() {
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get("q") || "";
  const initialDoc = searchParams.get("doc") || "";

  const [query, setQuery] = useState(initialQuery);
  const [topK, setTopK] = useState(5);
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string>(initialDoc);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [highlightedChunkId, setHighlightedChunkId] = useState<string | null>(null);

  useEffect(() => {
    async function loadDocs() {
      try {
        const docs = await api.listDocuments(0, 50);
        setDocuments(docs);
      } catch {
        // Fallback or offline
      }
    }
    loadDocs();
  }, []);

  const handleAsk = useCallback(
    async (textToQuery?: string) => {
      const q = textToQuery || query;
      if (!q.trim()) return;

      setError(null);
      setLoading(true);
      setHighlightedChunkId(null);

      try {
        const res = await api.query({
          query: q.trim(),
          top_k: topK,
          document_ids: selectedDocId ? [selectedDocId] : undefined,
        });
        setResponse(res);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Failed to execute query.";
        setError(msg);
      } finally {
        setLoading(false);
      }
    },
    [query, topK, selectedDocId]
  );

  useEffect(() => {
    if (initialQuery) {
      const timer = setTimeout(() => {
        handleAsk(initialQuery);
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [initialQuery, handleAsk]);

  const handleCitationClick = (citation: Citation) => {
    setHighlightedChunkId(citation.chunk_id);
    const elem = document.getElementById(`chunk-${citation.chunk_id}`);
    if (elem) {
      elem.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  return (
    <AppShell
      title="Research Assistant"
      description="Grounded scientific question answering with mathematical attribution and exact page citations"
    >
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Query Console and Controls */}
        <div className="lg:col-span-7 space-y-5">
          {/* Query Input Box */}
          <div className="p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                <MessageSquare className="w-4 h-4 text-blue-600" />
                Research Inquiry
              </label>

              {documents.length > 0 && (
                <div className="flex items-center gap-1.5 text-xs">
                  <span className="text-slate-400">Filter to:</span>
                  <select
                    value={selectedDocId}
                    onChange={(e) => setSelectedDocId(e.target.value)}
                    className="py-1 px-2.5 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs font-medium focus:outline-hidden"
                  >
                    <option value="">All Indexed Papers ({documents.length})</option>
                    {documents.map((d) => (
                      <option key={d.document_id} value={d.document_id}>
                        {d.filename.slice(0, 30)}...
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            <div className="relative">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a technical question e.g. How does AdamW decouple weight decay from gradient updates?"
                rows={3}
                className="w-full p-3.5 rounded-xl bg-slate-50 dark:bg-slate-950/60 border border-slate-200 dark:border-slate-800 text-xs sm:text-sm text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-hidden focus:ring-2 focus:ring-blue-500 font-medium resize-none"
              />
            </div>

            {/* Retrieval Tuning Parameters */}
            <div className="pt-2 border-t border-slate-100 dark:border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-3">
                <span className="text-slate-500 flex items-center gap-1">
                  <Sliders className="w-3.5 h-3.5 text-slate-400" />
                  Top-K Evidence:
                </span>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="w-24 accent-blue-600 cursor-pointer"
                  />
                  <span className="font-mono font-bold text-slate-800 dark:text-slate-200 w-4">
                    {topK}
                  </span>
                </div>
              </div>

              <button
                type="button"
                onClick={() => handleAsk()}
                disabled={loading || !query.trim()}
                className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-semibold text-xs sm:text-sm flex items-center gap-2 transition-all shadow-md shadow-blue-600/20"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Retrieving & Synthesizing...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    <span>Run Query</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Quick Academic Prompts */}
          <div className="space-y-2">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
              Suggested Benchmark Queries
            </span>
            <div className="flex flex-wrap gap-1.5">
              {SAMPLE_PROMPTS.map((prompt, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => {
                    setQuery(prompt);
                    handleAsk(prompt);
                  }}
                  className="text-left text-xs px-3 py-1.5 rounded-xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 hover:border-blue-400 dark:hover:border-blue-600 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors shadow-2xs"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>

          {/* Error Banner */}
          {error && (
            <div className="p-4 rounded-2xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-xs text-rose-700 dark:text-rose-300 flex items-start gap-3">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Query Execution Error</p>
                <p className="mt-0.5 opacity-90">{error}</p>
              </div>
            </div>
          )}

          {/* Answer Card */}
          {response && (
            <div className="p-6 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 shadow-sm space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-100 dark:border-slate-800">
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 rounded-lg bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
                    <Sparkles className="w-4 h-4" />
                  </div>
                  <h4 className="font-bold text-sm text-slate-900 dark:text-slate-100">
                    Grounded Research Answer
                  </h4>
                </div>

                <div className="flex items-center gap-3 text-[11px] text-slate-400 font-mono">
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3 text-slate-400" />
                    Retrieval: {response.timing.retrieval_ms.toFixed(1)}ms
                  </span>
                  <span>•</span>
                  <span>Total: {response.timing.total_ms.toFixed(1)}ms</span>
                </div>
              </div>

              {/* Refusal Notice */}
              {response.insufficient_evidence && (
                <div className="p-3.5 rounded-xl bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800/80 text-xs text-amber-800 dark:text-amber-300 flex items-start gap-2.5">
                  <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-bold">Insufficient Evidence Safeguard:</span> The retrieved
                    passages did not contain sufficient verified evidence to answer this inquiry with
                    high confidence. To avoid hallucination, an explicit refusal statement was
                    returned.
                  </div>
                </div>
              )}

              {/* Formatted Answer Body */}
              <div className="text-xs sm:text-sm text-slate-800 dark:text-slate-200 leading-relaxed whitespace-pre-line font-normal">
                {response.answer}
              </div>

              {/* Citation Chips */}
              {response.citations.length > 0 && (
                <div className="pt-3 border-t border-slate-100 dark:border-slate-800">
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                    Verified Citations (Click to highlight passage)
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {response.citations.map((c) => (
                      <CitationChip
                        key={c.citation_id}
                        citation={c}
                        isActive={highlightedChunkId === c.chunk_id}
                        onClick={() => handleCitationClick(c)}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Retrieved Evidence Inspection Drawer */}
        <div className="lg:col-span-5 space-y-4">
          <div className="p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 shadow-xs">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-600" />
                <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  Retrieved Evidence Chunks
                </h4>
              </div>
              {response && (
                <span className="text-xs px-2 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800 font-mono text-slate-600 dark:text-slate-300 font-semibold">
                  {response.evidence.length} Chunks
                </span>
              )}
            </div>

            <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
              Inspect exact excerpts, page coordinates, and RRF/Cross-Encoder scores supplied to the LLM.
            </p>

            {loading ? (
              <div className="py-12 text-center text-xs text-slate-400 flex flex-col items-center gap-2">
                <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
                <span>Running dual-channel hybrid search...</span>
              </div>
            ) : !response || response.evidence.length === 0 ? (
              <div className="py-12 text-center border-2 border-dashed rounded-xl border-slate-200 dark:border-slate-800 text-slate-400 text-xs">
                <FileText className="w-8 h-8 mx-auto text-slate-300 dark:text-slate-600 mb-2" />
                <span>Submit an inquiry to inspect retrieved candidate evidence.</span>
              </div>
            ) : (
              <div className="space-y-3 max-h-[680px] overflow-y-auto pr-1">
                {response.evidence.map((ev, idx) => (
                  <div key={ev.chunk.chunk_id} id={`chunk-${ev.chunk.chunk_id}`}>
                    <EvidenceCard
                      evidence={ev}
                      index={idx}
                      highlighted={highlightedChunkId === ev.chunk.chunk_id}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}

export default function AssistantPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center text-xs text-slate-400">
          Loading Research Assistant...
        </div>
      }
    >
      <AssistantContent />
    </Suspense>
  );
}
