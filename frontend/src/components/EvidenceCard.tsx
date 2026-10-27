"use client";

import React, { useState } from "react";
import { RetrievedEvidence } from "@/types/api";
import { ChevronDown, ChevronUp, FileText, Layers, Hash, Sparkles } from "lucide-react";

interface EvidenceCardProps {
  evidence: RetrievedEvidence;
  index: number;
  highlighted?: boolean;
}

export function EvidenceCard({ evidence, index, highlighted }: EvidenceCardProps) {
  const [expanded, setExpanded] = useState(false);
  const { chunk, scoring } = evidence;

  return (
    <div
      className={`rounded-xl border transition-all duration-200 overflow-hidden ${
        highlighted
          ? "border-blue-500 bg-blue-50/40 dark:bg-blue-950/20 ring-2 ring-blue-400/40"
          : "border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 hover:border-slate-300 dark:hover:border-slate-700"
      }`}
    >
      <div
        onClick={() => setExpanded(!expanded)}
        className="p-3.5 flex items-start justify-between gap-3 cursor-pointer select-none"
      >
        <div className="flex items-start gap-2.5 min-w-0">
          <div className="w-6 h-6 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 font-semibold text-xs flex items-center justify-center shrink-0 mt-0.5">
            #{index + 1}
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className="font-medium text-xs text-slate-900 dark:text-slate-100 truncate max-w-[220px]">
                {chunk.source_filename}
              </span>
              <span className="inline-flex items-center gap-1 text-[11px] px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 font-mono">
                <FileText className="w-3 h-3 text-slate-400" />
                p.{chunk.page_number}
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded uppercase font-semibold tracking-wider bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 border border-indigo-200/60 dark:border-indigo-800/60">
                {chunk.content_type}
              </span>
            </div>

            {chunk.section && (
              <p className="text-xs text-slate-500 dark:text-slate-400 truncate mb-1">
                {chunk.section}
              </p>
            )}

            <p className="text-xs text-slate-700 dark:text-slate-300 line-clamp-2 leading-relaxed">
              {chunk.text}
            </p>
          </div>
        </div>

        <button
          type="button"
          className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 p-1 shrink-0"
        >
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
      </div>

      {/* Expanded Details & Scores */}
      {expanded && (
        <div className="px-3.5 pb-3.5 pt-1 border-t border-slate-100 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-950/40 text-xs">
          <div className="mb-3">
            <h5 className="font-semibold text-slate-800 dark:text-slate-200 mb-1 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-blue-500" />
              Full Extracted Passage
            </h5>
            <div className="p-2.5 rounded-lg bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 font-mono text-[11px] leading-relaxed text-slate-800 dark:text-slate-200 whitespace-pre-wrap max-h-60 overflow-y-auto">
              {chunk.text}
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-[11px]">
            {scoring.fusion_rank !== undefined && scoring.fusion_rank !== null && (
              <div className="p-2 rounded-md bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800">
                <span className="text-slate-400 block text-[10px] uppercase font-medium">RRF Rank</span>
                <span className="font-semibold text-slate-800 dark:text-slate-100 flex items-center gap-1">
                  <Hash className="w-3 h-3 text-blue-500" />
                  {scoring.fusion_rank}
                </span>
              </div>
            )}
            {scoring.fusion_score !== undefined && scoring.fusion_score !== null && (
              <div className="p-2 rounded-md bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800">
                <span className="text-slate-400 block text-[10px] uppercase font-medium">RRF Score</span>
                <span className="font-mono font-semibold text-slate-800 dark:text-slate-100">
                  {scoring.fusion_score.toFixed(4)}
                </span>
              </div>
            )}
            {scoring.dense_score !== undefined && scoring.dense_score !== null && (
              <div className="p-2 rounded-md bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800">
                <span className="text-slate-400 block text-[10px] uppercase font-medium">FAISS Cosine</span>
                <span className="font-mono font-semibold text-slate-800 dark:text-slate-100">
                  {scoring.dense_score.toFixed(4)}
                </span>
              </div>
            )}
            {scoring.rerank_score !== undefined && scoring.rerank_score !== null && (
              <div className="p-2 rounded-md bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800">
                <span className="text-slate-400 block text-[10px] uppercase font-medium">Cross-Encoder</span>
                <span className="font-mono font-semibold text-indigo-600 dark:text-indigo-400 flex items-center gap-1">
                  <Sparkles className="w-3 h-3 text-indigo-500" />
                  {scoring.rerank_score.toFixed(4)}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
