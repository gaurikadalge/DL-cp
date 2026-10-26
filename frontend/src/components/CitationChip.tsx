import React from "react";
import { Citation } from "@/types/api";
import { Bookmark, FileText } from "lucide-react";

interface CitationChipProps {
  citation: Citation;
  onClick?: () => void;
  isActive?: boolean;
}

export function CitationChip({ citation, onClick, isActive }: CitationChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold transition-all shadow-xs ${
        isActive
          ? "bg-blue-600 text-white shadow-blue-500/25 ring-2 ring-blue-400"
          : "bg-blue-50 hover:bg-blue-100 text-blue-700 dark:bg-blue-950/60 dark:hover:bg-blue-900/60 dark:text-blue-300 border border-blue-200 dark:border-blue-800"
      }`}
      title={`${citation.source_filename} (p. ${citation.page_number}) - Click to inspect snippet`}
    >
      <Bookmark className="w-3 h-3 shrink-0" />
      <span>{citation.citation_id}</span>
      <span className="text-[10px] opacity-75 font-normal flex items-center gap-0.5">
        <FileText className="w-2.5 h-2.5 inline" />
        p.{citation.page_number}
      </span>
    </button>
  );
}
