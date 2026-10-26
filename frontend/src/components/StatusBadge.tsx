import React from "react";
import { cn } from "@/lib/utils";
import { CheckCircle2, Clock, AlertTriangle, XCircle, Trash2 } from "lucide-react";
import { DocumentProcessingStatus } from "@/types/api";

interface StatusBadgeProps {
  status: DocumentProcessingStatus | string;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const normalized = status.toLowerCase();

  if (normalized === "indexed" || normalized === "ok" || normalized === "available") {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20",
          className
        )}
      >
        <CheckCircle2 className="w-3.5 h-3.5" />
        <span className="capitalize">{status}</span>
      </span>
    );
  }

  if (normalized === "processing" || normalized === "uploaded") {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 animate-pulse",
          className
        )}
      >
        <Clock className="w-3.5 h-3.5" />
        <span className="capitalize">{status}</span>
      </span>
    );
  }

  if (normalized === "degraded" || normalized === "not_configured" || normalized === "disabled") {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20",
          className
        )}
      >
        <AlertTriangle className="w-3.5 h-3.5" />
        <span className="capitalize">{status.replace("_", " ")}</span>
      </span>
    );
  }

  if (normalized === "deleting") {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/20",
          className
        )}
      >
        <Trash2 className="w-3.5 h-3.5" />
        <span className="capitalize">{status}</span>
      </span>
    );
  }

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20",
        className
      )}
    >
      <XCircle className="w-3.5 h-3.5" />
      <span className="capitalize">{status.replace("_", " ")}</span>
    </span>
  );
}

export default StatusBadge;
