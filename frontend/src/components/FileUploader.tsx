"use client";

import React, { useState, useRef } from "react";
import { Upload, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import { DocumentUploadResponse } from "@/types/api";

interface FileUploaderProps {
  onUploadSuccess?: (doc: DocumentUploadResponse) => void;
  className?: string;
}

export function FileUploader({ onUploadSuccess, className }: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only academic PDF documents (.pdf) are supported.");
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      setError("File exceeds maximum upload size of 25MB.");
      return;
    }

    setError(null);
    setSuccess(null);
    setUploading(true);

    try {
      const response = await api.uploadDocument(file);
      setSuccess(`"${response.filename}" successfully extracted & indexed!`);
      if (onUploadSuccess) {
        onUploadSuccess(response);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to upload document.";
      setError(msg);
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className={`space-y-3 ${className || ""}`}>
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 ${
          isDragging
            ? "border-blue-500 bg-blue-50/50 dark:bg-blue-950/20"
            : "border-slate-300 dark:border-slate-700 hover:border-blue-400 dark:hover:border-blue-500 bg-white/50 dark:bg-slate-900/50"
        } ${uploading ? "pointer-events-none opacity-60" : ""}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFile(e.target.files[0]);
            }
          }}
        />

        <div className="flex flex-col items-center justify-center gap-3">
          <div className="w-12 h-12 rounded-2xl bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400 flex items-center justify-center shadow-xs">
            {uploading ? (
              <Loader2 className="w-6 h-6 animate-spin" />
            ) : (
              <Upload className="w-6 h-6" />
            )}
          </div>

          <div>
            <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">
              {uploading ? "Ingesting and Indexing PDF..." : "Click or drag & drop research paper"}
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Supports standard PDFs, two-column layouts, tables & scanned pages (Up to 25MB)
            </p>
          </div>
        </div>
      </div>

      {/* Notifications */}
      {error && (
        <div className="p-3 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 flex items-start gap-2.5 text-xs text-rose-700 dark:text-rose-300">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold">Upload Failed:</span> {error}
          </div>
        </div>
      )}

      {success && (
        <div className="p-3 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800 flex items-center gap-2.5 text-xs text-emerald-700 dark:text-emerald-300">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <div className="flex-1">{success}</div>
        </div>
      )}
    </div>
  );
}
