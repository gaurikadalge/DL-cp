"use client";

import React, { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { AppShell } from "@/components/AppShell";
import { FileUploader } from "@/components/FileUploader";
import { StatusBadge } from "@/components/StatusBadge";
import { api } from "@/lib/api";
import { DocumentMetadata } from "@/types/api";
import { formatDate, formatBytes } from "@/lib/utils";
import {
  Files,
  Trash2,
  Search,
  MessageSquare,
  AlertCircle,
  FileText,
  Loader2,
  Layers,
  CheckCircle2,
} from "lucide-react";

export default function DocumentLibraryPage() {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<{ type: "success" | "error"; text: string } | null>(
    null
  );

  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.listDocuments(0, 50);
      setDocuments(data);
    } catch {
      // Offline or network error
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    api
      .listDocuments(0, 50)
      .then((data) => {
        if (!active) return;
        setDocuments(data);
      })
      .catch(() => {
        // Offline or network error
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  const handleDelete = async (docId: string, filename: string) => {
    if (!confirm(`Are you sure you want to delete "${filename}" and purge all its vectors from FAISS and BM25?`)) {
      return;
    }

    setDeletingId(docId);
    setActionMessage(null);

    try {
      await api.deleteDocument(docId);
      setActionMessage({
        type: "success",
        text: `Document "${filename}" and its vector representations have been purged.`,
      });
      // Refresh library
      await fetchDocuments();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to delete document.";
      setActionMessage({
        type: "error",
        text: msg,
      });
    } finally {
      setDeletingId(null);
    }
  };

  const filteredDocs = documents.filter((d) =>
    d.filename.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <AppShell
      title="Document Ingestion & Library"
      description="Manage uploaded research PDFs, trigger multimodal parsing, and inspect indexing statuses"
      onRefresh={fetchDocuments}
      isRefreshing={loading}
    >
      <div className="space-y-6">
        {/* Upload Card */}
        <div className="p-6 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 shadow-xs">
          <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 mb-1 flex items-center gap-2">
            <Files className="w-4 h-4 text-blue-600" />
            Upload Research Publication
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
            Upload single- or dual-column scientific papers. PyMuPDF will extract layout, OCR will parse scanned figures, and tables will be converted to Markdown.
          </p>

          <FileUploader onUploadSuccess={() => fetchDocuments()} />
        </div>

        {/* Action Status Notification */}
        {actionMessage && (
          <div
            className={`p-3.5 rounded-xl border flex items-center gap-2.5 text-xs ${
              actionMessage.type === "success"
                ? "bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-300"
                : "bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-300"
            }`}
          >
            {actionMessage.type === "success" ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
            )}
            <div className="flex-1 font-medium">{actionMessage.text}</div>
          </div>
        )}

        {/* Document Library Table Card */}
        <div className="p-6 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 shadow-xs space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                <span>Indexed Library</span>
                <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
                  {documents.length} Total
                </span>
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Persistent records stored in SQLite and synchronized to FAISS vector index
              </p>
            </div>

            <div className="relative w-full sm:w-64">
              <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search by filename..."
                className="w-full pl-9 pr-3 py-1.5 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs font-medium text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-hidden focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          {loading ? (
            <div className="py-16 text-center text-xs text-slate-400 flex flex-col items-center gap-2">
              <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
              <span>Fetching documents from SQLite store...</span>
            </div>
          ) : filteredDocs.length === 0 ? (
            <div className="py-16 text-center border-2 border-dashed rounded-xl border-slate-200 dark:border-slate-800 text-slate-400 text-xs">
              <FileText className="w-8 h-8 mx-auto text-slate-300 dark:text-slate-600 mb-2" />
              {searchTerm ? (
                <span>No documents match your search &quot;{searchTerm}&quot;.</span>
              ) : (
                <span>No papers uploaded yet. Drag and drop a research PDF above.</span>
              )}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-100 dark:border-slate-800 text-slate-400 font-semibold tracking-wider uppercase text-[10px]">
                    <th className="py-3 px-3">Document</th>
                    <th className="py-3 px-3">Status</th>
                    <th className="py-3 px-3">Pages</th>
                    <th className="py-3 px-3">Chunks</th>
                    <th className="py-3 px-3">Indexed Date</th>
                    <th className="py-3 px-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                  {filteredDocs.map((doc) => (
                    <tr
                      key={doc.document_id}
                      className="hover:bg-slate-50/50 dark:hover:bg-slate-800/40 transition-colors"
                    >
                      <td className="py-3.5 px-3">
                        <div className="flex items-center gap-2.5">
                          <div className="w-8 h-8 rounded-lg bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400 flex items-center justify-center shrink-0">
                            <FileText className="w-4 h-4" />
                          </div>
                          <div className="min-w-0 max-w-xs sm:max-w-md">
                            <span className="font-semibold text-slate-800 dark:text-slate-200 block truncate">
                              {doc.filename}
                            </span>
                            <span className="text-[11px] text-slate-400 font-mono">
                              ID: {doc.document_id}
                              {doc.file_size_bytes ? ` • ${formatBytes(doc.file_size_bytes)}` : ""}
                            </span>
                          </div>
                        </div>
                      </td>

                      <td className="py-3.5 px-3">
                        <StatusBadge status={doc.status} />
                      </td>

                      <td className="py-3.5 px-3 font-mono text-slate-600 dark:text-slate-300">
                        {doc.page_count ?? 1}
                      </td>

                      <td className="py-3.5 px-3 font-mono font-semibold text-slate-800 dark:text-slate-200">
                        <span className="inline-flex items-center gap-1">
                          <Layers className="w-3 h-3 text-slate-400" />
                          {doc.chunk_count ?? 0}
                        </span>
                      </td>

                      <td className="py-3.5 px-3 text-slate-500 dark:text-slate-400 font-mono text-[11px]">
                        {formatDate(doc.created_at)}
                      </td>

                      <td className="py-3.5 px-3 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          <Link
                            href={`/assistant?doc=${doc.document_id}`}
                            className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:text-blue-600 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                            title="Query this document in Assistant"
                          >
                            <MessageSquare className="w-3.5 h-3.5" />
                          </Link>

                          <button
                            type="button"
                            onClick={() => handleDelete(doc.document_id, doc.filename)}
                            disabled={deletingId === doc.document_id}
                            className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-400 hover:text-rose-600 hover:border-rose-300 dark:hover:border-rose-800 hover:bg-rose-50 dark:hover:bg-rose-950/40 transition-colors disabled:opacity-50"
                            title="Delete document and purge vectors"
                          >
                            {deletingId === doc.document_id ? (
                              <Loader2 className="w-3.5 h-3.5 animate-spin text-rose-600" />
                            ) : (
                              <Trash2 className="w-3.5 h-3.5" />
                            )}
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}
