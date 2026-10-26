/**
 * API Type Definitions matching FastAPI schemas (app/schemas.py).
 */

export type DependencyStatus =
  | "configured"
  | "not_configured"
  | "available"
  | "unavailable"
  | "not_loaded_scaffold"
  | "disabled";

export type LiteralStatus = "ok" | "degraded" | "down";

export interface AppInfo {
  app_name: string;
  version: string;
  environment: string;
  docs_url: string;
  health_url: string;
}

export interface HealthCheckResponse {
  status: LiteralStatus;
  app_name: string;
  version: string;
  environment: string;
  timestamp: string;
  dependencies: Record<string, DependencyStatus>;
}

export interface ErrorDetail {
  error_code: string;
  message: string;
  details?: unknown;
  timestamp: string;
}

export type DocumentProcessingStatus =
  | "uploaded"
  | "processing"
  | "indexed"
  | "failed"
  | "deleting";

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  status: DocumentProcessingStatus;
  upload_timestamp: string;
  file_size_bytes: number;
  message: string;
}

export interface DocumentMetadata {
  document_id: string;
  filename: string;
  status: DocumentProcessingStatus;
  page_count: number | null;
  chunk_count: number | null;
  created_at: string;
  updated_at: string;
  error_message?: string | null;
  file_size_bytes?: number;
}

export interface StableChunkMetadata {
  chunk_id: string;
  document_id: string;
  text: string;
  page_number: number;
  page_end?: number | null;
  section?: string | null;
  content_type: "text" | "table" | "caption" | "ocr" | string;
  source_filename: string;
  extraction_method: string;
  metadata?: Record<string, unknown>;
}

export interface RetrievalScoring {
  dense_score?: number | null;
  bm25_score?: number | null;
  fusion_score?: number | null;
  fusion_rank?: number | null;
  rerank_score?: number | null;
}

export interface RetrievedEvidence {
  chunk: StableChunkMetadata;
  scoring: RetrievalScoring;
}

export interface Citation {
  citation_id: string;
  chunk_id: string;
  document_id: string;
  source_filename: string;
  page_number: number;
  section?: string | null;
  snippet: string;
}

export interface QueryRequest {
  query: string;
  document_ids?: string[] | null;
  top_k?: number;
  stream?: boolean;
}

export interface QueryTimingMetadata {
  retrieval_ms: number;
  reranking_ms?: number | null;
  generation_ms: number;
  total_ms: number;
}

export interface QueryResponse {
  query_id?: string | null;
  query: string;
  answer: string;
  citations: Citation[];
  evidence: RetrievedEvidence[];
  insufficient_evidence: boolean;
  timing: QueryTimingMetadata;
}

// -----------------------------------------------------------------------------
// Evaluation Benchmark Types (evaluation/retrieval_benchmark_results.json)
// -----------------------------------------------------------------------------

export interface BenchmarkMetrics {
  "Hit@K": number;
  "Recall@K": number;
  MRR: number;
  "NDCG@K": number;
  Mean_Latency_ms: number;
}

export interface BenchmarkMetadata {
  total_queries: number;
  answerable_queries: number;
  unanswerable_queries: number;
  total_fixture_chunks: number;
  k: number;
  timestamp: string;
  evaluation_encoder: string;
  evaluation_reranker: string;
}

export interface BenchmarkResultsData {
  metadata: BenchmarkMetadata;
  overall_results: Record<string, BenchmarkMetrics>;
  category_results: Record<string, Record<string, BenchmarkMetrics>>;
}
