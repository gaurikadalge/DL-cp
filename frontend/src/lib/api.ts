/**
 * Centralized API Client for FastAPI Backend Integration.
 */

import {
  AppInfo,
  BenchmarkResultsData,
  DocumentMetadata,
  DocumentUploadResponse,
  HealthCheckResponse,
  QueryRequest,
  QueryResponse,
} from "@/types/api";
import { FALLBACK_BENCHMARK_DATA } from "@/data/benchmarkData";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

class ApiClientError extends Error {
  status?: number;
  code?: string;
  details?: unknown;

  constructor(message: string, status?: number, code?: string, details?: unknown) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = new Headers(options.headers || {});

  if (!(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  try {
    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      let errMessage = `HTTP error ${response.status}: ${response.statusText}`;
      let errCode = `HTTP_${response.status}`;
      let errDetails: unknown = null;

      try {
        const errData = await response.json();
        if (errData && typeof errData === "object") {
          errMessage = errData.message || errData.detail || errMessage;
          errCode = errData.error_code || errCode;
          errDetails = errData.details || null;
        }
      } catch {
        // Response was not JSON
      }

      throw new ApiClientError(errMessage, response.status, errCode, errDetails);
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiClientError) {
      throw error;
    }
    // Network errors (e.g. server down / connection refused)
    throw new ApiClientError(
      `Could not connect to backend server at ${API_BASE_URL}. Ensure FastAPI is running.`,
      0,
      "NETWORK_ERROR",
      error
    );
  }
}

export const api = {
  getBaseUrl(): string {
    return API_BASE_URL;
  },

  /**
   * Health and System metadata endpoints.
   */
  async getHealth(): Promise<HealthCheckResponse> {
    return request<HealthCheckResponse>("/health");
  },

  async getAppInfo(): Promise<AppInfo> {
    return request<AppInfo>("/");
  },

  /**
   * Document lifecycle endpoints.
   */
  async listDocuments(skip = 0, limit = 50): Promise<DocumentMetadata[]> {
    return request<DocumentMetadata[]>(`/documents?skip=${skip}&limit=${limit}`);
  },

  async getDocument(documentId: string): Promise<DocumentMetadata> {
    return request<DocumentMetadata>(`/documents/${documentId}`);
  },

  async getDocumentStatus(documentId: string): Promise<DocumentMetadata> {
    return request<DocumentMetadata>(`/documents/${documentId}/status`);
  },

  async uploadDocument(file: File): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append("file", file);

    return request<DocumentUploadResponse>("/documents/upload", {
      method: "POST",
      body: formData,
    });
  },

  async deleteDocument(documentId: string): Promise<{ document_id: string; status: string; message: string }> {
    return request<{ document_id: string; status: string; message: string }>(
      `/documents/${documentId}`,
      {
        method: "DELETE",
      }
    );
  },

  /**
   * Research query and question answering.
   */
  async query(req: QueryRequest): Promise<QueryResponse> {
    return request<QueryResponse>("/query", {
      method: "POST",
      body: JSON.stringify(req),
    });
  },

  /**
   * Quantitative retrieval benchmark results.
   */
  async getBenchmarkResults(): Promise<BenchmarkResultsData> {
    try {
      const data = await request<BenchmarkResultsData>("/evaluation/benchmark");
      if (data && data.overall_results) {
        return data;
      }
      return FALLBACK_BENCHMARK_DATA;
    } catch {
      // Fallback to verified local evaluation data when backend is not hosting the file
      return FALLBACK_BENCHMARK_DATA;
    }
  },
};
