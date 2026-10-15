# Project Overview: Multimodal RAG Research Assistant

## 1. Executive Summary
The **Multimodal RAG Research Assistant** is an academic AI research system designed to parse, index, search, and synthesize knowledge from dense academic publications (primarily complex multi-column PDFs containing equations, tables, figures, and textual arguments).

Academic research papers present distinctive challenges for standard Retrieval-Augmented Generation (RAG) pipelines:
1. **Structural Complexity**: Multi-column layouts, footnotes, figure captions, and tabular data are frequently mangled by generic text extractors.
2. **Vocabulary and Acronym Diversity**: Dense academic domain jargon and abbreviations fail under pure semantic/dense search, while pure keyword search misses conceptual synonyms.
3. **Hallucination Risk**: Academic synthesis requires strict factual grounding; claims made without direct evidence or traceable citations degrade user trust.
4. **Traceability**: Researchers require immediate attribution to exact source documents, sections, and page numbers.

The Multimodal RAG Research Assistant addresses these challenges through a modular pipeline featuring PyMuPDF layout-aware extraction with OCR fallbacks, hybrid dense (FAISS) and sparse (BM25) reciprocal rank fusion, optional cross-encoder reranking, and grounded generation with verifiable inline citations.

---

## 2. Core Objectives
- **Document Ingestion & Parsing**: Provide high-fidelity parsing of complex academic PDFs, preserving page numbers, section boundaries, tables, and captions.
- **Deterministic Semantic Chunking**: Generate stable, identifiable chunks with reproducible IDs (`{document_id}#chunk_{index}`) and comprehensive provenance metadata.
- **Hybrid Retrieval**: Combine dense semantic representations (`sentence-transformers/all-MiniLM-L6-v2` + FAISS) with lexical keyword matching (`rank-bm25`) using Reciprocal Rank Fusion (RRF).
- **Grounded Answer Synthesis**: Enforce strict context grounding in LLM answer generation with explicit failure modes (`insufficient_evidence: true`) when evidence is lacking.
- **Traceable Attribution**: Provide page-accurate citations and verbatim snippets for every synthesized statement.
- **Developer & Researcher Ergonomics**: Deliver a clean FastAPI backend with predictable schemas, comprehensive test coverage, and a modular architecture allowing rapid experimentation.

---

## 3. Scope & Phasing
To prevent premature complexity, the project is structured across distinct engineering phases:
- **Phase 0 (Current Foundation)**: Repository architecture, baseline FastAPI service, configuration management, API contracts, testing scaffold, and team workflow.
- **Phase 1 (Ingestion & Extraction)**: PDF extraction with PyMuPDF, OCR fallback with Tesseract, and semantic chunking.
- **Phase 2 (Hybrid Indexing & Retrieval)**: FAISS vector indexing, BM25 keyword index, and RRF ranking.
- **Phase 3 (Grounded Synthesis & Citations)**: LLM abstraction layer, prompt engineering for citation anchoring, and hallucination guardrails.
- **Phase 4 (User Interface & Exploration)**: Streamlit interactive research portal with PDF viewer and citation highlights.
- **Phase 5 (Evaluation & Benchmarks)**: Retrieval metrics (Hit@K, MRR) and RAG triad evaluation (faithfulness, answer relevance, context relevance).

---

## 4. Key Planned Capabilities vs. Current Status

| Capability | Description | Status |
| :--- | :--- | :--- |
| **Service Foundation** | FastAPI app with `/health`, `/` and config validation | **Implemented & Tested** |
| **Configuration Engine** | Centralized pydantic-settings with secret masking | **Implemented & Tested** |
| **API Contracts** | Formal request/response schemas for all lifecycle stages | **Implemented & Documented** |
| **Test & Smoke Suite** | Pytest unit tests and CLI smoke test script | **Implemented & Tested** |
| **PDF Extraction** | PyMuPDF text, table, and layout extraction | Planned (Phase 1) |
| **OCR Fallback** | Tesseract OCR for scanned/image-based figures | Planned (Phase 1) |
| **Semantic Chunking** | Chunking with page-span preservation and stable IDs | Planned (Phase 1) |
| **Dense Vector Index** | Sentence Transformers + FAISS vector search | Planned (Phase 2) |
| **Lexical Search** | BM25 keyword indexing for domain terms/acronyms | Planned (Phase 2) |
| **Hybrid Ranking** | Reciprocal Rank Fusion (RRF) algorithm | Planned (Phase 2) |
| **Reranking** | Cross-Encoder reranking stage | Optional / Planned (Phase 2) |
| **Grounded LLM Generation**| Provider abstraction with strict citation prompts | Planned (Phase 3) |
| **Interactive UI** | Streamlit research interface with citation viewer | Planned (Phase 4) |
| **Evaluation Harness** | Retrieval and faithfulness benchmark suite | Planned (Phase 5) |
