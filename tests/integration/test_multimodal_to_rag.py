"""Integration test verifying end-to-end handoff from Person 3 (Ingestion) to Person 1 (RAG Engine)."""

import fitz  # PyMuPDF
import numpy as np

from ingestion.image_extractor import MockCaptioningEngine
from ingestion.ocr_engine import MockOCREngine
from ingestion.pipeline import MultimodalIngestionPipeline
from rag_engine.bm25_retriever import BM25Retriever
from rag_engine.dense_retriever import DenseRetriever
from rag_engine.embedding_model import EmbeddingService
from rag_engine.hybrid_retriever import HybridRetriever
from rag_engine.index_manager import IndexManager


class DeterministicBagOfWordsEncoder:
    """Lightweight deterministic encoder for integration testing without HuggingFace downloads."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, texts, **kwargs):
        import hashlib
        vectors = []
        for text in texts:
            words = text.lower().split()
            vec = np.zeros(self.dim, dtype=np.float32)
            for w in words:
                idx = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


def make_test_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 50), "1. Introduction to Attention Mechanisms", fontsize=14)
    page.insert_text(
        fitz.Point(50, 80),
        "Attention mechanisms allow models to focus on specific elements of an input sequence. "
        "Self-attention calculates alignment scores between all pairs of words in a sentence.",
        fontsize=10,
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_ingestion_to_rag_engine_handoff():
    # 1. Initialize Ingestion Pipeline (Person 3)
    pipeline = MultimodalIngestionPipeline(
        ocr_engine=MockOCREngine(),
        captioning_engine=MockCaptioningEngine(),
    )

    pdf_bytes = make_test_pdf()
    result = pipeline.process(pdf_bytes, filename="attention_paper.pdf")

    assert len(result.chunks) > 0

    # 2. Initialize RAG Engine IndexManager (Person 1)
    encoder = DeterministicBagOfWordsEncoder(dim=16)
    emb_service = EmbeddingService(custom_encoder=encoder)
    dense_ret = DenseRetriever(dimension=16)
    bm25_ret = BM25Retriever()
    index_mgr = IndexManager(
        dense_retriever=dense_ret,
        bm25_retriever=bm25_ret,
        embedding_service=emb_service,
    )

    # 3. Execute Handoff Contract: Add Person 3 chunks directly to Person 1 IndexManager
    index_mgr.add_chunks(result.chunks)

    assert dense_ret.total_chunks == len(result.chunks)
    assert bm25_ret.total_chunks == len(result.chunks)

    # 4. Execute Hybrid Search
    hybrid = HybridRetriever(
        dense_retriever=dense_ret,
        bm25_retriever=bm25_ret,
        embedding_service=emb_service,
    )
    evidence = hybrid.retrieve(query="self-attention alignment scores", top_k=2)

    assert len(evidence) > 0
    assert "attention" in evidence[0].chunk.text.lower()
    assert evidence[0].chunk.document_id == result.document_id
    assert evidence[0].chunk.source_filename == "attention_paper.pdf"
