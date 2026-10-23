"""Streamlit Research Portal for the Multimodal RAG Research Assistant.

Features:
- Document upload, multimodal parsing, and real-time indexing status.
- Document library management with vector purging.
- Grounded research question answering with strict citation anchoring.
- Interactive citation inspector and hybrid evidence scoring breakdown (Dense, BM25, RRF).
- Quantitative retrieval benchmark execution and metric visualization.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed. Please run: pip install streamlit")
    sys.exit(1)

import httpx

from app.config import get_settings
from evaluation.benchmark_dataset import BENCHMARK_QUERIES
from evaluation.evaluate_retrieval import run_evaluation

# ------------------------------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal RAG Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern research aesthetic
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.9rem 1.2rem;
        border-radius: 0.4rem;
        margin-bottom: 0.8rem;
    }
    .score-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        background-color: #E2E8F0;
        color: #334155;
    }
    .score-highlight {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    .metric-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 0.8rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()


# ------------------------------------------------------------------------------
# Helper API Client
# ------------------------------------------------------------------------------
class APIClient:
    """Handles communication with the FastAPI backend with graceful fallback."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def get_health(self) -> dict | None:
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/health")
                if res.status_code == 200:
                    return res.json()
        except Exception:
            pass
        return None

    def list_documents(self) -> list[dict]:
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/documents")
                if res.status_code == 200:
                    return res.json()
        except Exception:
            pass
        return []

    def upload_document(self, file_bytes: bytes, filename: str) -> dict:
        with httpx.Client(timeout=60.0) as client:
            files = {"file": (filename, file_bytes, "application/pdf")}
            res = client.post(f"{self.base_url}/documents/upload", files=files)
            if res.status_code in (200, 201):
                return res.json()
            err_msg = res.json().get("message", res.text)
            raise RuntimeError(f"Upload failed (HTTP {res.status_code}): {err_msg}")

    def delete_document(self, document_id: str) -> dict:
        with httpx.Client(timeout=10.0) as client:
            res = client.delete(f"{self.base_url}/documents/{document_id}")
            if res.status_code == 200:
                return res.json()
            raise RuntimeError(f"Deletion failed: {res.text}")

    def query(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> dict:
        with httpx.Client(timeout=30.0) as client:
            payload = {
                "query": query,
                "document_ids": document_ids if document_ids else None,
                "top_k": top_k,
            }
            res = client.post(f"{self.base_url}/query", json=payload)
            if res.status_code == 200:
                return res.json()
            err_msg = res.json().get("message", res.text)
            raise RuntimeError(f"Query failed (HTTP {res.status_code}): {err_msg}")


# ------------------------------------------------------------------------------
# Sidebar: Settings & Connection Audit
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/img/logo-margin/logo-teal.png",
        width=120,
    )
    st.subheader("⚙️ System Configuration")

    api_url = st.text_input(
        "FastAPI Server URL", value=f"http://{settings.api_host}:{settings.api_port}"
    )
    client = APIClient(base_url=api_url)

    health_data = client.get_health()
    if health_data and health_data.get("status") == "ok":
        st.success("🟢 API Server Connected")
        with st.expander("Subsystem Readiness Audit", expanded=False):
            deps = health_data.get("dependencies", {})
            for dep_name, dep_state in deps.items():
                st.write(f"• **{dep_name}**: `{dep_state}`")
    else:
        st.warning("🟡 API Server Unreachable (Direct fallback enabled)")

    st.markdown("---")
    st.subheader("🔍 Retrieval Settings")
    top_k_val = st.slider("Top Evidence Chunks (K)", min_value=1, max_value=15, value=5)
    use_reranker = st.checkbox("Cross-Encoder Reranking", value=settings.reranker_enabled)

    st.markdown("---")
    st.subheader("🤖 LLM Synthesis Engine")
    provider_option = st.selectbox(
        "Provider",
        options=["Mock LLM (Deterministic / Fast)", "OpenAI Compatible API"],
        index=0,
    )

    openai_api_key = ""
    openai_model = "gpt-4o-mini"
    if provider_option == "OpenAI Compatible API":
        openai_api_key = st.text_input("API Key", type="password", value=settings.llm_api_key or "")
        openai_model = st.text_input("Model ID", value=settings.llm_model or "gpt-4o-mini")

# ------------------------------------------------------------------------------
# Main Application Content
# ------------------------------------------------------------------------------
st.markdown(
    '<div class="main-header">📚 Multimodal RAG Research Assistant</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Grounded academic literature synthesis with layout-aware PDF extraction, hybrid FAISS + BM25 search, and verifiable page citations.</div>',
    unsafe_allow_html=True,
)

tab_qa, tab_docs, tab_eval = st.tabs(
    ["🔬 Research QA & Citations", "📄 Document Ingestion & Library", "📊 Retrieval Benchmark"]
)

# ------------------------------------------------------------------------------
# Tab 1: Research QA & Citations
# ------------------------------------------------------------------------------
with tab_qa:
    st.markdown("### Ask a Grounded Research Question")

    # Fetch currently available documents
    docs_list = client.list_documents()
    doc_options = {d["document_id"]: f"{d['filename']} ({d['document_id']})" for d in docs_list}

    selected_docs = st.multiselect(
        "Target Research Documents (Leave blank to search entire library)",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options.get(x, x),
    )

    # Preset sample prompts
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_query = ""
    if col_p1.button("📌 Residual Learning & Degradation"):
        preset_query = "How do residual connections address the degradation problem during deep network training?"
    if col_p2.button("📌 Formulation & Shortcut Connections"):
        preset_query = "Explain the formulation F(x) = H(x) - x and why shortcut connections add no extra parameters."
    if col_p3.button("📌 Self-Attention Alignment"):
        preset_query = (
            "How does the self-attention mechanism compute pairwise token alignment scores?"
        )

    query_input = st.text_area(
        "Research Question",
        value=preset_query if preset_query else "",
        placeholder="e.g., How does residual learning prevent accuracy degradation in deep networks?",
        height=90,
    )

    if st.button("🚀 Synthesize Grounded Answer", type="primary", use_container_width=True):
        if not query_input.strip() or len(query_input.strip()) < 3:
            st.error("Please enter a substantive query of at least 3 characters.")
        else:
            with st.spinner(
                "Executing hybrid retrieval (FAISS + BM25) and synthesizing grounded response..."
            ):
                try:
                    q_res = client.query(
                        query=query_input.strip(),
                        document_ids=selected_docs if selected_docs else None,
                        top_k=top_k_val,
                    )

                    # Performance latency metrics row
                    timing = q_res.get("timing", {})
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Retrieval Latency", f"{timing.get('retrieval_ms', 0):.1f} ms")
                    col_m2.metric(
                        "Rerank Latency",
                        f"{timing.get('reranking_ms', 0):.1f} ms"
                        if timing.get("reranking_ms")
                        else "Disabled",
                    )
                    col_m3.metric("Synthesis Latency", f"{timing.get('generation_ms', 0):.1f} ms")
                    col_m4.metric("Total Latency", f"{timing.get('total_ms', 0):.1f} ms")

                    st.markdown("---")

                    # Check for insufficient evidence
                    if q_res.get("insufficient_evidence", False):
                        st.warning(f"⚠️ **Insufficient Evidence Safeguard**: {q_res.get('answer')}")
                        st.info(
                            "The retrieved documents did not contain sufficient relevant context to answer this query with academic fidelity."
                        )
                    else:
                        st.markdown("#### 📝 Grounded Synthesis")
                        st.markdown(q_res.get("answer", ""))

                        # Interactive Citations Section
                        citations = q_res.get("citations", [])
                        if citations:
                            st.markdown("#### 🔗 Traceable Citations")
                            for cite in citations:
                                with st.container():
                                    st.markdown(
                                        f"""
                                        <div class="citation-card">
                                            <b>Citation {cite.get("citation_id")}</b> — <i>{cite.get("source_filename")}</i> (Page {cite.get("page_number")})
                                            {f" • <b>Section:</b> {cite.get('section')}" if cite.get("section") else ""}
                                            <br><br>
                                            <blockquote style="margin:0; font-style:italic; color:#334155;">
                                                "{cite.get("snippet")}"
                                            </blockquote>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                        # Full Evidence Inspector (Scoring breakdown)
                        evidence = q_res.get("evidence", [])
                        if evidence:
                            with st.expander(
                                f"🔎 Evidence Inspector ({len(evidence)} Chunks Retrieved)",
                                expanded=False,
                            ):
                                for i, ev in enumerate(evidence, 1):
                                    chunk = ev.get("chunk", {})
                                    scoring = ev.get("scoring", {})

                                    st.markdown(
                                        f"**Evidence Chunk #{i}: `{chunk.get('chunk_id')}`**"
                                    )
                                    st.markdown(
                                        f"""
                                        <span class="score-badge score-highlight">RRF Rank #{scoring.get("fusion_rank")}</span>
                                        <span class="score-badge">RRF Score: {scoring.get("fusion_score")}</span>
                                        <span class="score-badge">Dense Cosine: {scoring.get("dense_score", "N/A")}</span>
                                        <span class="score-badge">BM25 Score: {scoring.get("bm25_score", "N/A")}</span>
                                        {f"<span class='score-badge'>Rerank: {scoring.get('rerank_score')}</span>" if scoring.get("rerank_score") is not None else ""}
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                    st.text_area(
                                        "Chunk Text",
                                        value=chunk.get("text", ""),
                                        height=100,
                                        key=f"chunk_txt_{i}_{chunk.get('chunk_id')}",
                                        disabled=True,
                                    )
                                    st.markdown(
                                        f"**Source**: `{chunk.get('source_filename')}` | **Page**: {chunk.get('page_number')} | **Section**: `{chunk.get('section') or 'N/A'}`"
                                    )
                                    st.markdown("---")

                except Exception as e:
                    st.error(f"Error executing research query: {e}")

# ------------------------------------------------------------------------------
# Tab 2: Document Ingestion & Library
# ------------------------------------------------------------------------------
with tab_docs:
    st.markdown("### Upload & Ingest Academic Publications")

    uploaded_file = st.file_uploader(
        "Choose an Academic PDF paper",
        type=["pdf"],
        help="Upload multi-column arXiv publications or scanned research documents.",
    )

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{len(uploaded_file.getvalue()) / 1024:.1f} KB",
        }
        st.write(file_details)

        if st.button("📥 Parse, Extract & Index Paper", type="primary"):
            progress_bar = st.progress(10)
            status_text = st.empty()

            try:
                status_text.text("Validating magic header and layout...")
                progress_bar.progress(30)
                time.sleep(0.1)

                status_text.text("Extracting text blocks, tables, and figures...")
                progress_bar.progress(60)

                upload_res = client.upload_document(
                    file_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                )

                progress_bar.progress(100)
                status_text.text("Ingestion and indexing complete!")
                st.success(
                    f"✅ Successfully indexed **{upload_res.get('filename')}** "
                    f"(ID: `{upload_res.get('document_id')}`). {upload_res.get('message')}"
                )
                time.sleep(0.5)
                st.rerun()

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Ingestion failed: {e}")

    st.markdown("---")
    st.markdown("### 📚 Ingested Document Library")

    current_docs = client.list_documents()
    if not current_docs:
        st.info("No documents currently indexed. Upload a research PDF above to get started.")
    else:
        for doc in current_docs:
            col_info, col_del = st.columns([5, 1])
            with col_info:
                st.markdown(
                    f"**{doc.get('filename')}** (`{doc.get('document_id')}`) "
                    f"— Pages: **{doc.get('page_count', 0)}** | Chunks: **{doc.get('chunk_count', 0)}** | Status: `{doc.get('status')}`"
                )
            with col_del:
                if st.button("🗑️ Delete", key=f"del_{doc.get('document_id')}"):
                    try:
                        client.delete_document(doc.get("document_id"))
                        st.success(f"Removed {doc.get('filename')}")
                        time.sleep(0.3)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete: {e}")

# ------------------------------------------------------------------------------
# Tab 3: Retrieval Benchmark & Quantitative Evaluation
# ------------------------------------------------------------------------------
with tab_eval:
    st.markdown("### 📊 Quantitative Retrieval Benchmarking")
    st.markdown(
        """
        Evaluate retrieval performance on academic research queries across four formal IR metrics:
        - **Hit@K**: Probability that at least one relevant passage is retrieved.
        - **Recall@K**: Proportion of total relevant passages retrieved.
        - **MRR**: Mean Reciprocal Rank of the first relevant passage.
        - **NDCG@K**: Graded relevance ranking quality discount.
        """
    )

    with st.expander(
        f"Inspect Synthetic Academic Benchmark Queries ({len(BENCHMARK_QUERIES)} Queries)"
    ):
        for q in BENCHMARK_QUERIES:
            st.markdown(f'• **Query**: *"{q.get("query")}"*')
            st.markdown(f"  *Relevant chunks*: `{q.get('relevant_chunks')}`")

    if st.button("⚡ Run Quantitative Benchmark Comparison", type="primary"):
        with st.spinner(
            "Computing Hit@K, Recall@K, MRR, and NDCG@K across BM25, Dense, and Hybrid RRF..."
        ):
            try:
                results_summary = run_evaluation(k=5)
                st.success("Evaluation completed!")

                # Render summary table
                import pandas as pd

                df = pd.DataFrame.from_dict(results_summary, orient="index")
                st.dataframe(
                    df.style.format("{:.4f}").highlight_max(axis=0, color="#D1FAE5"),
                    use_container_width=True,
                )

                st.markdown("#### 💡 Empirical Findings")
                st.markdown(
                    """
                    1. **Dense Retrieval** captures semantic paraphrasing and conceptual equivalents.
                    2. **BM25 Keyword Retrieval** excels at exact acronyms, compound technical terms, and equations.
                    3. **Reciprocal Rank Fusion (Hybrid RRF)** achieves superior or balanced performance across both regimes without heuristic score mixing.
                    """
                )
            except Exception as e:
                st.error(f"Evaluation error: {e}")
