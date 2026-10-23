# Live Demonstration Script: Multimodal RAG Research Assistant

**Estimated Duration**: 5 to 7 Minutes  
**Target Audience**: Academic Evaluators, Examination Committee, Project Guides  
**Demonstration Mode**: Supports both **Online Mode** (Live OpenAI API) and **Deterministic Offline Mode** (Built-in CPU fallback)  

---

## 1. Pre-Demonstration Checklist & Setup (T - 2 Minutes)

Open two terminal windows side by side. Ensure you are in the repository root `/Users/gaurik/Documents/DL_CP`:

```bash
# Terminal 1: Application Backend & Commands
cd /Users/gaurik/Documents/DL_CP
source .venv/bin/activate

# Terminal 2: Frontend & Monitoring
cd /Users/gaurik/Documents/DL_CP
source .venv/bin/activate
```

Verify that port 8000 and port 8501 are not in use:
```bash
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :8501 | xargs kill -9 2>/dev/null || true
```

---

## 2. Minute-by-Minute Demonstration Walkthrough

### Minute 0:00 – 1:00 | System Health & Automated Verification
**Action**: In Terminal 1, run the automated test suite and static analysis check to demonstrate engineering rigor.

```bash
# Run the complete test suite (84 passing tests)
pytest -v

# Run the code formatting and linter
ruff check .
```

**Presenter Narration**:
> *"Good morning. Before interacting with the UI, we demonstrate our automated quality foundation. Our test suite contains 84 passing unit, integration, multimodal stress, and end-to-end tests executing in under two seconds. Our codebase complies with strict Ruff linting and formatting standards with zero errors."*

---

### Minute 1:00 – 2:00 | Launching Backend & Live Smoke Test
**Action**: Start the FastAPI backend in Terminal 1, then run the live smoke test script.

```bash
# Terminal 1: Launch FastAPI Backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In Terminal 2, run:
```bash
# Terminal 2: Run CLI Smoke Test against the live server
python scripts/smoke_test.py --url http://127.0.0.1:8000
```

**Expected Smoke Test Output**:
```
[OK] Root Service Info (http://127.0.0.1:8000/) returned HTTP 200
[OK] Health Check (http://127.0.0.1:8000/health) returned HTTP 200
[OK] Document Library (http://127.0.0.1:8000/documents) returned HTTP 200
[OK] Research QA Query (http://127.0.0.1:8000/query) returned HTTP 200
[SUCCESS] All smoke test assertions passed successfully!
```

**Presenter Narration**:
> *"The backend is live on port 8000. Our standalone HTTP smoke test verifies that all service contracts—health monitoring, document metadata persistence in SQLite, and hybrid query synthesis—return HTTP 200 with sub-millisecond in-memory retrieval."*

---

### Minute 2:00 – 3:00 | Launching the Streamlit Research Portal
**Action**: In Terminal 2, launch the Streamlit frontend.

```bash
# Terminal 2: Launch Streamlit UI
streamlit run app/frontend.py --server.port 8501
```
Open your browser to `http://localhost:8501`.

**Presenter Narration**:
> *"We now open the Streamlit Research Portal. The interface is organized into three tabs: Research QA, Document Library & Ingestion, and Retrieval Benchmarks. Let's inspect the Document Library tab to see our indexed research collection."*

---

### Minute 3:00 – 4:30 | Ingestion & Asking a Verified Research Question
**Action**:
1. Click the **Research QA** tab.
2. In the query box, enter the technical query:
   ```
   How does Reciprocal Rank Fusion combine rankings from dense and sparse retrievers?
   ```
3. Set **Top-K Results** to `5` and check **Enable Cross-Encoder Reranker**.
4. Click **Ask Assistant**.

**Expected Result**:
- **Answer**: Synthesizes the exact RRF formula ($1 / (k + r(d))$ with $k=60$) citing Cormack et al. (2009).
- **Citations**: Displays verifiable citation badge chips (e.g., `[doc_...#chunk_0007]`).
- **Evidence Drawer**: Expand the "Retrieved Evidence Chunks" accordion to show the exact source text, page number, and similarity score.
- **Timing Badge**: Shows retrieval time ($~0.28\text{ ms}$) and LLM synthesis time.

**Presenter Narration**:
> *"Notice the output: the system synthesizes a precise answer explaining Reciprocal Rank Fusion. Every assertion is accompanied by an authoritative citation badge. When we expand the evidence drawer, you can inspect the exact source chunk, page number, and fusion score. The citation validator has mathematically cross-referenced this chunk ID against the retrieved set to eliminate hallucination."*

---

### Minute 4:30 – 5:30 | Adversarial & Insufficient-Evidence Safeguard
**Action**: In the same query box, submit an out-of-domain question that is absent from the indexed literature:
```
What is the catalytic mechanism of carbonic anhydrase in human red blood cells?
```
Click **Ask Assistant**.

**Expected Result**:
- The system returns an explicit refusal:
  > *"I do not have sufficient evidence in the uploaded research papers to answer this question. Please upload relevant literature."*
- An amber alert banner flags: `Status: Insufficient Evidence in Context`.
- Zero hallucinated citations are emitted.

**Presenter Narration**:
> *"Here we demonstrate our hallucination safeguard. When asked about biochemistry—a domain absent from our deep learning papers—the system does not invent a plausible answer. Instead, the minimum evidence filter detects insufficient retrieval confidence and immediately issues an explicit refusal statement."*

---

### Minute 5:30 – 6:30 | Cold Restart & Persistence Verification
**Action**:
1. In Terminal 1, stop the backend with `CTRL+C`.
2. Re-start the backend immediately:
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
3. In the Streamlit UI, refresh the page and navigate to **Document Library**.

**Expected Result**:
- The document collection and indexed chunk counts are completely intact, restored from SQLite (`data/research_assistant.db`) and disk FAISS/BM25 index files (`indexes/`).
- Submitting the previous query immediately succeeds without re-indexing.

**Presenter Narration**:
> *"We just performed a cold restart of the backend process. Notice that our document library and retrieval indices were immediately restored from SQLite and disk storage with zero data loss or index corruption."*

---

### Minute 6:30 – 7:00 | Benchmark Visualizer & Conclusion
**Action**: Click the **Retrieval Benchmarks** tab in the Streamlit UI.

**Expected Result**:
- Renders the interactive comparison table and high-resolution category chart ([docs/benchmark_comparison.png](file:///Users/gaurik/Documents/DL_CP/docs/benchmark_comparison.png) and [docs/category_performance.png](file:///Users/gaurik/Documents/DL_CP/docs/category_performance.png)).
- Highlight the **0.9444 Multi-Passage Recall** of Hybrid + Reranker.

**Presenter Narration**:
> *"Finally, in our benchmark dashboard, evaluators can inspect our quantitative evaluation across 35 queries. Notice that while BM25 achieves a high baseline on acronyms, our Hybrid + Cross-Encoder Reranker achieves 0.9444 Recall@5 on multi-passage synthesis queries, proving the value of deep cross-attention for complex academic synthesis. This concludes our live demonstration."*

---

## 3. Comprehensive Troubleshooting Matrix

| Issue Encountered | Root Cause | Immediate Remediation Command |
| :--- | :--- | :--- |
| **Port 8000 Conflict** | Lingering background uvicorn process | `lsof -ti :8000 \| xargs kill -9` |
| **Port 8501 Conflict** | Lingering Streamlit process | `lsof -ti :8501 \| xargs kill -9` |
| **Missing OpenAI API Key** | `OPENAI_API_KEY` not set in `.env` | System defaults automatically to deterministic `MockLLMProvider` (offline demo continues seamlessly). |
| **No Module Named 'evaluation'** | Running script without repository root in path | Fixed in repository via dynamic `sys.path` insertion; run `python evaluation/evaluate_retrieval.py`. |
| **Tesseract OCR Warning** | System binary `tesseract` not installed on host OS | `MockOCREngine` automatically handles simulated OCR; install via `brew install tesseract` on macOS if live OCR binary is desired. |
| **Empty Document Library** | SQLite database file was deleted | Re-upload any test PDF or run `python -m evaluation.evaluate_retrieval` to populate baseline indexes. |

---

## 4. Deterministic Offline Demonstration Route

If presenting in an environment without internet access or an OpenAI API key:
1. Ensure `OPENAI_API_KEY=""` in `.env`.
2. The application will log:
   `[INFO] LLM provider initialized: deterministic_mock (offline mode)`
3. All retrieval, FAISS search, BM25 matching, RRF fusion, cross-encoder reranking, and citation validations execute **100% identically** on local CPU.
4. Answer synthesis extracts ground-truth excerpts directly from top retrieved chunks and tags citations authoritatively.
5. In your narration, state:
   > *"We are currently demonstrating in Offline Fallback Mode using our deterministic in-memory provider. The full retrieval, reranking, and citation validation pipelines are executing live on CPU."*
