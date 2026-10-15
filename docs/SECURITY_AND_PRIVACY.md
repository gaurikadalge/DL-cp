# Security, Privacy & Compliance Guidelines

## 1. Threat Model & Safeguards

The Multimodal RAG Research Assistant ingests untrusted user files and integrates with third-party LLM providers. The following defensive safeguards are enforced:

### 1.1 Ingestion Security & Malicious File Prevention
- **Extension & Magic Byte Verification**: Uploaded files must match `.pdf` extension AND begin with the standard `%PDF-` file header magic bytes. Files masking executables or HTML will be rejected immediately.
- **Upload Size Boundaries**: Uploads are restricted by `MAX_UPLOAD_SIZE_MB` (default 25MB) to protect against Denial-of-Service (DoS) and memory exhaustion.
- **Path Traversal Protection**: Filenames submitted in `multipart/form-data` are sanitized. Document IDs are generated as UUIDs, preventing directory traversal (`../../etc/passwd`) attacks.
- **PDF Decompression & Parser Sandboxing**: PyMuPDF operations are executed with timeouts to defend against decompression bombs.

### 1.2 Credential & Secret Management
- **Zero Committed Secrets**: `.env` is permanently excluded via `.gitignore`. The repository contains only `.env.example` with empty dummy placeholders.
- **Secret Redaction in Logging & Debugging**: `Settings.safe_dict()` automatically redacts `llm_api_key` and other sensitive parameters. Plain text API tokens are strictly forbidden from log streams.
- **Least Privilege Access**: If deploying with cloud providers (e.g. AWS S3, OpenAI, Anthropic), read/write scopes must be restricted to project-specific buckets and keys.

### 1.3 Information Leakage Prevention
- **Sanitized Error Responses**: Production error handlers in `app/main.py` intercept unhandled exceptions and return unified `ErrorDetail` objects. Internal server paths, Python stack traces, and local filesystem structures are never exposed to API consumers.
- **Database & Index Isolation**: SQLite database files (`*.db`) and FAISS indexes (`*.faiss`) are barred from Git commits.

### 1.4 Data Privacy & Academic IP Protection
- **Ephemeral and Local Processing**: In local development, all document embeddings, indexes, and chunks reside solely on the researcher's local filesystem (`./data/uploads/`, `./indexes/`).
- **External LLM Safeguards**: When using commercial LLM APIs, users should ensure the provider's terms of service comply with their institution's confidentiality policies. A local LLM provider option (e.g. Ollama) is supported for private research environments.

---

## 2. Checklist Before Committing Code
- [ ] No `.env`, `*.key`, `*.pem`, or plain text credentials in the commit diff.
- [ ] No proprietary academic PDFs or user documents staged in `data/uploads/`.
- [ ] No large binary model weights (`*.pt`, `*.onnx`) or index binaries (`*.faiss`) staged.
- [ ] Only `.gitkeep` files remain in `data/uploads/` and `indexes/`.
- [ ] Ran `git status` and verified untracked files are appropriate.
