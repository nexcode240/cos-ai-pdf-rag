# Learning guide: Cosmos AI PDF RAG

Use this project to understand `ollama_pdf_rag` without the Next.js / Streamlit noise.

## Mental model

Three layers:

1. **Core** (`src/core/`) — pure RAG building blocks, no HTTP
2. **Services** (`src/api/services/`) — business logic that wires core + DB
3. **Routers** (`src/api/routers/`) — HTTP endpoints only

The original repo also has a Streamlit app and a Next.js UI. Both talk to the same ideas; FastAPI is the clearest place to learn them.

## Step 1 — Ingest a PDF

Follow `PDFService.upload_and_process`:

1. Save the uploaded file under `data/pdfs/uploads/`
2. `DocumentProcessor.load_pdf` → `UnstructuredPDFLoader`
3. `DocumentProcessor.split_documents` → chunks of **7500** chars, overlap **100**
4. Stamp each chunk with `pdf_id`, `pdf_name`, `chunk_index`
5. `VectorStore.create_vector_db` → embed with `nomic-embed-text`, store in Chroma
6. Insert a row into SQLite `pdfs`

Try: upload a short PDF via `/docs`, then inspect `data/pdfs/uploads/` and `data/api.db`.

## Step 2 — Ask a question

Follow `RAGService.query_multi_pdf`:

1. Load selected PDF rows (or all)
2. For each PDF's Chroma collection, run **MultiQueryRetriever**
   - LLM rewrites the question into ~2 variants
   - Retrieves top `k=3` chunks per collection
3. Merge chunks, keep top **10**, label with `[Source: name]`
4. RAG prompt asks the chat model to answer **only** from context and cite sources
5. Optional: thinking models (`qwen`, `deepseek`) use `ollama.chat(..., think=True)`
6. Persist user + assistant messages in SQLite

Compare with `src/core/rag.py`: that file is a simpler **single-collection** LangChain chain. The API service is the multi-PDF production path.

## Step 3 — Map endpoints to code

| Endpoint | Router | Service |
|---|---|---|
| `POST /api/v1/pdfs/upload` | `routers/pdfs.py` | `PDFService.upload_and_process` |
| `GET /api/v1/pdfs` | `routers/pdfs.py` | `PDFService.list_pdfs` |
| `DELETE /api/v1/pdfs/{id}` | `routers/pdfs.py` | `PDFService.delete_pdf` |
| `POST /api/v1/query` | `routers/query.py` | `RAGService.query_multi_pdf` |
| `GET /api/v1/sessions/{id}/messages` | `routers/query.py` | `RAGService.get_session_messages` |
| `GET /api/v1/models` | `routers/models.py` | Ollama list + chat filter |
| `GET /api/v1/health` | `routers/health.py` | Ollama + PDF counts |

## Step 4 — Config knobs

All in `src/api/config.py` / `.env`:

- `OLLAMA_HOST` — where Ollama listens
- `VECTOR_DB_DIR` — Chroma persistence
- `PDF_STORAGE_DIR` — uploaded files
- `EMBEDDING_MODEL` / `DEFAULT_CHAT_MODEL`

Note: some services still hardcode `nomic-embed-text` and chunk sizes (same as the original). When you spot that, you've found a real improvement opportunity.

## Experiments to try

1. Change chunk size to `1000` / overlap `200` and re-upload — how do answers change?
2. Query with `pdf_ids` set vs unset — confirm multi-PDF filtering
3. Read `reasoning_steps` in the query response metadata — watch retrieval unfold
4. Compare `RAGPipeline.get_response` (core) vs `RAGService.query_multi_pdf` (API)

## What was intentionally left out

- Next.js chat UI and its SQLite (`web-ui/`)
- Streamlit UI (`src/app/`)
- Docs site / notebooks / CI polish

Those are presentation layers. The RAG brain lives here.
