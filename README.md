# Cosmos AI PDF RAG

A FastAPI learning clone of [ollama_pdf_rag](../ollama_pdf_rag). Same RAG pipeline and API shape, stripped down so you can study how PDF upload → chunk → embed → query works.

## What this mirrors

| This project | Original (`ollama_pdf_rag`) |
|---|---|
| `src/core/` | Shared RAG primitives (load, chunk, embed, LLM, pipeline) |
| `src/api/` | FastAPI routers, services, PostgreSQL metadata |
| No `web-ui/` | Skip Next.js — call the API via Swagger or curl |
| No Streamlit | One backend path only |

## Architecture (study path)

```
Upload PDF                Ask a question
    │                           │
    ▼                           ▼
POST /api/v1/pdfs/upload   POST /api/v1/query
    │                           │
    ▼                           ▼
DocumentProcessor          RAGService
  UnstructuredPDFLoader      MultiQueryRetriever (per PDF)
  RecursiveCharacterTextSplitter
    │                           │
    ▼                           ▼
VectorStore (Chroma)       ChatOllama answer + sources
  nomic-embed-text           (+ optional thinking models)
    │
    ▼
PostgreSQL pdfs table
```

Read files in this order while learning:

1. `src/core/document.py` — PDF load + chunking
2. `src/core/embeddings.py` — Chroma + Ollama embeddings
3. `src/core/llm.py` — prompts
4. `src/core/rag.py` — simple single-collection chain
5. `src/api/services/pdf_service.py` — upload orchestration
6. `src/api/services/rag_service.py` — multi-PDF query (what the real UI uses)
7. `src/api/routers/*` — thin HTTP layer

See [LEARNING.md](LEARNING.md) for a guided walkthrough.

## Prerequisites

- Python 3.10–3.12
- [PostgreSQL](https://www.postgresql.org/) running locally
- [Ollama](https://ollama.com) running locally
- Models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Setup

```bash
cd cosmos_ai_pdf_rag
uv sync
cp .env.example .env   # set DATABASE_URL (required)
createdb cosmos_ai_pdf_rag
uv run alembic upgrade head
```

## Run

```bash
uv run python run_api.py
```

- API: http://localhost:8001
- Swagger: http://localhost:8001/docs

## Quick try

```bash
# Health
curl http://localhost:8001/api/v1/health

# Upload
curl -X POST http://localhost:8001/api/v1/pdfs/upload \
  -F "file=@/path/to/doc.pdf"

# List PDFs
curl http://localhost:8001/api/v1/pdfs

# Query
curl -X POST http://localhost:8001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "model": "llama3.2"}'
```

## Defaults (same as original)

| Setting | Value |
|---|---|
| Chunk size / overlap | 7500 / 100 |
| Embedding model | `nomic-embed-text` |
| Default chat model | `llama3.2` |
| Vector store | Chroma under `data/vectors` |
| Metadata DB | PostgreSQL (`DATABASE_URL` in `.env`) |
