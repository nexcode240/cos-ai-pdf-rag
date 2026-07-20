"""FastAPI main application."""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import health, models, pdfs, query

app = FastAPI(
    title="Cosmos AI PDF RAG API",
    description="Learning clone of ollama_pdf_rag — FastAPI PDF RAG with Ollama",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pdfs.router)
app.include_router(query.router)
app.include_router(models.router)
app.include_router(health.router)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Cosmos AI PDF RAG API",
        "docs": "/docs",
        "health": "/api/v1/health",
        "default_model": settings.DEFAULT_CHAT_MODEL,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
