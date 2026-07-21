"""FastAPI main application."""
import logging

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .routers import health, models, pdfs, query
from .ui import create_ui

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
app = gr.mount_gradio_app(
    app,
    create_ui(),
    path="/ui",
    show_error=True,
    max_file_size="100mb",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/")
def root() -> RedirectResponse:
    """Redirect the root route to the Gradio interface."""
    return RedirectResponse(url="/ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
