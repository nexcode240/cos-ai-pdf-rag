"""Gradio client for the existing FastAPI endpoints."""
from pathlib import Path
from uuid import uuid4

import gradio as gr
import httpx

from .config import settings
from .models import (
    ModelInfo,
    PDFListItem,
    PDFUploadResponse,
    QueryRequest,
    QueryResponse,
)


def _api_client() -> httpx.AsyncClient:
    """Create a client for the application's REST API."""
    return httpx.AsyncClient(base_url=settings.API_BASE_URL, timeout=600)


def _error_detail(response: httpx.Response) -> str:
    """Extract a FastAPI error detail from an unsuccessful response."""
    try:
        body = response.json()
        if isinstance(body, dict) and isinstance(body.get("detail"), str):
            return body["detail"]
    except ValueError:
        pass
    return response.text or f"HTTP {response.status_code}"


async def _document_data() -> tuple[
    list[list[str | int]], list[tuple[str, str]]
]:
    """Fetch document rows and selector choices from the PDFs API."""
    async with _api_client() as client:
        response = await client.get("/api/v1/pdfs")
    response.raise_for_status()
    pdfs = [PDFListItem.model_validate(item) for item in response.json()]
    rows = [
        [
            pdf.name,
            pdf.page_count,
            pdf.doc_count,
            pdf.upload_timestamp.strftime("%Y-%m-%d %H:%M"),
        ]
        for pdf in pdfs
    ]
    choices = [(pdf.name, pdf.pdf_id) for pdf in pdfs]
    return rows, choices


async def _model_choices() -> list[str]:
    """Fetch installed chat models from the models API."""
    async with _api_client() as client:
        response = await client.get("/api/v1/models")
    if response.is_error:
        return [settings.DEFAULT_CHAT_MODEL]
    models = [ModelInfo.model_validate(item) for item in response.json()]
    return [model.name for model in models] or [settings.DEFAULT_CHAT_MODEL]


async def _refresh_ui() -> tuple[object, object, object]:
    """Refresh documents and models through the REST API."""
    try:
        rows, choices = await _document_data()
    except (httpx.HTTPError, ValueError):
        rows, choices = [], []

    models = await _model_choices()
    selected_model = (
        settings.DEFAULT_CHAT_MODEL
        if settings.DEFAULT_CHAT_MODEL in models
        else models[0]
    )
    return (
        gr.update(value=rows),
        gr.update(choices=choices, value=[]),
        gr.update(choices=models, value=selected_model),
    )


async def _refresh_library() -> tuple[object, object]:
    """Refresh documents through the REST API."""
    try:
        rows, choices = await _document_data()
        return gr.update(value=rows), gr.update(choices=choices, value=[])
    except (httpx.HTTPError, ValueError):
        return gr.update(value=[]), gr.update(choices=[], value=[])


async def _upload_pdf(file_path: str | None) -> tuple[str, object, object]:
    """Upload a PDF through POST /api/v1/pdfs/upload."""
    if not file_path:
        table, selector = await _refresh_library()
        return "Select a PDF first.", table, selector

    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        table, selector = await _refresh_library()
        return "Only PDF files are supported.", table, selector

    try:
        with path.open("rb") as file_handle:
            async with _api_client() as client:
                response = await client.post(
                    "/api/v1/pdfs/upload",
                    files={"file": (path.name, file_handle, "application/pdf")},
                )
        if response.is_error:
            raise RuntimeError(_error_detail(response))

        uploaded = PDFUploadResponse.model_validate(response.json())
        rows, choices = await _document_data()
        status = (
            f"Uploaded **{uploaded.name}** "
            f"({uploaded.page_count} pages, {uploaded.doc_count} chunks)."
        )
        return (
            status,
            gr.update(value=rows),
            gr.update(choices=choices, value=[uploaded.pdf_id]),
        )
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        table, selector = await _refresh_library()
        return f"Upload failed: {exc}", table, selector


async def _chat(
    question: str,
    history: list[dict[str, str]] | None,
    model: str,
    pdf_ids: list[str] | None,
    session_id: str,
) -> tuple[list[dict[str, str]], str, str, str]:
    """Submit a question through POST /api/v1/query."""
    history = list(history or [])
    question = question.strip()
    if not question:
        return history, "", session_id, ""

    session_id = session_id or str(uuid4())
    history.append({"role": "user", "content": question})
    request = QueryRequest(
        question=question,
        model=model or settings.DEFAULT_CHAT_MODEL,
        pdf_ids=pdf_ids or None,
        session_id=session_id,
    )

    try:
        async with _api_client() as client:
            response = await client.post(
                "/api/v1/query",
                json=request.model_dump(exclude_none=True),
            )
        if response.is_error:
            raise RuntimeError(_error_detail(response))

        result = QueryResponse.model_validate(response.json())
        history.append({"role": "assistant", "content": result.answer})
        source_names = list(
            dict.fromkeys(source.pdf_name for source in result.sources)
        )
        source_text = (
            "**Sources:** " + ", ".join(source_names)
            if source_names
            else "**Sources:** No matching chunks"
        )
        return history, "", result.session_id, source_text
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        history.append(
            {"role": "assistant", "content": f"Query failed: {exc}"}
        )
        return history, "", session_id, ""


def create_ui() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(title="Cosmos AI PDF RAG") as demo:
        gr.Markdown(
            "# Cosmos AI PDF RAG\n"
            "Upload PDFs, choose the documents to search, and ask questions."
        )
        session_id = gr.State(value=lambda: str(uuid4()))

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    model = gr.Dropdown(
                        choices=[settings.DEFAULT_CHAT_MODEL],
                        value=settings.DEFAULT_CHAT_MODEL,
                        allow_custom_value=True,
                        label="Ollama model",
                    )
                    pdf_selector = gr.CheckboxGroup(
                        choices=[],
                        label="Documents",
                        info="Leave empty to search all uploaded PDFs.",
                        show_select_all=True,
                    )
                    refresh_button = gr.Button("Refresh documents")
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="PDF assistant",
                        height=500,
                        placeholder="Upload a PDF, then ask a question.",
                    )
                    sources = gr.Markdown()
                    with gr.Row():
                        question = gr.Textbox(
                            placeholder="Ask about your PDFs...",
                            show_label=False,
                            scale=5,
                        )
                        send_button = gr.Button("Send", variant="primary", scale=1)
                    clear_button = gr.Button("Clear chat")

        with gr.Tab("Documents"):
            file_input = gr.File(
                label="PDF file",
                file_types=[".pdf"],
                type="filepath",
            )
            upload_button = gr.Button("Upload and index", variant="primary")
            upload_status = gr.Markdown()
            document_table = gr.Dataframe(
                headers=["Name", "Pages", "Chunks", "Uploaded"],
                datatype=["str", "number", "number", "str"],
                interactive=False,
                label="Document library",
            )

        demo.load(
            _refresh_ui,
            outputs=[document_table, pdf_selector, model],
        )
        refresh_button.click(
            _refresh_library,
            outputs=[document_table, pdf_selector],
        )
        upload_button.click(
            _upload_pdf,
            inputs=[file_input],
            outputs=[upload_status, document_table, pdf_selector],
        )

        chat_inputs = [question, chatbot, model, pdf_selector, session_id]
        chat_outputs = [chatbot, question, session_id, sources]
        send_button.click(_chat, inputs=chat_inputs, outputs=chat_outputs)
        question.submit(_chat, inputs=chat_inputs, outputs=chat_outputs)
        clear_button.click(
            lambda: ([], "", str(uuid4()), ""),
            outputs=[chatbot, question, session_id, sources],
        )

    return demo
