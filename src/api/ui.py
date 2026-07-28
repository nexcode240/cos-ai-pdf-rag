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


async def _document_data() -> tuple[str, list[tuple[str, str]]]:
    """Fetch library HTML and selector choices from the PDFs API."""
    async with _api_client(timeout=30) as client:
        response = await client.get("/api/v1/pdfs")
    response.raise_for_status()
    pdfs = [PDFListItem.model_validate(item) for item in response.json()]
    return _format_library_html(pdfs), [
        (pdf.name, pdf.pdf_id) for pdf in pdfs
    ]


def _escape_html(text: str) -> str:
    """Escape text for safe HTML table rendering."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _format_library_html(pdfs: list[PDFListItem]) -> str:
    """Render PDF list API results as a full-width HTML table."""
    if not pdfs:
        return (
            '<p style="opacity:0.7;margin:0.5rem 0">'
            "No documents uploaded yet."
            "</p>"
        )

    cell = "padding:0.55rem 0.75rem;border-bottom:1px solid rgba(255,255,255,0.12)"
    header = (
        f'<th style="text-align:left;{cell};opacity:0.8;font-weight:600">Name</th>'
        f'<th style="text-align:right;{cell};opacity:0.8;font-weight:600;width:5.5rem">Pages</th>'
        f'<th style="text-align:right;{cell};opacity:0.8;font-weight:600;width:5.5rem">Chunks</th>'
        f'<th style="text-align:left;{cell};opacity:0.8;font-weight:600;width:9rem">Uploaded</th>'
    )
    rows: list[str] = []
    for pdf in pdfs:
        uploaded = pdf.upload_timestamp.strftime("%Y-%m-%d %H:%M")
        rows.append(
            "<tr>"
            f'<td style="{cell}">{_escape_html(pdf.name)}</td>'
            f'<td style="text-align:right;{cell}">{pdf.page_count}</td>'
            f'<td style="text-align:right;{cell}">{pdf.doc_count}</td>'
            f'<td style="{cell}">{uploaded}</td>'
            "</tr>"
        )
    return (
        '<div style="width:100%;overflow-x:auto">'
        '<table style="width:100%;border-collapse:collapse;table-layout:fixed">'
        f"<thead><tr>{header}</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody>'
        "</table></div>"
    )


def _api_client(timeout: float = 600) -> httpx.AsyncClient:
    """Create a client for the application's REST API."""
    return httpx.AsyncClient(base_url=settings.API_BASE_URL, timeout=timeout)


def _error_detail(response: httpx.Response) -> str:
    """Extract a FastAPI error detail from an unsuccessful response."""
    try:
        body = response.json()
        if isinstance(body, dict) and isinstance(body.get("detail"), str):
            return body["detail"]
    except ValueError:
        pass
    return response.text or f"HTTP {response.status_code}"


async def _model_choices() -> list[str]:
    """Fetch installed chat models from the models API."""
    async with _api_client(timeout=30) as client:
        response = await client.get("/api/v1/models")
    if response.is_error:
        return [settings.DEFAULT_CHAT_MODEL]
    models = [ModelInfo.model_validate(item) for item in response.json()]
    return [model.name for model in models] or [settings.DEFAULT_CHAT_MODEL]


async def _refresh_ui() -> tuple[object, object, object]:
    """Refresh documents and models through the REST API."""
    try:
        library, choices = await _document_data()
    except (httpx.HTTPError, ValueError):
        library, choices = _format_library_html([]), []

    models = await _model_choices()
    selected_model = (
        settings.DEFAULT_CHAT_MODEL
        if settings.DEFAULT_CHAT_MODEL in models
        else models[0]
    )
    return (
        gr.update(value=library),
        gr.update(choices=choices, value=[]),
        gr.update(choices=models, value=selected_model),
    )


async def _refresh_library() -> tuple[object, object]:
    """Refresh documents through the REST API."""
    try:
        library, choices = await _document_data()
        return gr.update(value=library), gr.update(choices=choices, value=[])
    except (httpx.HTTPError, ValueError) as exc:
        return (
            gr.update(
                value=(
                    '<p style="opacity:0.7;margin:0.5rem 0">'
                    f"Failed to load documents: {_escape_html(str(exc))}"
                    "</p>"
                )
            ),
            gr.update(choices=[], value=[]),
        )


async def _upload_pdf(file_path: str | None) -> tuple[str, object, object]:
    """Upload a PDF through POST /api/v1/pdfs/upload."""
    if not file_path:
        library, selector = await _refresh_library()
        return "Select a PDF first.", library, selector

    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        library, selector = await _refresh_library()
        return "Only PDF files are supported.", library, selector

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
        library, choices = await _document_data()
        status = (
            f"Uploaded **{uploaded.name}** "
            f"({uploaded.page_count} pages, {uploaded.doc_count} chunks)."
        )
        return (
            status,
            gr.update(value=library),
            gr.update(choices=choices, value=[uploaded.pdf_id]),
        )
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        library, selector = await _refresh_library()
        return f"Upload failed: {exc}", library, selector


async def _delete_pdfs(
    pdf_ids: list[str] | None,
) -> tuple[str, object, object]:
    """Delete selected PDFs through DELETE /api/v1/pdfs/{pdf_id}."""
    pdf_ids = list(pdf_ids or [])
    if not pdf_ids:
        library, selector = await _refresh_library()
        return "Select at least one PDF to remove.", library, selector

    deleted: list[str] = []
    errors: list[str] = []
    try:
        async with _api_client() as client:
            for pdf_id in pdf_ids:
                response = await client.delete(f"/api/v1/pdfs/{pdf_id}")
                if response.is_error:
                    errors.append(f"{pdf_id}: {_error_detail(response)}")
                else:
                    deleted.append(pdf_id)
    except httpx.HTTPError as exc:
        library, selector = await _refresh_library()
        return f"Delete failed: {exc}", library, selector

    library, selector = await _refresh_library()
    if deleted and not errors:
        count = len(deleted)
        status = f"Removed {count} PDF{'s' if count != 1 else ''}."
    elif deleted and errors:
        status = (
            f"Removed {len(deleted)} PDF(s). "
            f"Failed: {'; '.join(errors)}"
        )
    else:
        status = f"Delete failed: {'; '.join(errors)}"
    return status, library, selector


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
                    with gr.Row():
                        refresh_button = gr.Button("Refresh")
                        delete_button = gr.Button(
                            "Remove selected",
                            variant="stop",
                        )
                    delete_status = gr.Markdown()
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
            with gr.Row(equal_height=True):
                gr.Markdown("### Document library")
                library_refresh_button = gr.Button(
                    "Refresh library",
                    scale=0,
                    min_width=140,
                )
            document_library = gr.HTML(
                value="<p style='opacity:0.7'>Loading documents...</p>",
                container=True,
                padding=False,
            )

        library_outputs = [document_library, pdf_selector]
        demo.load(
            _refresh_ui,
            outputs=[*library_outputs, model],
            show_progress="hidden",
        )
        refresh_button.click(
            _refresh_library,
            outputs=library_outputs,
            show_progress="hidden",
        )
        library_refresh_button.click(
            _refresh_library,
            outputs=library_outputs,
            show_progress="hidden",
        )
        upload_button.click(
            _upload_pdf,
            inputs=[file_input],
            outputs=[upload_status, *library_outputs],
        )
        delete_button.click(
            _delete_pdfs,
            inputs=[pdf_selector],
            outputs=[delete_status, *library_outputs],
            show_progress="hidden",
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
