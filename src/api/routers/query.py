"""RAG query endpoints."""
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..dependencies import get_db, get_rag_service
from ..models import QueryRequest, QueryResponse, SourceInfo
from ..services.rag_service import RAGService

router = APIRouter(prefix="/api/v1", tags=["query"])
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
def query_pdfs(
    request: QueryRequest,
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
):
    """Query across PDFs with source attribution."""
    logger.info(
        "Received query: question='%s...', model=%s",
        request.question[:50],
        request.model,
    )

    session_id = request.session_id or str(uuid.uuid4())

    rag_service.save_message(
        session_id=session_id,
        role="user",
        content=request.question,
        sources=None,
        db=db,
    )

    try:
        answer, sources, reasoning_steps = rag_service.query_multi_pdf(
            question=request.question,
            model=request.model,
            pdf_ids=request.pdf_ids,
            db=db,
        )
    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg.lower() and "404" in error_msg:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Model '{request.model}' not found. "
                    f"Install it with: ollama pull {request.model}"
                ),
            ) from e
        raise HTTPException(status_code=500, detail=f"Query failed: {error_msg}") from e

    message = rag_service.save_message(
        session_id=session_id,
        role="assistant",
        content=answer,
        sources=sources,
        db=db,
    )

    return QueryResponse(
        answer=answer,
        sources=[SourceInfo(**s) for s in sources],
        metadata={
            "model_used": request.model,
            "chunks_retrieved": len(sources),
            "pdfs_queried": len({s["pdf_id"] for s in sources}),
            "reasoning_steps": reasoning_steps,
        },
        session_id=session_id,
        message_id=message.message_id,
    )


@router.get("/sessions/{session_id}/messages")
def get_session_messages(
    session_id: str,
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
):
    """Get chat history for a session."""
    messages = rag_service.get_session_messages(session_id, db)
    return [
        {
            "message_id": msg.message_id,
            "role": msg.role,
            "content": msg.content,
            "sources": msg.sources,
            "timestamp": msg.timestamp,
        }
        for msg in messages
    ]
