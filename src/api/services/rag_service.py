"""RAG query service (multi-PDF production path via LangGraph)."""
from datetime import datetime
from typing import Optional, cast

from langchain_core.runnables import RunnableConfig
from sqlalchemy.orm import Session

from ..database import ChatMessage, ChatSession, PDFMetadata
from .rag_graph import RAGState, rag_graph


class RAGService:
    """Service for RAG operations across one or more PDF collections."""

    def query_multi_pdf(
        self,
        question: str,
        model: str,
        pdf_ids: Optional[list[str]],
        db: Session,
    ) -> tuple[str, list[dict[str, str | int | None]], list[str]]:
        """Query across PDFs using the LangGraph RAG flow."""
        query = db.query(PDFMetadata)
        if pdf_ids:
            query = query.filter(PDFMetadata.pdf_id.in_(pdf_ids))
        pdfs = query.all()

        if not pdfs:
            return "No PDFs found to query.", [], ["No PDFs found to query."]

        pdf_payload = [
            {
                "pdf_id": cast(str, cast(object, pdf.pdf_id)),
                "name": cast(str, cast(object, pdf.name)),
                "collection_name": cast(str, cast(object, pdf.collection_name)),
            }
            for pdf in pdfs
        ]
        initial_state: RAGState = {
            "question": question,
            "model": model,
            "pdfs": pdf_payload,
            "search_queries": [],
            "documents": [],
            "documents_relevant": False,
            "answer": "",
            "sources": [],
            "reasoning_steps": [
                "Searching across "
                f"{len(pdf_payload)} PDF(s): "
                f"{', '.join(pdf['name'] for pdf in pdf_payload)}",
                f"Using model: {model}",
            ],
        }
        result = rag_graph.invoke(
            initial_state,
            config=RunnableConfig(
                run_name="pdf_rag_graph",
                tags=["rag", "langgraph", "pdf"],
                metadata={
                    "model": model,
                    "pdf_count": len(pdf_payload),
                    "pdf_ids": [pdf["pdf_id"] for pdf in pdf_payload],
                    "pdf_names": [pdf["name"] for pdf in pdf_payload],
                    "question_preview": question[:120],
                },
            ),
        )
        return (
            result.get("answer") or "",
            result.get("sources") or [],
            result.get("reasoning_steps") or [],
        )

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[list[dict[str, str | int | None]]],
        db: Session,
    ) -> ChatMessage:
        """Save chat message to database."""
        session = (
            db.query(ChatSession)
            .filter(ChatSession.session_id == session_id)
            .first()
        )
        now = datetime.now()
        if not session:
            session = ChatSession(
                session_id=session_id,
                created_at=now,
                last_active=now,
            )
            db.add(session)
        else:
            setattr(session, "last_active", now)

        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
            timestamp=now,
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        return message

    def get_session_messages(self, session_id: str, db: Session) -> list[ChatMessage]:
        """Get all messages for a session."""
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.timestamp)
            .all()
        )
