"""Database models and session management."""
from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, Integer, JSON, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_DIR = Path("data")
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/api.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PDFMetadata(Base):
    """PDF metadata table."""

    __tablename__ = "pdfs"

    pdf_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    collection_name = Column(String, unique=True, nullable=False)
    upload_timestamp = Column(DateTime, nullable=False)
    doc_count = Column(Integer, nullable=False)
    page_count = Column(Integer, nullable=False)
    is_sample = Column(Boolean, default=False)
    file_path = Column(String)


class ChatSession(Base):
    """Chat session table."""

    __tablename__ = "chat_sessions"

    session_id = Column(String, primary_key=True)
    created_at = Column(DateTime, nullable=False)
    last_active = Column(DateTime, nullable=False)


class ChatMessage(Base):
    """Chat message table."""

    __tablename__ = "messages"

    message_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    sources = Column(JSON)
    timestamp = Column(DateTime, nullable=False)


Base.metadata.create_all(bind=engine)
