"""Configuration settings for FastAPI application."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env")

    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    PDF_STORAGE_DIR: str = "data/pdfs/uploads"
    VECTOR_DB_DIR: str = "data/vectors"
    DATABASE_URL: str = Field(..., description="PostgreSQL connection URL (required via .env)")
    API_BASE_URL: str = "http://127.0.0.1:8001"
    OLLAMA_HOST: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    DEFAULT_CHAT_MODEL: str = "llama3.2"


settings = Settings()  # pyright: ignore[reportCallIssue]
