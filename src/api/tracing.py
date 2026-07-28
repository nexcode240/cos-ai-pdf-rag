"""LangSmith tracing bootstrap for LangGraph / LangChain runs."""
from __future__ import annotations

import logging
import os

from .config import settings

logger = logging.getLogger(__name__)


def configure_langsmith() -> bool:
    """Enable LangSmith tracing from settings when an API key is present.

    Returns True when tracing is active.
    """
    api_key = (settings.LANGSMITH_API_KEY or "").strip()
    if not settings.LANGSMITH_TRACING or not api_key:
        os.environ.pop("LANGSMITH_TRACING", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        logger.info("LangSmith tracing disabled")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
    if settings.LANGSMITH_ENDPOINT:
        os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT

    logger.info(
        "LangSmith tracing enabled (project=%s)",
        settings.LANGSMITH_PROJECT,
    )
    return True
