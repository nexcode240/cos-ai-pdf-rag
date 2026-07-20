"""Ollama model endpoints."""
import logging
from typing import List

import ollama
from fastapi import APIRouter, HTTPException

from ..models import ModelInfo

router = APIRouter(prefix="/api/v1/models", tags=["models"])
logger = logging.getLogger(__name__)


def is_chat_model(model_name: str, model_size: int) -> bool:
    """Detect if a model supports chat (vs embedding-only)."""
    size_threshold = 1_000_000_000  # 1 GB
    if model_size < size_threshold:
        return False

    try:
        model_info = ollama.show(model_name)

        if hasattr(model_info, "template"):
            template = (
                model_info.template if isinstance(model_info.template, str) else ""
            )
            if template:
                return True

        if hasattr(model_info, "modelfile"):
            modelfile = (
                model_info.modelfile if isinstance(model_info.modelfile, str) else ""
            )
            if "embed" in modelfile.lower():
                return False
    except Exception as e:
        logger.debug("Could not get detailed info for %s: %s", model_name, e)

    embedding_indicators = [
        "embed",
        "embedding",
        "bge",
        "e5",
        "sentence",
        "mpnet",
        "minilm",
        "retrieval",
    ]
    model_lower = model_name.lower()
    for indicator in embedding_indicators:
        if indicator in model_lower:
            return False

    return True


@router.get("", response_model=List[ModelInfo])
def list_models():
    """List available Ollama chat models (excludes embedding models)."""
    try:
        models_info = ollama.list()
        chat_models = []

        for model in models_info.models:
            model_dict = model.model_dump()
            model_name = model_dict.get("model", "")
            model_size = model_dict.get("size", 0)

            if is_chat_model(model_name, model_size):
                chat_models.append(
                    ModelInfo(
                        name=model_name,
                        size=model_size,
                        modified_at=str(model_dict.get("modified_at", "")),
                    )
                )

        if not chat_models:
            raise HTTPException(
                status_code=404,
                detail="No chat models found. Install one with: ollama pull llama3.2",
            )

        return chat_models
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch models: {e!s}"
        ) from e
