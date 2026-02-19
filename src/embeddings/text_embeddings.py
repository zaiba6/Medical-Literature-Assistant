"""Text embedding pipeline using SPECTER (scientific paper embeddings)."""
from pathlib import Path
from typing import Union

from sentence_transformers import SentenceTransformer

from src.config import PROJECT_ROOT, TEXT_EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def load_text_model(model_name: str = TEXT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Load and cache the text embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(
    texts: list[str],
    model: SentenceTransformer | None = None,
) -> list[list[float]]:
    """Embed a list of texts. Returns list of embedding vectors."""
    if model is None:
        model = load_text_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return [e.tolist() for e in embeddings]


def embed_query_text(
    query: str,
    model: SentenceTransformer | None = None,
) -> list[float]:
    """Embed a single query string."""
    if model is None:
        model = load_text_model()
    return model.encode(query, convert_to_numpy=True).tolist()
