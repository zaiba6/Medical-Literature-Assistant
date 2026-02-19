"""ChromaDB collections for text and image embeddings."""
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DIR, DEFAULT_TOP_K_IMAGES, DEFAULT_TOP_K_TEXT

TEXT_COLLECTION_NAME = "medical_papers"
IMAGE_COLLECTION_NAME = "medical_images"

_client: chromadb.PersistentClient | None = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    return _client


def get_or_create_collections():
    """Get or create text and image collections."""
    client = get_client()
    text_coll = client.get_or_create_collection(TEXT_COLLECTION_NAME, metadata={"description": "Paper abstracts"})
    image_coll = client.get_or_create_collection(IMAGE_COLLECTION_NAME, metadata={"description": "Paper figures"})
    return text_coll, image_coll


def add_papers_to_store(
    ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]] | None = None,
) -> None:
    """Upsert papers into the text collection."""
    text_coll, _ = get_or_create_collections()
    if metadatas is None:
        metadatas = [{}] * len(ids)
    text_coll.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)


def add_images_to_store(
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    """Upsert image embeddings (no documents)."""
    _, image_coll = get_or_create_collections()
    image_coll.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
