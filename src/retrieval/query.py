"""Dual retrieval: text + image by query (text and/or image)."""
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from src.config import DEFAULT_TOP_K_IMAGES, DEFAULT_TOP_K_TEXT
from src.embeddings import embed_query_image, embed_query_text
from src.retrieval.store import get_or_create_collections


def process_query(
    query: Optional[str] = None,
    query_image: Optional[Image.Image] = None,
    top_k_text: int = DEFAULT_TOP_K_TEXT,
    top_k_images: int = DEFAULT_TOP_K_IMAGES,
) -> dict[str, Any]:
    """
    Run text and/or image retrieval.
    Returns {"text_results": [...], "image_results": [...]}.
    """
    text_coll, image_coll = get_or_create_collections()
    results: dict[str, Any] = {"text_results": [], "image_results": []}

    if query and query.strip():
        query_embedding = embed_query_text(query.strip())
        text_results = text_coll.query(
            query_embeddings=[query_embedding],
            n_results=top_k_text,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma returns dict with lists: ids[0], documents[0], metadatas[0]
        if text_results["ids"] and text_results["ids"][0]:
            results["text_results"] = [
                {
                    "id": id_,
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                }
                for id_, doc, meta, dist in zip(
                    text_results["ids"][0],
                    text_results["documents"][0],
                    text_results["metadatas"][0],
                    text_results["distances"][0],
                )
            ]

    if query_image is not None:
        img_embedding = embed_query_image(query_image)
        image_results = image_coll.query(
            query_embeddings=[img_embedding],
            n_results=top_k_images,
            include=["metadatas", "distances"],
        )
        if image_results["ids"] and image_results["ids"][0]:
            results["image_results"] = [
                {
                    "id": id_,
                    "metadata": meta,
                    "distance": dist,
                }
                for id_, meta, dist in zip(
                    image_results["ids"][0],
                    image_results["metadatas"][0],
                    image_results["distances"][0],
                )
            ]

    return results
