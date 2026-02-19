"""Image embedding pipeline using CLIP."""
import io
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.config import IMAGE_EMBEDDING_MODEL

_model: CLIPModel | None = None
_processor: CLIPProcessor | None = None


def load_image_model(model_name: str = IMAGE_EMBEDDING_MODEL) -> tuple[CLIPModel, CLIPProcessor]:
    """Load and cache CLIP model and processor."""
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(model_name)
        _processor = CLIPProcessor.from_pretrained(model_name)
    return _model, _processor  # type: ignore


def embed_image(
    image: Union[Image.Image, bytes, str, Path],
    model: CLIPModel | None = None,
    processor: CLIPProcessor | None = None,
) -> list[float]:
    """Embed a single image. Accepts PIL Image, bytes, or file path."""
    if model is None or processor is None:
        model, processor = load_image_model()
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features.squeeze(0).numpy().tolist()


def embed_query_image(
    image: Union[Image.Image, bytes, str, Path],
    model: CLIPModel | None = None,
    processor: CLIPProcessor | None = None,
) -> list[float]:
    """Same as embed_image; used for query-side image."""
    return embed_image(image, model=model, processor=processor)


