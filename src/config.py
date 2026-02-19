"""Configuration loaded from environment."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths (project root = parent of src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
FIGURES_DIR = DATA_DIR / "figures"
CHROMA_DIR = os.environ.get("CHROMA_PERSIST_DIR") or str(DATA_DIR / "chroma")

# PubMed
PUBMED_EMAIL = os.environ.get("PUBMED_EMAIL", "")

# OpenAI (for LLM)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Model names
TEXT_EMBEDDING_MODEL = "allenai/specter"
IMAGE_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"
LLM_MODEL = "gpt-4"

# Retrieval
DEFAULT_TOP_K_TEXT = 5
DEFAULT_TOP_K_IMAGES = 5
