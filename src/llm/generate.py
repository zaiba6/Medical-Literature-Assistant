"""Synthesize retrieval results into an answer using GPT-4."""
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.config import LLM_MODEL, OPENAI_API_KEY


def build_context(
    text_results: list[dict],
    image_results: list[dict],
) -> tuple[str, str]:
    """Build text context and image context strings from retrieval results."""
    text_context = "\n\n".join(
        f"Paper: {r.get('metadata', {}).get('source', r.get('id', '?'))}\n{r.get('text', '')}"
        for r in text_results
    )
    image_context = "\n".join(
        f"Figure: {r.get('metadata', {}).get('path', r.get('id', '?'))} (from paper: {r.get('metadata', {}).get('source_paper', '?')})"
        for r in image_results
    )
    return text_context, image_context


def generate_response(
    query: str,
    text_results: list[dict],
    image_results: list[dict],
    model: str = LLM_MODEL,
    api_key: str | None = None,
) -> str:
    """
    Combine retrieved papers and figures into a coherent answer.
    Returns the model's text response.
    """
    key = api_key or OPENAI_API_KEY
    if not key:
        return "OpenAI API key not set. Set OPENAI_API_KEY in .env to enable LLM answers."
    text_context, image_context = build_context(text_results, image_results)
    prompt = f"""You are a medical research assistant. Answer the user's question using the following context.

TEXT SOURCES (abstracts/methods):
{text_context or '(No text sources retrieved)'}

RELEVANT FIGURES:
{image_context or '(No figures retrieved)'}

User question: {query}

Provide a concise, accurate answer with citations to specific papers and figures where relevant."""

    llm = ChatOpenAI(model=model, temperature=0, api_key=key)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content if hasattr(response, "content") else str(response)
