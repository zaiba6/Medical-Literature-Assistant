"""
Multimodal Medical Literature Assistant — Streamlit UI.
Run: streamlit run app.py
"""
import sys
from pathlib import Path

from PIL import Image
import streamlit as st

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.llm import generate_response
from src.retrieval import process_query

st.set_page_config(page_title="Medical Literature Assistant", layout="wide")
st.title("Multimodal Medical Literature Assistant")
st.caption("Search by text, by image, or both. Get answers with citations to papers and figures.")

query_type = st.radio(
    "Query type",
    ["Text only", "Image only", "Both"],
    horizontal=True,
)

text_query: str | None = None
image_query: Image.Image | None = None

if query_type in ("Text only", "Both"):
    text_query = st.text_input("Your question", placeholder="e.g. What papers discuss theta wave activity in sleep?")
if query_type in ("Image only", "Both"):
    uploaded = st.file_uploader("Reference image (optional)", type=["png", "jpg", "jpeg"])
    if uploaded:
        image_query = Image.open(uploaded).convert("RGB")
        st.image(image_query, caption="Uploaded image", use_container_width=False)

if st.button("Search", type="primary"):
    if not text_query and not image_query:
        st.warning("Enter a text question and/or upload an image.")
    else:
        with st.spinner("Searching literature..."):
            results = process_query(
                query=text_query or None,
                query_image=image_query,
            )
        text_results = results.get("text_results", [])
        image_results = results.get("image_results", [])

        st.subheader("Retrieved papers (text)")
        if text_results:
            for r in text_results[:5]:
                with st.expander(r.get("metadata", {}).get("source", r.get("id", "?"))):
                    st.write(r.get("text", "")[:800] + ("..." if len(r.get("text", "")) > 800 else ""))
        else:
            st.info("No text results. Index papers first (see README).")

        st.subheader("Relevant figures")
        if image_results:
            cols = st.columns(min(3, len(image_results)))
            for i, r in enumerate(image_results[:6]):
                path = r.get("metadata", {}).get("path")
                if path and Path(path).exists():
                    cols[i % 3].image(path, caption=r.get("metadata", {}).get("source_paper", path))
        else:
            st.info("No image results.")

        if text_query and (text_results or image_results):
            st.subheader("Answer")
            answer = generate_response(text_query, text_results, image_results)
            st.write(answer)
        elif not text_query:
            st.info("Add a text question to get an LLM-synthesized answer.")

st.sidebar.header("About")
st.sidebar.markdown(
    "RAG over medical papers using **text** (SPECTER) and **image** (CLIP) embeddings. "
    "Index papers with `python -m scripts.collect_papers` and `python -m scripts.build_index`."
)
