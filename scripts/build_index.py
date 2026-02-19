"""
Build Chroma index from collected papers and figures.
Run from project root: python -m scripts.build_index
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FIGURES_DIR, PAPERS_DIR
from src.data_collection.pdf_extract import extract_abstract_and_methods, extract_images_from_pdf
from src.embeddings import embed_image, embed_texts, load_image_model, load_text_model
from src.retrieval import add_images_to_store, add_papers_to_store, get_or_create_collections


def main():
    get_or_create_collections()

    # Text: use saved abstracts or extract from PDF
    paper_ids = []
    paper_texts = []
    for abstract_file in sorted(PAPERS_DIR.glob("*_abstract.txt")):
        text = abstract_file.read_text(encoding="utf-8")
        stem = abstract_file.stem.replace("_abstract", "")
        paper_ids.append(stem)
        paper_texts.append(text)
    for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
        stem = pdf_path.stem
        if stem in paper_ids:
            continue
        extracted = extract_abstract_and_methods(pdf_path)
        text = (extracted.get("abstract") or "") + "\n" + (extracted.get("methods") or "")
        if not text.strip():
            continue
        paper_ids.append(stem)
        paper_texts.append(text)

    if not paper_ids:
        print("No papers found. Run scripts/collect_papers.py first.")
        return
    print(f"Embedding {len(paper_ids)} papers...")
    model = load_text_model()
    embeddings = embed_texts(paper_texts, model=model)
    metadatas = [{"source": pid} for pid in paper_ids]
    add_papers_to_store(paper_ids, paper_texts, embeddings, metadatas)
    print("Text index done.")

    # Images: embed each figure file
    load_image_model()
    img_ids = []
    img_embeddings = []
    img_metadatas = []
    for paper_dir in sorted(FIGURES_DIR.iterdir()):
        if not paper_dir.is_dir():
            continue
        source_paper = paper_dir.name
        for img_path in sorted(paper_dir.glob("*")):
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            try:
                emb = embed_image(str(img_path))
                img_ids.append(str(img_path))
                img_embeddings.append(emb)
                img_metadatas.append({"path": str(img_path), "source_paper": source_paper})
            except Exception as e:
                print(f"Skip {img_path}: {e}")
    if img_ids:
        print(f"Embedding {len(img_ids)} figures...")
        add_images_to_store(img_ids, img_embeddings, img_metadatas)
        print("Image index done.")
    else:
        print("No figures found under data/figures/.")


if __name__ == "__main__":
    main()
