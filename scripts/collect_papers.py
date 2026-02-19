"""
Collect papers: search PubMed, download PMC PDFs, extract text and figures.
Run from project root: python -m scripts.collect_papers
"""
import sys
import time
from pathlib import Path

import requests

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, FIGURES_DIR, PAPERS_DIR
from src.data_collection.pdf_extract import extract_abstract_and_methods, extract_images_from_pdf
from src.data_collection.pubmed import fetch_pmc_pdf_links, fetch_pubmed_pmids


def download_pdf(url: str, path: Path) -> bool:
    """Download PDF from URL to path. Returns True on success."""
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "MedicalLiteratureAssistant/1.0"})
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        return True
    except Exception:
        return False


def main():
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Example queries for neuroscience/EEG
    queries = [
        "EEG theta waves",
        "seizure detection EEG",
        "brain imaging fMRI",
    ]
    all_pmids = []
    for q in queries:
        pmids = fetch_pubmed_pmids(q, retmax=25)
        all_pmids.extend(pmids)
        time.sleep(0.4)  # rate limit

    unique_pmids = list(dict.fromkeys(all_pmids))[:75]  # cap ~75 papers
    print(f"Found {len(unique_pmids)} unique PMIDs")

    summaries = fetch_pmc_pdf_links(unique_pmids)
    with_pdf = [s for s in summaries if s.get("pdf_url")]
    print(f"Of those, {len(with_pdf)} have PMC PDF links")

    for i, meta in enumerate(with_pdf):
        pdf_url = meta["pdf_url"]
        pmc = (meta.get("pmc_id") or "").replace("PMC", "")
        fname = f"PMC{pmc}.pdf"
        out_path = PAPERS_DIR / fname
        if out_path.exists():
            print(f"Skip (exists): {fname}")
            continue
        print(f"Downloading {i+1}/{len(with_pdf)}: {fname}")
        if download_pdf(pdf_url, out_path):
            time.sleep(0.5)
        else:
            print(f"  Failed: {pdf_url}")

    # Extract text and figures from downloaded PDFs
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    print(f"\nExtracting text and figures from {len(pdf_files)} PDFs...")
    for pdf_path in pdf_files:
        stem = pdf_path.stem
        text = extract_abstract_and_methods(pdf_path)
        # Save abstract for embedding (we'll also use it in retrieval)
        abstract_path = PAPERS_DIR / f"{stem}_abstract.txt"
        if text.get("abstract"):
            abstract_path.write_text(text["abstract"], encoding="utf-8")
        fig_dir = FIGURES_DIR / stem
        extract_images_from_pdf(pdf_path, output_dir=fig_dir)
    print("Done.")


if __name__ == "__main__":
    main()
