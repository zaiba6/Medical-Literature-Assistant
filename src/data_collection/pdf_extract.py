"""Extract text (abstract, methods) and images from PDFs."""
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber


def extract_abstract_and_methods(pdf_path: str | Path) -> dict[str, Optional[str]]:
    """
    Extract abstract and methods section text from a PDF.
    Returns {"abstract": str | None, "methods": str | None}.
    """
    path = Path(pdf_path)
    if not path.exists():
        return {"abstract": None, "methods": None}

    full_text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except Exception:
        full_text = ""
        with fitz.open(path) as doc:
            for page in doc:
                full_text += page.get_text() + "\n"

    abstract = _extract_section(full_text, "abstract")
    methods = _extract_section(full_text, "methods")
    return {"abstract": abstract or None, "methods": methods or None}


def _extract_section(text: str, section: str) -> Optional[str]:
    """Naive section extraction by common headings."""
    lower = text.lower()
    markers = {
        "abstract": ["abstract", "summary"],
        "methods": ["methods", "methodology", "materials and methods", "materials & methods"],
    }
    start_markers = markers.get(section, [])
    start_idx = -1
    for m in start_markers:
        idx = lower.find(m)
        if idx != -1:
            start_idx = idx
            break
    if start_idx == -1:
        return None
    # End at next common section
    end_markers = [
        "introduction", "background", "results", "discussion",
        "references", "acknowledgment", "conflict of interest",
    ]
    rest = lower[start_idx:]
    end_idx = len(rest)
    for em in end_markers:
        pos = rest.find(em, 20)
        if pos != -1 and pos < end_idx:
            end_idx = pos
    return text[start_idx : start_idx + end_idx].strip()


def extract_images_from_pdf(
    pdf_path: str | Path,
    output_dir: Optional[str | Path] = None,
    min_width: int = 100,
    min_height: int = 100,
) -> list[dict]:
    """
    Extract embedded images from PDF. Optionally save to output_dir.
    Returns list of {"image": bytes, "page": int, "index": int, "path": str | None}.
    """
    path = Path(pdf_path)
    if not path.exists():
        return []

    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    doc = fitz.open(path)
    stem = path.stem

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base = doc.extract_image(xref)
                except Exception:
                    continue
                w, h = base.get("width", 0), base.get("height", 0)
                if w < min_width or h < min_height:
                    continue
                img_bytes = base["image"]
                ext = base.get("ext", "png")
                save_path = None
                if out_dir:
                    save_path = out_dir / f"{stem}_p{page_num + 1}_i{img_index}.{ext}"
                    save_path.write_bytes(img_bytes)
                    save_path = str(save_path)
                results.append({
                    "image": img_bytes,
                    "page": page_num + 1,
                    "index": img_index,
                    "path": save_path,
                    "source_paper": stem,
                })
    finally:
        doc.close()

    return results
