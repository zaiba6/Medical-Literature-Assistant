"""Fetch paper metadata and PMC PDF links from PubMed."""
from typing import Optional

from Bio import Entrez

from src.config import PUBMED_EMAIL

if PUBMED_EMAIL:
    Entrez.email = PUBMED_EMAIL


def fetch_pubmed_pmids(
    query: str,
    retmax: int = 50,
    db: str = "pubmed",
) -> list[str]:
    """Search PubMed and return list of PMIDs."""
    handle = Entrez.esearch(db=db, term=query, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])


def _pmids_to_pmc_ids(pmids: list[str]) -> dict[str, str]:
    """Use elink to get PMC IDs for PMIDs. Returns {pmid: pmc_id}."""
    if not pmids:
        return {}
    try:
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmids, linkname="pubmed_pmc")
        result = Entrez.read(handle)
        handle.close()
    except Exception:
        return {}
    pmid_to_pmc = {}
    for rec in result:
        id_list = rec.get("IdList", [])
        if not id_list:
            continue
        pmid = str(id_list[0])
        for link_db in rec.get("LinkSetDb", []):
            if link_db.get("DbTo") != "pmc":
                continue
            for link in link_db.get("Link", []):
                pmc_id = link.get("Id")
                if pmc_id:
                    pmc_str = str(pmc_id)
                    if not pmc_str.upper().startswith("PMC"):
                        pmc_str = f"PMC{pmc_str}"
                    pmid_to_pmc[pmid] = pmc_str
                    break
            break
    return pmid_to_pmc


def fetch_pubmed_summaries(pmids: list[str]) -> list[dict]:
    """Fetch title, abstract, and PMC ID for each PMID."""
    if not pmids:
        return []
    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="xml")
    records = Entrez.read(handle)
    handle.close()
    articles = records.get("PubmedArticle", [])
    if not articles and "PubmedArticle" in records:
        articles = [records["PubmedArticle"]]
    if not isinstance(articles, list):
        articles = [articles]
    pmc_map = _pmids_to_pmc_ids(pmids)
    out = []
    for art in articles:
        try:
            med = art["MedlineCitation"]["Article"]
            pmid = str(art["MedlineCitation"]["PMID"])
            title = " ".join(med.get("ArticleTitle", [])) if isinstance(med.get("ArticleTitle"), list) else (med.get("ArticleTitle") or "")
            abstract_nodes = med.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_nodes, str):
                abstract = abstract_nodes
            elif isinstance(abstract_nodes, list):
                abstract = " ".join(
                    n if isinstance(n, str) else n.get("#text", "")
                    for n in abstract_nodes
                )
            else:
                abstract = ""
            pmc_id = pmc_map.get(pmid)
            out.append({
                "pmid": pmid,
                "pmc_id": pmc_id,
                "title": title,
                "abstract": abstract,
            })
        except (KeyError, TypeError):
            continue
    return out


def pmc_id_to_pdf_url(pmc_id: str) -> Optional[str]:
    """Return PMC Open Access PDF URL if available (e.g. PMC1234567 -> https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/pdf/)."""
    if not pmc_id or not pmc_id.upper().startswith("PMC"):
        return None
    clean = pmc_id.upper().replace("PMC", "")
    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{clean}/pdf/"


def fetch_pmc_pdf_links(pmids: list[str]) -> list[dict]:
    """
    For each PMID, get summary; then for those with PMC ID, build PDF link.
    Returns list of {pmid, pmc_id, title, abstract, pdf_url}.
    """
    summaries = fetch_pubmed_summaries(pmids)
    results = []
    for s in summaries:
        pdf_url = pmc_id_to_pdf_url(s.get("pmc_id") or "") if s.get("pmc_id") else None
        results.append({
            **s,
            "pdf_url": pdf_url,
        })
    return results
