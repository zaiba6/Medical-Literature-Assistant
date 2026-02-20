# Multimodal Medical Literature Assistant

RAG system that answers questions about medical/neuroscience papers using **both text** (abstracts, methods) and **images** (figures, diagrams). Query by text, by reference image, or both—e.g. *"Show papers about theta waves with similar EEG patterns."*

## Architecture

```
User Query → Query Processing → Dual Retrieval → LLM (GPT-4) → Response
                   ↓                    ↓
               Text Search         Image Search
               (SPECTER)           (CLIP)
                   ↓                    ↓
               ChromaDB             ChromaDB
```

## Tech Stack

| Component        | Choice                          |
|-----------------|----------------------------------|
| Data collection | PubMed API, PyMuPDF, pdfplumber |
| Text embeddings | sentence-transformers (SPECTER) |
| Image embeddings| CLIP (transformers)             |
| Vector DB       | ChromaDB                        |
| LLM             | OpenAI GPT-4 (LangChain)        |
| UI              | Streamlit                       |

## Setup

1. **Clone and install**

   ```bash
   cd Multimodal-Medical-Literature-Assistant-
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Environment**

   Copy `.env.example` to `.env` and set:

   - `PUBMED_EMAIL` — your email (required for PubMed API).
   - `OPENAI_API_KEY` — for GPT-4 answers (get from [OpenAI](https://platform.openai.com/api-keys)).

3. **Collect papers** (downloads PMC PDFs, extracts text and figures)

   ```bash
   python -m scripts.collect_papers
   ```

4. **Build index** (embed papers and figures into ChromaDB)

   ```bash
   python -m scripts.build_index
   ```

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

## Usage

- **Text only:** e.g. *"What papers discuss theta wave activity in sleep?"*
- **Image only:** upload an EEG/figure to find papers with similar figures.
- **Both:** text question + reference image for combined retrieval and a synthesized answer with citations.

## Project layout

```
├── app.py                 # Streamlit UI
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py         # Paths, API keys, model names
│   ├── data_collection/  # PubMed fetch, PDF text/figure extraction
│   ├── embeddings/       # SPECTER (text), CLIP (image)
│   ├── retrieval/       # Chroma store + dual query
│   └── llm/              # LangChain + GPT-4 response
├── scripts/
│   ├── collect_papers.py # Download papers, extract text & figures
│   ├── build_index.py    # Embed and index into Chroma
│   └── evaluate.py       # Precision/recall on test queries
└── data/                 # papers/, figures/, chroma/ (gitignored)
```

## Evaluation

Edit `scripts/evaluate.py`: set `expected_papers` for each test query (paper IDs that should appear in top results). Then:

```bash
python -m scripts.evaluate
```

## License

MIT.
