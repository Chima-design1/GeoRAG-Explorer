# GeoRAG Explorer

**AI-powered geological knowledge retrieval system combining geological reports, maps, geochemical data, semantic search, and Retrieval-Augmented Generation.**

## Overview

GeoRAG Explorer is a professional end-to-end geological Retrieval-Augmented Generation (RAG) system designed to answer complex questions about geological resources, mineral occurrences, geochemical anomalies, and geological formations using a corpus of geological documents, reports, and thematic maps.

The system retrieves evidence from authoritative geological sources and generates grounded answers with full source traceability—essential for geological and mineral-resource applications where accuracy and provenance are critical.

## Problem Statement

Geologists, mineral explorers, and resource professionals need rapid, accurate access to geological information distributed across:

- Geological survey reports
- Geochemical maps and anomaly data
- Mineral-resource assessments
- State and regional geological documentation
- Mineral-corridor and schist-belt mapping

Traditional keyword search is insufficient for complex geological queries. GeoRAG Explorer uses semantic retrieval combined with grounded LLM generation to answer questions like:

- *What minerals occur in Ogun State?*
- *Where are copper anomalies reported?*
- *What geological units are associated with lithium occurrences?*
- *What are the strongest geochemical anomalies?*

All answers include **verified sources and precise citations**.

## Architecture

```
                     User Question
                           │
                ┌──────────┴──────────┐
                │                     │
         Document Corpus         Configuration
                │                     │
                └──────────┬──────────┘
                           │
                    Document Loading
                    & Cleaning
                           │
                    Chunk with Metadata
                           │
                    Generate Embeddings
                    (OpenAI API)
                           │
                   Cache Embeddings
                    (Local Storage)
                           │
                    Vector Retrieval
                    (Cosine Similarity)
                           │
                    Top-K Evidence
                           │
                    LLM Generation
                    (OpenAI API)
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
    Grounded Answer                   Source Citations
    (with disclaimer)              (filename, title, URL)
```

### Phase 1 — Text Document Pipeline

**Phase 1 implements the complete text-based retrieval pipeline:**

```
Geological Reports (TXT)
         ↓
Document Loading
         ↓
Text Cleaning
         ↓
Metadata Extraction
         ↓
Chunking (with metadata preservation)
         ↓
OpenAI Embeddings (cached locally)
         ↓
Vector Index (NumPy + cosine similarity)
         ↓
Retrieval & Ranking
         ↓
LLM Generation (grounded with sources)
```

### Future Phases

- **Phase 2**: PDF geological report extraction
- **Phase 3**: Geological map processing (inspection, OCR, metadata extraction)
- **Phase 4**: Hybrid retrieval (semantic + BM25 + metadata filtering)
- **Phase 5**: Reranking
- **Phase 6**: Evaluation against geological test questions
- **Phase 7**: Portfolio presentation (architecture diagrams, screenshots, evaluation results)

## Data Sources

### Current (Phase 1)

- **Geological Reports**: Text documents from publicly available Nigerian Geological Survey Agency (NGSA) reports
- **Location**: `data/reports/`
- **Format**: UTF-8 text files (`.txt`)

### Planned (Future Phases)

- Geological PDF reports
- Geological maps (national, state-level, geochemical, mineral-corridor, schist-belt)
- Mineral-resource maps
- Geochemical anomaly maps
- Additional geological surveys and institutions (extensible architecture)

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key (for embeddings and LLM generation)

### Setup

```bash
# Clone the repository
git clone https://github.com/Chima-design1/GeoRAG-Explorer.git
cd GeoRAG-Explorer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (defaults provided)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4-turbo
LOG_LEVEL=INFO
```

## Usage

### Basic Retrieval-Augmented Generation

```python
from src.config import Config
from src.document_loader import DocumentLoader
from src.chunker import Chunker
from src.embeddings import EmbeddingGenerator
from src.retriever import VectorRetriever
from src.rag import RAG

# Initialize config
config = Config()

# Load and process documents
loader = DocumentLoader(config.reports_dir)
documents = loader.load_all()

# Chunk documents
chunker = Chunker()
chunks = chunker.chunk_documents(documents)

# Generate embeddings (cached after first run)
embedding_gen = EmbeddingGenerator(config)
embeddings = embedding_gen.embed_chunks(chunks, cache_path="artifacts/embeddings.pkl")

# Initialize retriever
retriever = VectorRetriever(chunks, embeddings)

# Create RAG pipeline
rag = RAG(retriever, config)

# Ask a geological question
answer = rag.query("What minerals occur in Ogun State?", top_k=5)

print("Answer:", answer["answer"])
print("\nSources:")
for source in answer["sources"]:
    print(f"  - {source['title']} ({source['source_file']})")
```

### Jupyter Notebooks

Start with the provided notebooks:

- **01_data_exploration.ipynb** – Explore loaded reports
- **02_document_processing.ipynb** – Test chunking and embedding quality
- **03_retrieval_evaluation.ipynb** – Evaluate retrieval performance
- **04_rag_evaluation.ipynb** – Full pipeline evaluation

```bash
jupyter notebook notebooks/
```

## Project Structure

```
GeoRAG-Explorer/
│
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── .env.example                        # Example environment configuration
├── .gitignore                          # Git ignore rules
│
├── data/
│   ├── reports/                        # Geological text reports
│   ├── maps/                           # (Phase 3) Geological maps
│   │   ├── national/
│   │   ├── state/
│   │   ├── geochemical/
│   │   ├── corridors/
│   │   ├── schist_belts/
│   │   └── geological/
│   └── test_questions.csv              # (Phase 6) Geological evaluation questions
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Configuration management
│   ├── logger.py                       # Logging utilities
│   ├── document_loader.py              # Load TXT reports
│   ├── pdf_processor.py                # (Phase 2) Extract text from PDFs
│   ├── map_processor.py                # (Phase 3) Process geological maps
│   ├── chunker.py                      # Chunk documents with metadata
│   ├── embeddings.py                   # Embedding generation & caching
│   ├── retriever.py                    # Vector retrieval (semantic search)
│   ├── reranker.py                     # (Phase 5) Reranking
│   ├── rag.py                          # RAG pipeline (retrieve + generate)
│   ├── scraper.py                      # (Optional) Scrape geological reports
│   └── evaluation.py                   # Evaluation metrics
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Explore reports
│   ├── 02_document_processing.ipynb    # Test processing pipeline
│   ├── 03_retrieval_evaluation.ipynb   # Retrieval metrics
│   └── 04_rag_evaluation.ipynb         # Full RAG evaluation
│
├── assets/
│   ├── architecture.png                # (Phase 7) Architecture diagram
│   ├── rag-demo.png                    # (Phase 7) Example Q&A
│   └── evaluation.png                  # (Phase 7) Evaluation results
│
├── artifacts/
│   ├── embeddings.pkl                  # Cached embeddings
│   └── evaluation_results.json         # Evaluation metrics
│
└── tests/
    ├── __init__.py
    ├── test_chunker.py
    ├── test_embeddings.py
    ├── test_retriever.py
    └── test_rag.py
```

## Key Features

### Metadata Preservation

Every retrieved chunk retains:
- `document_id` — Unique document identifier
- `chunk_id` — Unique chunk identifier
- `title` — Document title
- `source_file` — Original filename
- `source_url` — Original URL (if available)
- `document_type` — e.g., "geological_report", "map", "assessment"
- Additional domain fields (location, commodity, geological_unit, etc.)

### Grounded Generation

The LLM generation prompt explicitly instructs the model to:

1. Use only retrieved evidence
2. Never invent geological facts
3. State when evidence is insufficient
4. Cite specific sources
5. Distinguish explicit information from inference
6. Preserve geological terminology

### Embeddings Caching

Embeddings are cached locally after generation to avoid re-embedding and excessive API calls:

```python
embeddings = embedding_gen.embed_chunks(chunks, cache_path="artifacts/embeddings.pkl")
```

Subsequent runs load from cache.

### Configurable Retrieval

```python
retriever = VectorRetriever(chunks, embeddings, top_k=5)
answer = rag.query(question, top_k=10)  # Override at query time
```

### Error Handling & Logging

All modules include:
- Type hints
- Docstrings
- Error handling with clear messages
- Structured logging (INFO, DEBUG, WARNING, ERROR)

## Example Queries

The system is designed to handle geological questions such as:

```
What minerals occur in Ogun State?
Where are copper anomalies reported?
What geological units are associated with lithium occurrences?
Which Nigerian regions contain a particular mineral?
What are the strongest geochemical anomalies?
What geological formations occur in a particular area?
Which maps contain information about a particular commodity?
What mineral resources are associated with a particular geological unit?
What geological information is available for a particular state?
```

All answers include source citations and confidence disclaimers.

## Evaluation

### Phase 6 Evaluation

When `data/test_questions.csv` is provided, run:

```python
from src.evaluation import evaluate_rag

results = evaluate_rag(
    rag_pipeline=rag,
    questions_csv="data/test_questions.csv",
    output_path="artifacts/evaluation_results.json"
)

print(results.summary())
```

Metrics include:
- `Recall@K` (K=1, 3, 5)
- Mean Reciprocal Rank (MRR)
- Answer accuracy (if reference answers available)
- Source citation accuracy

### Current Status

**Phase 1**: Complete text-based pipeline.  
**Evaluation**: Awaiting test question CSV. The evaluation module is ready; provide `data/test_questions.csv` to run full evaluation.

## Limitations

### Phase 1

- **Text only** — Maps and geochemical PDFs not yet processed
- **Local vector index** — No persistent database; embeddings regenerated on startup
- **Single retrieval method** — Semantic search only; no BM25 or metadata filtering
- **No reranking** — Results ranked by cosine similarity only

### Data

- Geological information sourced from provided reports only
- No hard-coded geological knowledge
- System will return "insufficient evidence" rather than rely on background knowledge
- Map processing deferred to Phase 3 (after geological PDFs provided)

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Text-based RAG baseline | ✅ Complete |
| 2 | PDF report extraction | ⏳ Planned |
| 3 | Geological map processing | ⏳ Planned (after PDFs provided) |
| 4 | Hybrid retrieval (semantic + BM25 + metadata) | ⏳ Planned |
| 5 | Reranking | ⏳ Planned |
| 6 | Evaluation against test questions | ⏳ Planned (after test CSV provided) |
| 7 | Portfolio presentation | ⏳ Planned |

## Contributing

Geological expertise welcome. When providing:

- **PDF reports**: Will be integrated via Phase 2 PDF processor
- **Geological maps**: Will be analyzed for content and structure; map-specific processing designed based on actual PDF contents
- **Test questions**: Will drive Phase 6 evaluation and benchmark improvements

## Citation & Attribution

This project uses:
- Geological data from publicly available sources (primarily NGSA)
- OpenAI API for embeddings and LLM generation
- Open-source Python libraries (see requirements.txt)

When using this system for research or publication, cite:
1. Original geological data sources
2. This repository
3. OpenAI models used

## License

MIT License — see LICENSE file.

## Support

For questions, issues, or contributions, please open a GitHub issue.

---

**Built with geology expertise, modular Python design, and rigorous source grounding.**
