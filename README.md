# GeoRAG Explorer

**AI-powered geological knowledge retrieval system combining geological reports, geological maps, semantic search, lexical retrieval, reranking, and Retrieval-Augmented Generation (RAG).**

## Overview

GeoRAG Explorer is an end-to-end geological Retrieval-Augmented Generation system designed to answer questions about geological resources, mineral occurrences, geochemical information, geological formations, exploration targets, and mining-related information using a corpus of geological reports and thematic maps.

The system is designed around **grounded evidence retrieval**: answers are generated from retrieved geological evidence rather than relying on unsupported background knowledge. The pipeline also preserves document and map metadata to support source traceability.

## Current Project Status

The project has progressed substantially beyond the original text-only baseline.

- **2 geological source documents** currently loaded
- **551 geological report chunks**
- **189 geological map chunks**
- **740 combined chunks** in the retrieval corpus
- Local sentence-transformer embeddings with `all-MiniLM-L6-v2`
- Hybrid semantic + lexical retrieval
- Candidate-pool retrieval and feature-based reranking
- Diversity control and local context expansion
- Factual-evidence strengthening for detail-heavy questions
- Local `Qwen/Qwen2.5-0.5B-Instruct` answer generation
- Automated evaluation against **30 test questions**
- **30/30 questions successfully processed** in the latest evaluation

### Latest Evaluation Result

| Metric | Stage 4.5 Result |
|---|---:|
| Questions evaluated | 30 |
| Successful questions | 30 |
| Average retrieval | 1.1298 |
| Average lexical | 0.5504 |
| Average coverage | 0.6676 |
| Average keyword | 0.6917 |
| Average factual match | 0.2556 |
| Average direct match | 0.3000 |
| **Average answer quality** | **0.6558** |

**Answer quality distribution:** 10 Excellent, 9 Good, 7 Fair, 4 Poor.

## Problem Statement

Geologists, mineral explorers, researchers, and resource professionals often need rapid access to geological information distributed across:

- Geological survey reports
- Geological and geochemical maps
- Mineral-resource assessments
- State and regional geological documentation
- Mineral-corridor and schist-belt mapping
- Exploration-target documentation

Traditional keyword search can struggle with complex geological questions that require combining terminology, locations, numbers, sections, and contextual evidence. GeoRAG Explorer addresses this with semantic retrieval, lexical matching, reranking, context expansion, and grounded generation.

## Architecture

```text
                         User Question
                              │
                    ┌─────────┴─────────┐
                    │                   │
             Query Embedding      Query Analysis
                    │                   │
                    └─────────┬─────────┘
                              │
                    Hybrid Candidate Retrieval
                    ┌─────────┴─────────┐
                    │                   │
              Semantic Search      Lexical Search
                    │                   │
                    └─────────┬─────────┘
                              │
                       Candidate Pool
                              │
                     Feature-based Reranking
                              │
                       Diversity Control
                              │
                       Context Expansion
                              │
                  Factual Evidence Strengthening
                              │
                     Grounded Context
                              │
                   Local Qwen Generation
                              │
                  ┌───────────┴───────────┐
                  │                       │
             Grounded Answer         Source Evidence
```

## Data Processing Pipeline

### Geological Reports

```text
PDF / TXT Geological Reports
          ↓
Document Loading
          ↓
Text Extraction & Cleaning
          ↓
Metadata Preservation
          ↓
Chunking with Metadata
          ↓
Local Embeddings
          ↓
Cached Combined Embedding Index
```

### Geological Maps

```text
Geological Map PDFs
          ↓
PDF / Vector / Image Inspection
          ↓
Text Extraction
          ↓
Embedded Image Extraction
          ↓
High-resolution Rendering
          ↓
OCR for Image-based Maps
          ↓
Map Text Cleaning
          ↓
Map Chunk Construction
          ↓
Combined Retrieval Corpus
```

The current map corpus contains **29 processed geological map PDFs**, including corridor, schist-belt, state, national, and regional map material.

## Retrieval and Reranking

The current retrieval pipeline has evolved through several controlled stages.

### Stage 4.1 — Candidate Pool

Retrieval first creates a larger candidate pool before selecting the final evidence set. This gives later ranking stages more relevant candidates to work with.

### Stage 4.2 — Feature-based Reranking

Candidate results are reranked using the original combined retrieval score together with additional lexical and domain-specific signals.

### Stage 4.3 — Diversity Control

The final evidence set reduces unnecessary near-duplicate chunks so that multiple retrieved results can provide complementary evidence.

### Stage 4.4 — Context Expansion

Nearby chunks from the same document are considered to preserve local context around highly relevant evidence while avoiding obvious table-of-contents entries.

### Stage 4.5 — Factual-Evidence Strengthening

The retriever now gives additional weight to evidence containing factual details relevant to the question, including:

- Explicit numbers and percentages
- Section identifiers
- Target Uranium identifiers
- Factual query terms such as location, depth, grade, results, and recommendations

This stage improved the latest average answer-quality score from **0.6327 to 0.6558**.

## Grounded Generation

The system uses the local instruction model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The generation pipeline is designed to:

1. Use retrieved geological evidence
2. Avoid unsupported geological claims
3. State when available evidence is insufficient
4. Preserve important geological terminology
5. Prefer explicit evidence over unsupported inference
6. Provide source information alongside the generated answer

For high-risk factual questions, the pipeline also includes deterministic extraction logic for information such as numerical values, locations, map-selected states, financing, project purposes, results, and recommendations.

## Evaluation

The project includes an automated evaluation pipeline in `src/evaluation.py` using the geological test-question dataset at `data/test_questions.csv`.

Run the full evaluation with:

```bash
python -m src.evaluation
```

Results are written to:

```text
artifacts/evaluation_results.json
```

The evaluation tracks retrieval and answer-oriented metrics including retrieval score, lexical relevance, evidence coverage, keyword matching, factual matching, direct matching, and overall answer quality.

### Latest Benchmark

The current Stage 4.5 benchmark contains **30 questions**, with all 30 successfully processed.

```text
Average answer quality: 0.6558

Excellent: 10
Good:      9
Fair:      7
Poor:      4
```

## Tests

The current automated regression test suite contains **20 passing tests**.

Run:

```bash
python -m pytest -q
```

Expected result for the current checkpoint:

```text
20 passed
```

## Installation

### Prerequisites

- Python 3.9+
- Windows, Linux, or macOS
- Sufficient local storage for geological PDFs, OCR outputs, and embedding caches

The current working configuration uses local embedding and language models, so an OpenAI API key is **not required for the current local RAG pipeline**.

### Setup

```bash
# Clone the repository
git clone https://github.com/Chima-design1/GeoRAG-Explorer.git
cd GeoRAG-Explorer

# Create virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Usage

The main end-to-end pipeline is implemented in `src/rag_pipeline.py`.

A typical query follows this workflow:

```python
from src.config import Config
from src.rag_pipeline import RAGPipeline

config = Config()
rag = RAGPipeline(config)

result = rag.query(
    "What minerals occur in Ogun State?",
    top_k=5,
)

print("Answer:", result["answer"])

print("Sources:")
for source in result["sources"]:
    print(source)
```

## Project Structure

```text
GeoRAG-Explorer/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── reports/                    # Geological reports
│   ├── maps/                       # Geological map corpus
│   │   ├── national/
│   │   ├── state/
│   │   ├── corridors/
│   │   └── schist_belts/
│   └── test_questions.csv          # Evaluation questions
│
├── src/
│   ├── config.py                   # Configuration management
│   ├── document_loader.py          # TXT/PDF document loading
│   ├── pdf_processor.py            # PDF extraction
│   ├── map_processor.py            # Geological map processing/OCR
│   ├── map_corpus.py               # Map cleaning/chunking/indexing
│   ├── chunker.py                  # Metadata-preserving chunking
│   ├── embeddings.py               # Local embeddings and caching
│   ├── retriever.py                # Hybrid retrieval/reranking
│   ├── rag_pipeline.py             # End-to-end RAG pipeline
│   ├── evaluation.py               # Automated evaluation
│   └── logger.py                   # Logging utilities
│
├── artifacts/
│   ├── combined_embeddings.pkl     # Cached combined embeddings
│   ├── map_inventory.json          # Processed map inventory
│   ├── map_ocr/                    # Map OCR outputs
│   └── evaluation_results.json     # Latest evaluation results
│
└── tests/
    ├── test_chunker.py
    ├── test_embeddings.py
    ├── test_retriever.py
    └── test_rag.py
```

## Key Features

### Multi-source Geological Corpus

The retrieval corpus combines geological reports with processed geological maps rather than relying on text reports alone.

### Metadata Preservation

Retrieved chunks retain metadata such as:

- `document_id`
- `chunk_id`
- `chunk_index`
- `title`
- `source_file`
- `document_type`
- `page_number` when available
- Map-specific metadata when available

### Local Embedding Cache

Embeddings are cached in:

```text
artifacts/combined_embeddings.pkl
```

This avoids recomputing embeddings every time the pipeline is initialized when the cached corpus remains valid.

### Hybrid Retrieval

The retriever combines semantic similarity with lexical relevance and domain-specific signals. Candidate results are subsequently reranked and diversified before context expansion.

### Domain-aware Retrieval

Special handling exists for geological structures and high-value evidence such as Target Uranium sections and the Criteria Catalogue.

### Map OCR

Image-based geological maps are rendered at higher resolution and processed with OCR so that map labels and legends can become searchable evidence.

### Reproducible Evaluation

The project includes a test-question evaluation pipeline and a 20-test regression suite so retrieval changes can be checked before and after modifications.

## Example Questions

The system is designed to handle questions such as:

```text
What minerals occur in Ogun State?
Where are copper anomalies reported?
Which states are shown along the copper corridor in Nigeria?
What financing was secured for implementation of the MinDiver Project?
What geological units are associated with lithium occurrences?
What are the strongest geochemical anomalies?
What further investigation was recommended?
What geological formations occur in a particular area?
Which maps contain information about a particular commodity?
What mineral resources are associated with a particular geological unit?
```

## Roadmap

| Stage | Focus | Status |
|---|---|---|
| 1 | Text-based RAG baseline | ✅ Complete |
| 2 | PDF geological report extraction | ✅ Complete |
| 3 | Geological map processing and OCR | ✅ Complete |
| 4.1 | Candidate-pool retrieval | ✅ Complete |
| 4.2 | Feature-based reranking | ✅ Complete |
| 4.3 | Diversity control | ✅ Complete |
| 4.4 | Context expansion | ✅ Complete |
| 4.5 | Factual-evidence strengthening | ✅ Complete |
| 5 | Systematic evaluation and targeted improvement | 🔄 In progress |
| 6 | Portfolio presentation and final documentation | ⏳ Planned |

## Current Next Steps

Stage 5 focuses on systematic analysis of the evaluation results rather than adding retrieval features without evidence. The next work includes:

1. Identify the remaining Fair and Poor answers
2. Separate retrieval failures from generation failures
3. Inspect weak-question evidence coverage
4. Apply targeted improvements only where justified
5. Re-run the 30-question benchmark
6. Lock the best-performing configuration
7. Prepare the project for portfolio presentation

## Limitations

- The current corpus is limited to the geological documents and maps available to the project.
- OCR on dense geological maps can contain recognition noise.
- The current local language model is relatively small and may produce weaker answers for complex synthesis questions.
- The vector index is currently maintained through local cached embeddings rather than a dedicated production vector database.
- Evaluation quality depends on the coverage and quality of the provided reference questions and answers.

## Data and Attribution

The project uses geological material supplied for the project, including publicly available Nigerian geological information and processed geological reports/maps.

When using this system for research or publication, cite the original geological data sources and the GeoRAG Explorer repository.

## License

MIT License — see `LICENSE` for details.

---

**GeoRAG Explorer — grounded geological information retrieval with local RAG, hybrid search, map OCR, and evidence-focused evaluation.**
