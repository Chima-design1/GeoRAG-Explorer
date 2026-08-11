# GeoRAG Explorer - Phase 1 Development Guide

## Phase 1 Complete: Text-Based RAG Baseline

This document describes the Phase 1 implementation of GeoRAG Explorer—a complete working text-based Retrieval-Augmented Generation system for geological knowledge retrieval.

---

## Architecture Overview

### Pipeline Flow

```
Geological Reports (TXT)
       ↓
Document Loading (document_loader.py)
       ↓
Text Cleaning & Metadata Extraction
       ↓
Chunking with Metadata (chunker.py)
       ↓
OpenAI Embeddings (embeddings.py)
       ↓
Local Caching (NumPy + pickle)
       ↓
Vector Retrieval (retriever.py)
       ↓
LLM Generation (rag.py)
       ↓
Grounded Answer + Sources
```

---

## Module Descriptions

### `src/config.py`

**Purpose**: Centralized configuration management using environment variables.

**Key Features**:
- Loads `.env` file for secure API key storage
- Provides default values for embedding and chat models
- Validates required configuration (OPENAI_API_KEY)
- Creates necessary directories automatically
- Configurable via environment variables

**Usage**:
```python
from src.config import Config
config = Config()
print(config.embedding_model)  # "text-embedding-3-small"
print(config.chat_model)        # "gpt-4-turbo"
```

**Environment Variables**:
- `OPENAI_API_KEY` (required)
- `OPENAI_EMBEDDING_MODEL` (default: "text-embedding-3-small")
- `OPENAI_CHAT_MODEL` (default: "gpt-4-turbo")
- `LOG_LEVEL` (default: "INFO")
- `DEFAULT_TOP_K` (default: 5)
- `REPORTS_DIR` (default: "data/reports")
- `EMBEDDINGS_CACHE_PATH` (default: "artifacts/embeddings.pkl")

---

### `src/logger.py`

**Purpose**: Structured logging utilities.

**Key Features**:
- Configurable logging levels
- Consistent formatting across modules
- Integration with Config for log level

**Usage**:
```python
from src.logger import get_logger
logger = get_logger(__name__)
logger.info("This is an info message")
```

---

### `src/document_loader.py`

**Purpose**: Load geological text documents from disk.

**Key Classes**:

#### `Document`
Represents a geological document with metadata:
- `document_id`: Unique identifier
- `title`: Document title
- `content`: Full text content
- `source_file`: Original filename
- `source_url`: Optional URL
- `document_type`: e.g., "geological_report"

#### `DocumentLoader`
Loads all `.txt` files from a directory:
- Handles encoding issues (UTF-8 with latin-1 fallback)
- Extracts title from filename or first line
- Preserves source information
- Logs loading progress

**Usage**:
```python
from src.document_loader import DocumentLoader
loader = DocumentLoader("data/reports")
documents = loader.load_all()
print(f"Loaded {len(documents)} documents")
```

---

### `src/chunker.py`

**Purpose**: Split documents into overlapping chunks while preserving metadata.

**Key Classes**:

#### `Chunk`
Represents a document chunk:
- `chunk_id`: Unique identifier (document_id + chunk_index)
- `document_id`: Parent document ID
- `text`: Chunk text content
- `chunk_index`: Index within document
- `start_char`, `end_char`: Character offsets
- `metadata`: Dict with title, source_file, source_url, document_type

#### `Chunker`
Splits documents into overlapping chunks:
- Configurable chunk size (default: 1000 chars)
- Configurable overlap (default: 200 chars)
- Attempts to break at sentence/word boundaries
- Preserves all metadata in each chunk

**Usage**:
```python
from src.chunker import Chunker
chunker = Chunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk_documents(documents)
print(f"Created {len(chunks)} chunks")
```

**Metadata Preservation**:
Each chunk retains:
```python
chunk.metadata = {
    "title": "Document Title",
    "source_file": "document.txt",
    "source_url": "http://example.com/document",
    "document_type": "geological_report"
}
```

---

### `src/embeddings.py`

**Purpose**: Generate and cache embeddings using OpenAI API.

**Key Class**: `EmbeddingGenerator`

**Features**:
- Generates embeddings via OpenAI API
- Caches embeddings locally (NumPy + pickle)
- Avoids redundant API calls
- Configurable embedding model
- Supports force regeneration

**Usage**:
```python
from src.embeddings import EmbeddingGenerator
gen = EmbeddingGenerator(config)
embeddings = gen.embed_chunks(
    chunks,
    cache_path="artifacts/embeddings.pkl",
    force_regenerate=False
)
print(f"Embeddings shape: {embeddings.shape}")  # (n_chunks, 1536)
```

**Caching**:
- First run: Generates embeddings via API, saves to cache
- Subsequent runs: Loads from cache if available
- Cache validation: Checks chunk count matches
- Force regeneration: Use `force_regenerate=True` to skip cache

---

### `src/retriever.py`

**Purpose**: Vector retrieval using cosine similarity.

**Key Classes**:

#### `RetrievedResult`
Represents a retrieved chunk:
- `chunk`: The Chunk object
- `score`: Similarity score (0-1)
- `rank`: Rank in results (1-indexed)

#### `VectorRetriever`
Semantic search using NumPy and cosine similarity:
- Configurable top_k (default: 5)
- Cosine similarity scoring
- Efficient batch retrieval
- Designed for replacement with FAISS/Qdrant in Phase 4

**Usage**:
```python
from src.retriever import VectorRetriever
retriever = VectorRetriever(chunks, embeddings, top_k=5)

# Retrieve for a query embedding
results = retriever.retrieve(query_embedding, top_k=5)
for result in results:
    print(f"Rank {result.rank}: {result.score:.4f}")
    print(f"Text: {result.chunk.text[:100]}...")
```

**Cosine Similarity**:
- Normalized vectors for stable scoring
- Shifted to [0, 1] range for interpretability
- Efficient NumPy implementation

---

### `src/rag.py`

**Purpose**: Complete RAG pipeline combining retrieval and generation.

**Key Class**: `RAG`

**Features**:
- Retrieves relevant chunks using VectorRetriever
- Generates grounded answers using OpenAI LLM
- Instructs model to cite sources
- Formats evidence for generation
- Returns answer + sources + metadata

**Grounding Instructions**:
The system instructs the LLM to:
1. Use ONLY provided evidence
2. Never invent geological facts
3. State when evidence is insufficient
4. Cite sources by reference number
5. Preserve geological terminology
6. Distinguish explicit facts from inference

**Usage**:
```python
from src.rag import RAG
rag = RAG(retriever, config, embedding_generator=gen)

result = rag.query("What minerals occur in Ogun State?", top_k=5)
print("Answer:", result["answer"])
print("Sources:", result["sources"])
```

**Return Format**:
```python
{
    "question": "User question",
    "answer": "Grounded LLM answer with citations",
    "sources": [
        {
            "rank": 1,
            "similarity_score": 0.85,
            "source_number": 1,
            "title": "Document Title",
            "source_file": "doc.txt",
            "source_url": "http://...",
            "chunk_id": "doc_chunk_0",
            "text_preview": "First 200 chars of chunk..."
        },
        # ... more sources
    ],
    "retrieved_chunks": [
        # Full chunk details for inspection
    ]
}
```

---

### `src/evaluation.py`

**Purpose**: Evaluate RAG pipeline against test questions.

**Key Classes**:

#### `EvaluationResults`
Stores and aggregates evaluation results:
- Per-question metrics
- Summary statistics
- JSON export

#### Functions:
- `evaluate_rag()`: Run evaluation on test questions CSV

**CSV Format**:
Expected columns:
- `question`: Geological question (required)
- `reference_answer`: Optional reference answer

**Usage**:
```python
from src.evaluation import evaluate_rag
results = evaluate_rag(
    rag_pipeline=rag,
    questions_csv="data/test_questions.csv",
    output_path="artifacts/evaluation_results.json"
)
summary = results.summary()
```

**Output Format**:
```json
{
    "total_questions": 10,
    "results_per_question": [
        {
            "question_idx": 0,
            "question": "What minerals...",
            "generated_answer": "...",
            "reference_answer": "...",
            "retrieved_sources": [...],
            "top_retrieval_score": 0.85
        }
    ]
}
```

---

## Getting Started with Phase 1

### Installation

```bash
# Clone repository
git clone https://github.com/Chima-design1/GeoRAG-Explorer.git
cd GeoRAG-Explorer

# Run setup script
bash setup.sh

# Or manual setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add OPENAI_API_KEY
```

### Quick Start

```python
from pathlib import Path
from src.config import Config
from src.document_loader import DocumentLoader
from src.chunker import Chunker
from src.embeddings import EmbeddingGenerator
from src.retriever import VectorRetriever
from src.rag import RAG

# Initialize
config = Config()

# Load documents
loader = DocumentLoader(config.reports_dir)
documents = loader.load_all()

# Chunk
chunker = Chunker()
chunks = chunker.chunk_documents(documents)

# Embed
embedding_gen = EmbeddingGenerator(config)
embeddings = embedding_gen.embed_chunks(chunks, cache_path=config.embeddings_cache_path)

# Retrieve
retriever = VectorRetriever(chunks, embeddings, top_k=5)

# Generate
rag = RAG(retriever, config, embedding_generator=embedding_gen)

# Query
result = rag.query("What minerals are found in Nigeria?", top_k=5)
print(result["answer"])
```

---

## Data Flow

### 1. Document Loading

```
data/reports/*.txt
         ↓
    DocumentLoader
         ↓
    Document objects (id, title, content, metadata)
```

### 2. Chunking

```
Document
    ↓
Chunker (1000 char size, 200 char overlap)
    ↓
Multiple Chunk objects (with preserved metadata)
```

### 3. Embedding

```
Chunks
    ↓
EmbeddingGenerator (OpenAI API)
    ↓
NumPy array (n_chunks × 1536)
    ↓
Cache (pickle file)
```

### 4. Retrieval

```
Query Question
    ↓
Query Embedding (OpenAI API)
    ↓
VectorRetriever (cosine similarity)
    ↓
Top-K RetrievedResult objects (with scores)
```

### 5. Generation

```
Question + Retrieved Chunks
    ↓
LLM Prompt (with grounding instructions)
    ↓
OpenAI API (gpt-4-turbo)
    ↓
Grounded Answer + Citations
```

---

## Key Design Decisions

### 1. Local Vector Index (Phase 1)

**Why NumPy + cosine similarity instead of FAISS/Qdrant?**
- Phase 1 focuses on correctness and simplicity
- NumPy is lightweight and no external dependencies
- Easy to understand and debug
- Perfect for initial testing with moderate datasets
- Planned migration to FAISS/Qdrant in Phase 4

### 2. Embeddings Caching

**Why cache embeddings?**
- OpenAI API charges per embedding call
- Regenerating for every session is expensive
- Cache is fast and enables quick iteration
- Validation ensures cache consistency

### 3. Metadata Preservation

**Why retain metadata in chunks?**
- Source traceability is critical for geology
- Users need to know where information came from
- Enables future metadata-based filtering
- Supports debugging and evaluation

### 4. Grounding Prompt

**Why explicit grounding instructions?**
- LLMs tend to hallucinate geological facts
- Geological accuracy requires evidence
- Must avoid inventing mineral occurrences
- Explicit instructions improve compliance

---

## Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_chunker.py

# Verbose output
pytest -v tests/
```

### Test Coverage

- `test_chunker.py`: Document chunking and metadata preservation
- `test_embeddings.py`: Embedding generation and caching
- `test_retriever.py`: Cosine similarity and retrieval
- `test_rag.py`: RAG pipeline integration

---

## Notebooks

### 01_data_exploration.ipynb
- Load and explore geological reports
- Display document statistics
- Prepare for chunking

### 02_document_processing.ipynb
- Chunk documents
- Generate embeddings
- Verify metadata preservation
- Test caching

### 03_retrieval_evaluation.ipynb
- Test semantic search
- Evaluate retrieval quality
- Analyze score distribution
- Sample queries

### 04_rag_evaluation.ipynb
- End-to-end RAG testing
- Test answer generation
- Show source citations
- Evaluate grounding

**Run Notebooks**:
```bash
jupyter notebook notebooks/
```

---

## Phase 1 Limitations

### Text Only
- Maps and geochemical PDFs not yet supported
- Phase 2 will add PDF extraction
- Phase 3 will add map processing

### Local Index
- Vector index reloaded on each session
- No persistent storage
- Scalability limited to embeddings that fit in memory
- Phase 4 will add database persistence

### Single Retrieval Method
- Semantic search only
- No BM25 keyword matching
- No metadata filtering
- Phase 4 will add hybrid retrieval

### No Reranking
- Results ranked by cosine similarity only
- No cross-encoder reranking
- Phase 5 will add reranking

### Limited Evaluation
- Awaiting test_questions.csv for full evaluation
- Phase 6 will run comprehensive evaluation
- Currently supports manual inspection

---

## Moving to Phase 2

When ready to add PDF processing:

1. Inspect existing scraper code (if available)
2. Refactor to `src/scraper.py`
3. Implement PDF text extraction in `src/pdf_processor.py`
4. Integrate into document loading pipeline
5. Test with actual PDF reports

---

## Moving to Phase 3

When geological PDF maps are provided:

1. Inspect the actual PDF files
2. Determine content (selectable text, OCR, vector graphics, etc.)
3. Design map-specific metadata extraction
4. Implement in `src/map_processor.py`
5. Create structured map representations
6. Integrate into retrieval system

**Do NOT**:
- Assume map processing requirements
- Invent map metadata
- Convert maps to simple text
- Lose cartographic information

---

## Performance Considerations

### Embedding Generation
- ~100 chunks: ~1 second (cached)
- ~1000 chunks: ~10 seconds (via API)
- Caching enables instant subsequent runs

### Retrieval
- ~100 chunks: <1ms (NumPy operations)
- ~1000 chunks: <10ms
- Top-K extraction: O(n log k) with NumPy argsort

### Answer Generation
- OpenAI API call: ~2-5 seconds
- Depends on prompt length and token count

### Total Time (First Run)
- Documents + chunking: <1 second
- Embedding generation: ~10 seconds (depends on chunk count)
- Retrieval + generation: ~5 seconds
- **Total: ~15 seconds per query**

### Total Time (Cached)
- Load cache: <1 second
- Retrieval: <10ms
- Generation: ~5 seconds
- **Total: ~5 seconds per query**

---

## Security & Best Practices

### API Keys
- Never commit `.env` file
- `.env` is in `.gitignore`
- Use `.env.example` as template
- Rotate keys regularly

### Cache Files
- `artifacts/embeddings.pkl` should not be committed
- Cache is git-ignored
- Regenerate on deployment if needed

### Error Handling
- All modules include try-except blocks
- Clear error messages for debugging
- Logging at INFO and DEBUG levels

### Type Hints
- All functions include type hints
- Better IDE support and error detection

### Docstrings
- All classes and functions documented
- Usage examples provided
- Parameter descriptions included

---

## Troubleshooting

### "OPENAI_API_KEY not set"
```bash
# Check .env file
cat .env
# Should have: OPENAI_API_KEY=sk-...
```

### "Reports directory does not exist"
```bash
# Create and add reports
mkdir -p data/reports
# Place .txt files in data/reports/
```

### "No embeddings generated"
```bash
# Check OpenAI API key is valid
# Check network connectivity
# Check API rate limits haven't been exceeded
```

### Cache validation errors
```bash
# Delete cache and regenerate
rm artifacts/embeddings.pkl
# Re-run embedding generation
```

---

## Next Steps After Phase 1

1. **Add geological reports to `data/reports/`** as `.txt` files
2. **Run notebooks in order** to test the system
3. **Provide test questions CSV** for Phase 6 evaluation
4. **Provide PDF reports** for Phase 2 implementation
5. **Provide geological maps** for Phase 3 map processing
6. **Hybrid retrieval** in Phase 4
7. **Reranking** in Phase 5
8. **Full evaluation** in Phase 6
9. **Portfolio presentation** in Phase 7

---

## Support

For questions or issues:
1. Check this development guide
2. Review docstrings in source code
3. Run notebook cells interactively
4. Check logs (INFO and DEBUG levels)
5. Open GitHub issue if needed

---

**Phase 1 Status**: ✅ Complete and ready for geological reports

**Next**: Awaiting geological data to proceed with Phase 2
