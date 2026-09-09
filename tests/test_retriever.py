"""Unit tests for retriever module."""

import pytest
import numpy as np
from src.retriever import VectorRetriever, RetrievedResult
from src.chunker import Chunk


def create_test_chunks(n: int = 5) -> list:
    """Create test chunks."""
    chunks = []
    for i in range(n):
        chunk = Chunk(
            chunk_id=f"chunk_{i}",
            document_id=f"doc_{i % 2}",
            text=f"Test chunk {i} with content.",
            chunk_index=i,
            start_char=0,
            end_char=100,
            metadata={
                "title": f"Document {i % 2}",
                "source_file": f"doc_{i % 2}.txt"
            }
        )
        chunks.append(chunk)
    return chunks


def create_test_embeddings(n: int = 5, dim: int = 1536) -> np.ndarray:
    """Create random test embeddings."""
    return np.random.randn(n, dim).astype(np.float32)


def test_retriever_initialization():
    """Test VectorRetriever initialization."""
    chunks = create_test_chunks(5)
    embeddings = create_test_embeddings(5, 1536)

    retriever = VectorRetriever(chunks, embeddings, top_k=3)
    assert len(retriever.chunks) == 5
    assert retriever.embeddings.shape == (5, 1536)
    assert retriever.top_k == 3


def test_retriever_mismatch():
    """Test that retriever rejects mismatched chunks/embeddings."""
    chunks = create_test_chunks(5)
    embeddings = create_test_embeddings(3, 1536)  # Wrong count

    with pytest.raises(ValueError):
        VectorRetriever(chunks, embeddings)


def test_retrieve_top_k():
    """Test retrieval with top_k parameter."""
    chunks = create_test_chunks(10)
    embeddings = create_test_embeddings(10, 1536)

    retriever = VectorRetriever(chunks, embeddings, top_k=3)

    query_embedding = embeddings[0]  # Use first embedding as query
    results = retriever.retrieve(query_embedding, top_k=3)

    assert len(results) == 3
    assert all(isinstance(r, RetrievedResult) for r in results)

    # Results should be sorted by score (descending)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_override_top_k():
    """Test overriding default top_k at query time."""
    chunks = create_test_chunks(10)
    embeddings = create_test_embeddings(10, 1536)

    retriever = VectorRetriever(chunks, embeddings, top_k=3)
    query_embedding = embeddings[0]

    # Override with different top_k
    results = retriever.retrieve(query_embedding, top_k=5)
    assert len(results) == 5


def test_retrieved_result():
    """Test RetrievedResult object."""
    chunk = create_test_chunks(1)[0]
    result = RetrievedResult(chunk, score=0.95, rank=1)

    assert result.score == 0.95
    assert result.rank == 1

    result_dict = result.to_dict()
    assert result_dict["score"] == 0.95
    assert result_dict["rank"] == 1
    assert result_dict["chunk_id"] == "chunk_0"


def test_target_uranium_query_prioritizes_exact_section():
    """Exact Target Uranium queries must not retrieve another target first."""
    chunks = [
        Chunk(
            chunk_id="uranium_1_heading",
            document_id="uranium_report",
            text="7.5.2.1 Target Uranium 1",
            chunk_index=0,
            start_char=0,
            end_char=25,
            metadata={},
        ),
        Chunk(
            chunk_id="uranium_1_evidence",
            document_id="uranium_report",
            text=(
                "It is recommend to investigate the ash of the coal for its "
                "U content and search the accompanying sandstones."
            ),
            chunk_index=1,
            start_char=25,
            end_char=140,
            metadata={},
        ),
        Chunk(
            chunk_id="uranium_2_heading",
            document_id="uranium_report",
            text="7.5.2.2 Target Uranium 2",
            chunk_index=2,
            start_char=140,
            end_char=165,
            metadata={},
        ),
        Chunk(
            chunk_id="uranium_2_evidence",
            document_id="uranium_report",
            text="Target Uranium 2 has a different recommendation.",
            chunk_index=3,
            start_char=165,
            end_char=220,
            metadata={},
        ),
    ]

    embeddings = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    retriever = VectorRetriever(chunks, embeddings, top_k=2)

    results = retriever.retrieve(
        query_embedding=np.array([0.0, 1.0], dtype=np.float32),
        query_text=(
            "What further investigation was recommended for Target Uranium 1?"
        ),
        top_k=2,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "uranium_1_heading",
        "uranium_1_evidence",
    ]


def test_criteria_query_prioritizes_catalogue_body_chunks():
    """Criteria queries should retrieve the catalogue body, not its TOC."""
    chunks = [
        Chunk(
            chunk_id="criteria_toc",
            document_id="report",
            text="7.5.1.1 Criteria Catalogue 317",
            chunk_index=0,
            start_char=0,
            end_char=35,
            metadata={},
        ),
        Chunk(
            chunk_id="criteria_body",
            document_id="report",
            text=(
                "7.5.1.1 Criteria Catalogue\n"
                "Uranium MPM was not created. Criteria for selection of areas "
                "include high eU/eTh values and high eUranium values."
            ),
            chunk_index=1,
            start_char=35,
            end_char=180,
            metadata={},
        ),
        Chunk(
            chunk_id="criteria_continuation",
            document_id="report",
            text=(
                "Uranium occurrences known and sedimentary clastic rocks "
                "with organic material, coal, bitumen, and phosphorite occurrences known."
            ),
            chunk_index=2,
            start_char=180,
            end_char=320,
            metadata={},
        ),
        Chunk(
            chunk_id="next_section",
            document_id="report",
            text="7.5.1.2 Location of targets",
            chunk_index=3,
            start_char=320,
            end_char=350,
            metadata={},
        ),
    ]
    embeddings = np.zeros((4, 2), dtype=np.float32)
    embeddings[0] = [1.0, 0.0]
    embeddings[1] = [0.0, 1.0]
    embeddings[2] = [0.0, 1.0]
    embeddings[3] = [1.0, 0.0]

    retriever = VectorRetriever(chunks, embeddings, top_k=2)
    results = retriever.retrieve(
        np.array([1.0, 0.0], dtype=np.float32),
        query_text="What criteria were used to select favourable areas?",
        top_k=2,
    )

    result_ids = {result.chunk.chunk_id for result in results}
    assert "criteria_body" in result_ids or "criteria_continuation" in result_ids
    assert "criteria_toc" not in result_ids
