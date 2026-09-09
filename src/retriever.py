"""Hybrid retrieval for geological document analysis.

Combines:
1. Semantic similarity using embeddings
2. Keyword relevance
3. Exact phrase matching
4. Title/section relevance
5. Candidate reranking

Designed to improve retrieval of specific geological project information
such as objectives, background, scope, findings, recommendations, and
investigation details.
"""

import re
from typing import List, Dict, Any

import numpy as np

from src.chunker import Chunk
from src.logger import get_logger


class RetrievedResult:
    """Represents a retrieved chunk with relevance score."""

    def __init__(
        self,
        chunk: Chunk,
        score: float,
        rank: int,
    ):
        self.chunk = chunk
        self.score = score
        self.rank = rank

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "rank": self.rank,
            "score": float(self.score),
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "text": self.chunk.text,
            "metadata": self.chunk.metadata,
        }


class VectorRetriever:
    """Hybrid semantic + lexical document retriever."""

    # Only remove words that are genuinely uninformative.
    # IMPORTANT: objective, purpose, scope, background, etc.
    # are intentionally preserved because they describe the
    # information type being requested.
    STOP_WORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "for",
        "from",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "does",
        "do",
        "did",
        "has",
        "have",
        "had",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "at",
        "into",
        "about",
        "can",
        "could",
        "would",
        "should",
        "will",
        "their",
        "there",
        "than",
        "then",
        "also",
        "any",
        "some",
    }

    # Query-intent words. These are not normal keywords;
    # they indicate what kind of content the user is asking for.
    INTENT_TERMS = {
        "objective",
        "objectives",
        "purpose",
        "scope",
        "aim",
        "aims",
        "goal",
        "goals",
        "background",
        "overview",
        "finding",
        "findings",
        "result",
        "results",
        "recommendation",
        "recommendations",
        "method",
        "methods",
        "investigation",
        "investigations",
        "conclusion",
        "conclusions",
    }

    def __init__(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        top_k: int = 5,
        semantic_weight: float = 0.60,
        lexical_weight: float = 0.40,
    ):
        """Initialize hybrid retriever."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) != "
                f"embedding count ({len(embeddings)})"
            )

        if not np.isclose(
            semantic_weight + lexical_weight,
            1.0,
        ):
            raise ValueError(
                "semantic_weight + lexical_weight must equal 1.0"
            )

        self.chunks = chunks
        self.embeddings = embeddings
        self.top_k = top_k

        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.target_uranium_sections = (
            self._index_target_uranium_sections()
        )
        self.criteria_catalogue_chunks = (
            self._index_criteria_catalogue_chunks()
        )

        self.logger = get_logger(__name__)

        self.logger.info(
            f"Initialized Improved Hybrid VectorRetriever with "
            f"{len(chunks)} chunks, "
            f"embedding dimension {embeddings.shape[1]}, "
            f"default top_k={top_k}, "
            f"semantic_weight={semantic_weight}, "
            f"lexical_weight={lexical_weight}"
        )

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
        query_text: str | None = None,
    ) -> List[RetrievedResult]:
        """Retrieve the most relevant chunks."""

        if top_k is None:
            top_k = self.top_k

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if query_embedding.ndim != 1:
            raise ValueError(
                "Query embedding must be a 1-dimensional vector"
            )

        if query_embedding.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query embedding dimension "
                f"({query_embedding.shape[0]}) != "
                f"stored embedding dimension "
                f"({self.embeddings.shape[1]})"
            )

        # ---------------------------------------------------------
        # 1. Semantic similarity
        # ---------------------------------------------------------

        semantic_scores = self._cosine_similarity(
            query_embedding,
            self.embeddings,
        )

        # ---------------------------------------------------------
        # 2. Lexical + intent relevance
        # ---------------------------------------------------------

        if query_text:
            lexical_scores = self._lexical_scores(query_text)

            combined_scores = (
                self.semantic_weight * semantic_scores
                + self.lexical_weight * lexical_scores
            )

            combined_scores += self._target_uranium_section_boost(
                query_text
            )

            combined_scores += self._criteria_catalogue_boost(
                query_text
            )

            self.logger.debug(
                "Using improved hybrid semantic + lexical retrieval"
            )

        else:
            combined_scores = semantic_scores

            self.logger.debug(
                "Using semantic-only retrieval"
            )

        # ---------------------------------------------------------
        # 3. Rank results
        # ---------------------------------------------------------

        top_indices = np.argsort(
            combined_scores
        )[::-1][:top_k]

        results = []

        for rank, idx in enumerate(
            top_indices,
            start=1,
        ):
            results.append(
                RetrievedResult(
                    chunk=self.chunks[idx],
                    score=float(combined_scores[idx]),
                    rank=rank,
                )
            )

        return results

    def _index_target_uranium_sections(
        self,
    ) -> Dict[str, List[int]]:
        """Index body chunks that belong to each Target Uranium section.

        The report's table of contents also names the targets, so this only
        recognizes the numbered body heading format (for example,
        ``7.5.2.1 Target Uranium 1``).  A target's continuation chunks are
        included until the next Target Uranium body heading.
        """

        sections: Dict[str, List[int]] = {}
        active_target: str | None = None
        active_document: str | None = None

        heading_pattern = re.compile(
            r"(?m)^\s*7\.5\.2\.(?P<number>\d+)\s+"
            r"Target\s+Uranium\s+(?P=number)\s*$",
            re.IGNORECASE,
        )

        for index, chunk in enumerate(self.chunks):
            if chunk.document_id != active_document:
                active_document = chunk.document_id
                active_target = None

            heading_match = heading_pattern.search(chunk.text)

            if heading_match:
                active_target = heading_match.group("number")
                sections.setdefault(active_target, [])

            if active_target:
                sections[active_target].append(index)

        return sections

    def _index_criteria_catalogue_chunks(self) -> List[int]:
        """Index body chunks for the uranium Criteria Catalogue section."""

        heading_pattern = re.compile(
            r"7\.5\.1\.1\s+Criteria\s+Catalogue",
            re.IGNORECASE,
        )
        next_section_pattern = re.compile(
            r"7\.5\.1\.2\s+Location\s+of\s+targets",
            re.IGNORECASE,
        )

        indices = []
        active = False

        for index, chunk in enumerate(self.chunks):
            if (
                heading_pattern.search(chunk.text)
                and "uranium mpm" in chunk.text.lower()
            ):
                active = True
                indices.append(index)
                continue

            if active and next_section_pattern.search(chunk.text):
                break

            if active and chunk.document_id:
                indices.append(index)

        return indices

    def _criteria_catalogue_boost(
        self,
        query_text: str,
    ) -> np.ndarray:
        """Prioritize the complete Criteria Catalogue for criteria queries."""

        scores = np.zeros(len(self.chunks), dtype=float)
        query_lower = query_text.lower()

        if not any(
            phrase in query_lower
            for phrase in [
                "criteria",
                "favourable areas",
                "favorable areas",
                "select targets",
                "coal exploration",
            ]
        ):
            return scores

        for index in self.criteria_catalogue_chunks:
            scores[index] = 3.0

        return scores

    def _target_uranium_section_boost(
        self,
        query_text: str,
    ) -> np.ndarray:
        """Strongly prioritize the exact Target Uranium section in a query."""

        scores = np.zeros(len(self.chunks), dtype=float)

        target_match = re.search(
            r"\btarget\s+uranium\s+(\d+)\b",
            query_text,
            re.IGNORECASE,
        )

        if not target_match:
            return scores

        for index in self.target_uranium_sections.get(
            target_match.group(1),
            [],
        ):
            scores[index] = 1.0

        return scores

    def _lexical_scores(
        self,
        query_text: str,
    ) -> np.ndarray:
        """Calculate improved lexical relevance.

        Scoring considers:

        1. Meaningful query-term overlap
        2. Exact phrase overlap
        3. Query-intent matches
        4. Title/metadata matches
        5. Section-like headings in the chunk
        """

        query_terms = self._extract_terms(query_text)
        query_lower = query_text.lower().strip()

        if not query_terms:
            return np.zeros(
                len(self.chunks),
                dtype=float,
            )

        scores = np.zeros(
            len(self.chunks),
            dtype=float,
        )

        for i, chunk in enumerate(self.chunks):

            chunk_text = chunk.text.lower()

            title = str(
                chunk.metadata.get(
                    "title",
                    "",
                )
            ).lower()

            combined_text = (
                f"{title}\n{chunk_text}"
            )

            chunk_terms = set(
                self._extract_terms(
                    combined_text
                )
            )

            if not chunk_terms:
                continue

            # -----------------------------------------------------
            # A. Normal meaningful-term overlap
            # -----------------------------------------------------

            matched_terms = query_terms.intersection(
                chunk_terms
            )

            term_score = (
                len(matched_terms)
                / max(len(query_terms), 1)
            )

            # -----------------------------------------------------
            # B. Exact query phrase
            # -----------------------------------------------------

            phrase_score = 0.0

            if len(query_lower) >= 8:
                if query_lower in chunk_text:
                    phrase_score = 1.0
                else:
                    # Try normalized whitespace.
                    normalized_query = re.sub(
                        r"\s+",
                        " ",
                        query_lower,
                    )

                    normalized_chunk = re.sub(
                        r"\s+",
                        " ",
                        chunk_text,
                    )

                    if normalized_query in normalized_chunk:
                        phrase_score = 1.0

            # -----------------------------------------------------
            # C. Intent matching
            # -----------------------------------------------------

            query_intent = (
                query_terms
                .intersection(self.INTENT_TERMS)
            )

            intent_score = 0.0

            if query_intent:

                matched_intent = (
                    query_intent
                    .intersection(chunk_terms)
                )

                intent_score = (
                    len(matched_intent)
                    / len(query_intent)
                )

            # -----------------------------------------------------
            # D. Title relevance
            # -----------------------------------------------------

            title_terms = set(
                self._extract_terms(title)
            )

            title_matches = (
                query_terms
                .intersection(title_terms)
            )

            title_score = (
                len(title_matches)
                / max(len(query_terms), 1)
            )

            # -----------------------------------------------------
            # E. Section/header relevance
            # -----------------------------------------------------

            header_score = 0.0

            header_patterns = [
                "project background",
                "background",
                "objectives",
                "objective",
                "purpose",
                "scope",
                "introduction",
                "findings",
                "recommendations",
                "conclusions",
                "investigation",
                "results",
            ]

            for header in header_patterns:
                if header in chunk_text:
                    if any(
                        term in query_terms
                        for term in self._extract_terms(header)
                    ):
                        header_score = 1.0
                        break

            # -----------------------------------------------------
            # F. Combine lexical signals
            # -----------------------------------------------------

            score = (
                0.45 * term_score
                + 0.20 * phrase_score
                + 0.15 * intent_score
                + 0.10 * title_score
                + 0.10 * header_score
            )

            scores[i] = min(
                score,
                1.0,
            )

        return scores

    @classmethod
    def _extract_terms(
        cls,
        text: str,
    ) -> set[str]:
        """Extract normalized meaningful terms."""

        words = re.findall(
            r"\b[a-zA-Z0-9][a-zA-Z0-9\-]+\b",
            text.lower(),
        )

        return {
            word
            for word in words
            if word not in cls.STOP_WORDS
            and len(word) > 2
        }

    @staticmethod
    def _cosine_similarity(
        query: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity.

        Returns values shifted from [-1, 1] into [0, 1].
        """

        query_norm = (
            query
            / (
                np.linalg.norm(query)
                + 1e-8
            )
        )

        embeddings_norm = (
            embeddings
            / (
                np.linalg.norm(
                    embeddings,
                    axis=1,
                    keepdims=True,
                )
                + 1e-8
            )
        )

        similarities = np.dot(
            embeddings_norm,
            query_norm,
        )

        similarities = (
            similarities + 1.0
        ) / 2.0

        return similarities
