"""Evaluation utilities for the GeoRAG system.

Evaluates retrieval quality and generated-answer quality
against a set of geological reference questions and answers.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from src.rag_pipeline import RAGPipeline
from src.config import Config
from src.logger import get_logger


# ---------------------------------------------------------------------
# Text normalization and similarity helpers
# ---------------------------------------------------------------------

def normalize_text(text: Any) -> str:
    """Normalize text for lightweight lexical comparison."""

    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: Any) -> List[str]:
    """Convert text into normalized word tokens."""

    normalized = normalize_text(text)

    if not normalized:
        return []

    return normalized.split()


def token_set(text: Any) -> set:
    """Return unique normalized tokens."""

    return set(tokenize(text))


def lexical_similarity(reference: Any, generated: Any) -> float:
    """Calculate token-set Jaccard similarity."""

    ref_tokens = token_set(reference)
    gen_tokens = token_set(generated)

    if not ref_tokens or not gen_tokens:
        return 0.0

    intersection = len(ref_tokens & gen_tokens)
    union = len(ref_tokens | gen_tokens)

    if union == 0:
        return 0.0

    return intersection / union


def reference_coverage(reference: Any, generated: Any) -> float:
    """Measure how much of the reference vocabulary appears in the answer."""

    ref_tokens = token_set(reference)
    gen_tokens = token_set(generated)

    if not ref_tokens:
        return 0.0

    return len(ref_tokens & gen_tokens) / len(ref_tokens)


def keyword_overlap(reference: Any, generated: Any) -> float:
    """Measure overlap using informative reference words."""

    stopwords = {
        "the", "a", "an", "and", "or", "but", "if", "then",
        "of", "to", "in", "on", "for", "from", "with", "by",
        "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "it", "its",
        "as", "at", "into", "through", "about", "which",
        "what", "why", "how", "where", "when", "who",
        "their", "they", "them", "there", "here",
        "has", "have", "had", "do", "does", "did",
        "will", "would", "can", "could", "should",
        "than", "also", "such"
    }

    ref_tokens = {
        token
        for token in tokenize(reference)
        if token not in stopwords and len(token) > 2
    }

    gen_tokens = token_set(generated)

    if not ref_tokens:
        return 0.0

    return len(ref_tokens & gen_tokens) / len(ref_tokens)


def direct_answer_match(reference: Any, generated: Any) -> float:
    """Detect concise answers that directly match the reference content.

    This prevents short factual answers such as ``16`` or ``769.18 km2``
    from being penalized simply because the reference contains a full sentence.
    """

    ref = normalize_text(reference)
    gen = normalize_text(generated)

    if not ref or not gen:
        return 0.0

    if ref == gen:
        return 1.0

    ref_tokens = ref.split()
    gen_tokens = gen.split()

    # Exact contiguous phrase contained in the reference.
    # Require at least two tokens to avoid rewarding arbitrary one-word matches.
    if gen in ref and (
        len(gen_tokens) >= 3
        or bool(re.search(r"\d", gen))
    ):
        return 1.0

    # A numeric-only answer is correct when that number occurs in the reference.
    if re.fullmatch(r"\d+(?:\.\d+)?", gen):
        if re.search(rf"(?<!\d){re.escape(gen)}(?!\d)", ref):
            return 1.0

    return 0.0


def factual_value_match(reference: Any, generated: Any) -> float:
    """Measure overlap of important numeric values in reference and answer."""

    reference_numbers = set(
        re.findall(r"\b\d+(?:\.\d+)?\b", normalize_text(reference))
    )
    generated_numbers = set(
        re.findall(r"\b\d+(?:\.\d+)?\b", normalize_text(generated))
    )

    if not reference_numbers:
        return 0.0

    return len(reference_numbers & generated_numbers) / len(reference_numbers)


def answer_quality_score(
    reference: Any,
    generated: Any,
) -> Dict[str, float]:
    """Calculate several lightweight answer-quality metrics."""

    lexical = lexical_similarity(reference, generated)
    coverage = reference_coverage(reference, generated)
    keywords = keyword_overlap(reference, generated)
    direct = direct_answer_match(reference, generated)
    factual = factual_value_match(reference, generated)

    # Preserve the original lexical metrics while rewarding exact concise
    # answers and correct numeric facts.
    overall = (
        0.20 * lexical
        + 0.25 * coverage
        + 0.30 * keywords
        + 0.15 * factual
        + 0.10 * direct
    )

    # An exact concise match should be treated as fully correct even when
    # the reference is a longer explanatory sentence.
    if direct == 1.0:
        overall = 1.0

    return {
        "lexical_similarity": round(lexical, 4),
        "reference_coverage": round(coverage, 4),
        "keyword_overlap": round(keywords, 4),
        "factual_value_match": round(factual, 4),
        "direct_answer_match": round(direct, 4),
        "answer_quality_score": round(overall, 4),
    }


def classify_answer(score: float) -> str:
    """Convert answer-quality score into a category."""

    if score >= 0.75:
        return "Excellent"
    elif score >= 0.55:
        return "Good"
    elif score >= 0.35:
        return "Fair"
    else:
        return "Poor"


# ---------------------------------------------------------------------
# Evaluation result container
# ---------------------------------------------------------------------

class EvaluationResults:
    """Container for evaluation metrics and per-question results."""

    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
        self.logger = get_logger(__name__)

    def summary(self) -> Dict[str, Any]:
        """Compute aggregate evaluation statistics."""

        if not self.results:
            return {"error": "No results to evaluate"}

        successful = [
            r for r in self.results
            if "error" not in r
        ]

        failed = [
            r for r in self.results
            if "error" in r
        ]

        summary = {
            "total_questions": len(self.results),
            "successful_questions": len(successful),
            "failed_questions": len(failed),
        }

        # Retrieval score
        retrieval_scores = [
            r.get("top_retrieval_score", 0.0)
            for r in successful
            if r.get("top_retrieval_score") is not None
        ]

        if retrieval_scores:
            summary["average_retrieval_score"] = round(
                sum(retrieval_scores) / len(retrieval_scores),
                4,
            )

        # Answer-quality metrics
        quality_results = [
            r for r in successful
            if r.get("answer_quality_score") is not None
        ]

        if quality_results:
            summary["average_lexical_similarity"] = round(
                sum(r["lexical_similarity"] for r in quality_results)
                / len(quality_results),
                4,
            )

            summary["average_reference_coverage"] = round(
                sum(r["reference_coverage"] for r in quality_results)
                / len(quality_results),
                4,
            )

            summary["average_keyword_overlap"] = round(
                sum(r["keyword_overlap"] for r in quality_results)
                / len(quality_results),
                4,
            )

            if all("factual_value_match" in r for r in quality_results):
                summary["average_factual_value_match"] = round(
                    sum(r["factual_value_match"] for r in quality_results)
                    / len(quality_results),
                    4,
                )

            if all("direct_answer_match" in r for r in quality_results):
                summary["average_direct_answer_match"] = round(
                    sum(r["direct_answer_match"] for r in quality_results)
                    / len(quality_results),
                    4,
                )

            summary["average_answer_quality_score"] = round(
                sum(r["answer_quality_score"] for r in quality_results)
                / len(quality_results),
                4,
            )

            categories = {
                "Excellent": 0,
                "Good": 0,
                "Fair": 0,
                "Poor": 0,
            }

            for result in quality_results:
                category = result.get("answer_quality", "Poor")

                if category in categories:
                    categories[category] += 1

            summary["answer_quality_distribution"] = categories

        return summary

    def to_json(self, output_path: Path) -> None:
        """Save evaluation results to JSON."""

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "summary": self.summary(),
            "results": self.results,
        }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

        self.logger.info(
            f"Evaluation results saved to {output_path}"
        )


# ---------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------

def evaluate_rag(
    rag_pipeline: RAGPipeline,
    questions_csv: Path,
    output_path: Optional[Path] = None,
) -> EvaluationResults:
    """Evaluate the current RAGPipeline.

    Expected CSV columns:

        question
        answer

    or:

        question
        reference_answer
    """

    logger = get_logger(__name__)

    questions_csv = Path(questions_csv)

    if not questions_csv.exists():
        raise FileNotFoundError(
            f"Test questions CSV not found: {questions_csv}"
        )

    logger.info(
        f"Loading test questions from {questions_csv}"
    )

    df = pd.read_csv(questions_csv)

    logger.info(
        f"Loaded {len(df)} test questions."
    )

    if "question" not in df.columns:
        raise ValueError(
            "CSV must contain a 'question' column."
        )

    if "reference_answer" in df.columns:
        reference_column = "reference_answer"
    elif "answer" in df.columns:
        reference_column = "answer"
    else:
        raise ValueError(
            "CSV must contain either 'answer' or "
            "'reference_answer' column."
        )

    results = []

    for idx, row in df.iterrows():

        question = str(row["question"])
        reference_answer = str(row[reference_column])

        logger.info(
            f"[{idx + 1}/{len(df)}] Evaluating: "
            f"{question[:80]}..."
        )

        try:

            # ---------------------------------------------------------
            # Run CURRENT RAG pipeline
            # ---------------------------------------------------------

            rag_result = rag_pipeline.query(
                question,
                top_k=rag_pipeline.top_k,
            )

            generated_answer = rag_result.get(
                "answer",
                "",
            )

            # ---------------------------------------------------------
            # Answer quality
            # ---------------------------------------------------------

            quality = answer_quality_score(
                reference_answer,
                generated_answer,
            )

            quality_category = classify_answer(
                quality["answer_quality_score"]
            )

            # ---------------------------------------------------------
            # Retrieved sources
            # ---------------------------------------------------------

            retrieved_sources = []

            for rank, source in enumerate(
                rag_result.get("sources", []),
                start=1,
            ):

                retrieved_sources.append({
                    "rank": source.get(
                        "rank",
                        rank,
                    ),
                    "document_id": source.get(
                        "document_id",
                        source.get("source_file"),
                    ),
                    "chunk_index": source.get(
                        "chunk_index"
                    ),
                    "score": source.get(
                        "score",
                        source.get(
                            "similarity_score",
                            0.0,
                        ),
                    ),
                })

            top_retrieval_score = 0.0

            if retrieved_sources:
                top_retrieval_score = float(
                    retrieved_sources[0].get(
                        "score",
                        0.0,
                    )
                )

            result = {
                "question_idx": int(idx),
                "question": question,
                "generated_answer": generated_answer,
                "reference_answer": reference_answer,
                "top_retrieval_score": round(
                    top_retrieval_score,
                    4,
                ),
                "retrieved_sources": retrieved_sources,
                **quality,
                "answer_quality": quality_category,
            }

            results.append(result)

        except Exception as e:

            logger.exception(
                f"Evaluation failed for question "
                f"{idx + 1}: {e}"
            )

            results.append({
                "question_idx": int(idx),
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": "",
                "error": str(e),
            })

    evaluation_results = EvaluationResults(results)

    if output_path:
        evaluation_results.to_json(output_path)

    return evaluation_results


# ---------------------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete 30-question evaluation."""

    logger = get_logger(__name__)

    project_root = Path(__file__).resolve().parent.parent

    questions_csv = (
        project_root
        / "data"
        / "test_questions.csv"
    )

    output_path = (
        project_root
        / "artifacts"
        / "evaluation_results.json"
    )

    print("=" * 80)
    print("GeoRAG Explorer - Full Evaluation")
    print("=" * 80)
    print()
    print(f"Questions: {questions_csv}")
    print(f"Output:    {output_path}")
    print()

    print("Loading current RAGPipeline...")

    config = Config()

    rag = RAGPipeline(
        config,
        top_k=config.default_top_k,
    )

    print(
        f"Pipeline loaded: "
        f"{len(rag.documents)} documents, "
        f"{len(rag.chunks)} chunks"
    )

    print()
    print("Running evaluation...")
    print()

    evaluation = evaluate_rag(
        rag_pipeline=rag,
        questions_csv=questions_csv,
        output_path=output_path,
    )

    summary = evaluation.summary()

    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f"Total questions:       "
        f"{summary.get('total_questions', 0)}"
    )

    print(
        f"Successful questions:  "
        f"{summary.get('successful_questions', 0)}"
    )

    print(
        f"Failed questions:      "
        f"{summary.get('failed_questions', 0)}"
    )

    print(
        f"Average retrieval:     "
        f"{summary.get('average_retrieval_score', 0):.4f}"
    )

    print(
        f"Average lexical:       "
        f"{summary.get('average_lexical_similarity', 0):.4f}"
    )

    print(
        f"Average coverage:      "
        f"{summary.get('average_reference_coverage', 0):.4f}"
    )

    print(
        f"Average keyword:       "
        f"{summary.get('average_keyword_overlap', 0):.4f}"
    )

    print(
        f"Average factual match: "
        f"{summary.get('average_factual_value_match', 0):.4f}"
    )

    print(
        f"Average direct match:  "
        f"{summary.get('average_direct_answer_match', 0):.4f}"
    )

    print(
        f"Average answer quality:"
        f" {summary.get('average_answer_quality_score', 0):.4f}"
    )

    distribution = summary.get(
        "answer_quality_distribution",
        {},
    )

    print()
    print("Answer quality distribution:")

    print(
        f"  Excellent: {distribution.get('Excellent', 0)}"
    )

    print(
        f"  Good:      {distribution.get('Good', 0)}"
    )

    print(
        f"  Fair:      {distribution.get('Fair', 0)}"
    )

    print(
        f"  Poor:      {distribution.get('Poor', 0)}"
    )

    print()
    print(f"Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
