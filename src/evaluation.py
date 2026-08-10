"""Evaluation utilities for RAG system.

Evaluate retrieval and generation quality against geological test questions.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from src.rag import RAG
from src.logger import get_logger


class EvaluationResults:
    """Container for evaluation metrics."""
    
    def __init__(self, results: List[Dict[str, Any]]):
        """Initialize with evaluation results.
        
        Args:
            results: List of per-question evaluation dicts.
        """
        self.results = results
        self.logger = get_logger(__name__)
    
    def summary(self) -> Dict[str, Any]:
        """Compute evaluation summary statistics.
        
        Returns:
            Dict with aggregate metrics.
        """
        if not self.results:
            return {"error": "No results to evaluate"}
        
        # Count successes
        total = len(self.results)
        
        # Compute metrics that are available
        summary = {
            "total_questions": total,
            "results_per_question": self.results,
        }
        
        # Add any computed metrics here as we add evaluation methods
        return summary
    
    def to_json(self, output_path: Path) -> None:
        """Save results to JSON file.
        
        Args:
            output_path: Path to save JSON.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Evaluation results saved to {output_path}")


def evaluate_rag(
    rag_pipeline: RAG,
    questions_csv: Path,
    output_path: Optional[Path] = None,
) -> EvaluationResults:
    """Evaluate RAG pipeline against geological test questions.
    
    Args:
        rag_pipeline: RAG instance.
        questions_csv: Path to CSV with test questions.
                      Expected columns: question, reference_answer (optional)
        output_path: Optional path to save evaluation results.
    
    Returns:
        EvaluationResults object.
    
    Raises:
        FileNotFoundError: If questions_csv doesn't exist.
        ValueError: If questions_csv doesn't have 'question' column.
    """
    logger = get_logger(__name__)
    
    questions_csv = Path(questions_csv)
    if not questions_csv.exists():
        raise FileNotFoundError(f"Test questions CSV not found: {questions_csv}")
    
    # Load test questions
    logger.info(f"Loading test questions from {questions_csv}")
    df = pd.read_csv(questions_csv)
    
    if "question" not in df.columns:
        raise ValueError("CSV must have 'question' column")
    
    # Evaluate each question
    results = []
    for idx, row in df.iterrows():
        question = row["question"]
        reference_answer = row.get("reference_answer", None) if "reference_answer" in df.columns else None
        
        logger.info(f"[{idx+1}/{len(df)}] Evaluating: {question[:50]}...")
        
        # Run RAG
        try:
            rag_result = rag_pipeline.query(question)
            
            result_dict = {
                "question_idx": idx,
                "question": question,
                "generated_answer": rag_result["answer"],
                "reference_answer": reference_answer,
                "retrieved_sources": [
                    {
                        "rank": s["rank"],
                        "score": s["similarity_score"],
                        "source_file": s["source_file"],
                    }
                    for s in rag_result["sources"]
                ],
                "top_retrieval_score": rag_result["retrieved_chunks"][0]["score"]
                if rag_result["retrieved_chunks"]
                else 0.0,
            }
            results.append(result_dict)
        except Exception as e:
            logger.error(f"Error evaluating question {idx}: {e}")
            results.append({
                "question_idx": idx,
                "question": question,
                "error": str(e),
            })
    
    # Create evaluation results
    eval_results = EvaluationResults(results)
    
    # Save if output path provided
    if output_path:
        eval_results.to_json(output_path)
    
    return eval_results
