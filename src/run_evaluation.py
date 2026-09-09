"""
Evaluate the GeoRAG system against the geological test questions.

This script:
1. Loads the RAG pipeline.
2. Loads test questions from data/test_questions.csv.
3. Runs every question through the RAG pipeline.
4. Prints the generated answer and retrieved evidence.
5. Saves all results to artifacts/evaluation_results.json.
"""

import csv
import json
from pathlib import Path

from src.config import Config
from src.rag_pipeline import RAGPipeline


# =============================================================
# PATHS
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "test_questions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "artifacts" / "evaluation_results.json"
)


# =============================================================
# LOAD TEST QUESTIONS
# =============================================================

def load_test_questions(path: Path):
    """Load questions from the test CSV file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Test questions file not found:\n{path}"
        )

    questions = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(
                "The CSV file does not contain a header."
            )

        # Try common column names.
        question_column = None

        for column in reader.fieldnames:
            if column.lower().strip() in {
                "question",
                "questions",
                "query",
                "queries",
            }:
                question_column = column
                break

        if question_column is None:
            raise ValueError(
                "Could not find a question column in "
                f"{reader.fieldnames}"
            )

        for row in reader:
            question = row.get(
                question_column,
                "",
            ).strip()

            if question:
                questions.append(question)

    if not questions:
        raise ValueError(
            "No questions were found in the test CSV."
        )

    return questions


# =============================================================
# MAIN EVALUATION
# =============================================================

def main():

    print("=" * 90)
    print("GeoRAG EVALUATION")
    print("=" * 90)

    print()
    print(
        f"Loading test questions from:\n"
        f"{TEST_QUESTIONS_PATH}"
    )

    questions = load_test_questions(
        TEST_QUESTIONS_PATH
    )

    print(
        f"\nLoaded {len(questions)} test questions."
    )

    # ---------------------------------------------------------
    # Initialize configuration
    # ---------------------------------------------------------

    config = Config()

    # ---------------------------------------------------------
    # Initialize RAG pipeline
    # ---------------------------------------------------------

    print("\nLoading RAG pipeline...\n")

    pipeline = RAGPipeline(
        config=config,
        top_k=5,
    )

    print("\nPipeline loaded.")
    print(
        f"Documents: {len(pipeline.documents)}"
    )
    print(
        f"Chunks: {len(pipeline.chunks)}"
    )

    print("\n" + "=" * 90)
    print("STARTING EVALUATION")
    print("=" * 90)

    results = []

    # ---------------------------------------------------------
    # Run every question
    # ---------------------------------------------------------

    for index, question in enumerate(
        questions,
        start=1,
    ):

        print("\n")
        print("=" * 90)
        print(f"QUESTION {index}")
        print("=" * 90)

        print(question)

        try:

            result = pipeline.query(
                question=question
            )

            answer = result.get(
                "answer",
                "",
            )

            sources = result.get(
                "sources",
                [],
            )

            print("\n" + "-" * 90)
            print("GENERATED ANSWER")
            print("-" * 90)

            print(answer)

            print("\n" + "-" * 90)
            print("RETRIEVED EVIDENCE")
            print("-" * 90)

            for source_number, source in enumerate(
                sources,
                start=1,
            ):

                print(
                    f"\nSOURCE {source_number}"
                )

                print(
                    f"Document: "
                    f"{source.get('document_id', 'Unknown')}"
                )

                print(
                    f"Chunk: "
                    f"{source.get('chunk_index', 'Unknown')}"
                )

                score = source.get(
                    "score",
                    source.get(
                        "similarity",
                        "Unknown",
                    ),
                )

                print(
                    f"Score: {score}"
                )

                text = source.get(
                    "text",
                    "",
                )

                print("\nTEXT:")
                print(text)

            # -------------------------------------------------
            # Store result
            # -------------------------------------------------

            results.append(
                {
                    "question_number": index,
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                    "status": "success",
                }
            )

        except Exception as exc:

            print("\nERROR")
            print("-" * 90)
            print(str(exc))

            results.append(
                {
                    "question_number": index,
                    "question": question,
                    "answer": "",
                    "sources": [],
                    "status": "error",
                    "error": str(exc),
                }
            )

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # =========================================================
    # SUMMARY
    # =========================================================

    successful = sum(
        1
        for result in results
        if result["status"] == "success"
    )

    failed = len(results) - successful

    print("\n")
    print("=" * 90)
    print("EVALUATION COMPLETE")
    print("=" * 90)

    print(
        f"\nTotal questions: {len(results)}"
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"\nResults saved to:\n{OUTPUT_PATH}"
    )

    print("\nNext step:")
    print(
        "Review evaluation_results.json and compare "
        "the generated answers with the expected answers."
    )


if __name__ == "__main__":
    main()
