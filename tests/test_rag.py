"""Unit tests for RAG pipeline."""

import pytest
from src.config import Config
from src.rag import RAG
from src.rag_pipeline import RAGPipeline
from src.retriever import VectorRetriever
from src.document_loader import Document
from src.chunker import Chunker
import numpy as np


def test_target_uranium_recommendation_extraction():
    """Use the source's Target Uranium 1 wording without loading models."""
    pipeline = RAGPipeline.__new__(RAGPipeline)

    context = (
        "Target Uranium 1\n"
        "Additional information: high values of eUranium are detected. "
        "It is recommend to investigate the ash of the coal for its U "
        "content, and searching for U in the sandstones accompanying the "
        "coal seams.\n\n"
        "Target Uranium 2\n"
        "It is recommended to verify several points with pits."
    )

    answer = pipeline._direct_recommendation_answer(
        "What further investigation was recommended for Target Uranium 1?",
        context,
    )

    assert answer == (
        "The report recommends investigating the ash of the coal for its U "
        "content, and searching for U in the sandstones accompanying the "
        "coal seams."
    )


def test_stage_1_5_deterministic_extraction():
    """Keep high-risk factual answers inside their matching evidence."""
    pipeline = RAGPipeline.__new__(RAGPipeline)

    target_1 = (
        "Target Uranium 1\n"
        "Total area: 769.18 km2.\n"
        "Target Uranium 2\nTotal area: 1,000.00 km2."
    )
    assert pipeline._direct_numeric_answer(
        "What is the total area of Target Uranium 1?",
        target_1,
    ) == "769.18 km2"

    target_2 = (
        "Target Uranium 2\n"
        "The Mayo Lope prospect has uranium concentrations of 0.18% and "
        "0.25% in sandstones. The Mika prospect has uranium veins and "
        "disseminations in brecciated granite with grades of 0.63% U."
    )
    occurrences = pipeline._direct_results_answer(
        "What uranium occurrences were identified around the Mayo Lope and Mika prospects?",
        target_2,
    )
    assert "Mayo Lope" in occurrences
    assert "0.18%" in occurrences and "0.25%" in occurrences
    assert "Mika" in occurrences and "0.63% U" in occurrences
    assert pipeline._direct_numeric_answer(
        "What uranium grade was reported at the Mika prospect?",
        target_2,
    ).startswith("The Mika prospect")

    target_7 = (
        "Target Uranium 7\n"
        "Mining activities: No documented occurrences or mining activities "
        "are documented in the area.\n"
        "Additional information: high values of eUranium are detected in "
        "the area and correlated with high values of eU/eTh. The values "
        "suggest accumulation by infiltration and not as lateritic enrichment. "
        "Intrusives in the basement complex with high eU values suggest a "
        "possible U source for sandstone hosted Uranium."
    )
    assert pipeline._direct_results_answer(
        "Were any mining activities documented at Target Uranium 7?",
        target_7,
    ) == "No documented occurrences or mining activities were documented in the area."
    source_answer = pipeline._direct_results_answer(
        "What geological features were suggested as possible sources of uranium at Target Uranium 7?",
        target_7,
    )
    assert "Intrusives in the basement complex" in source_answer
    assert "Sandstones, shales" not in source_answer

    genetic = (
        "Granite-related uranium occurs in hydrothermal veins or disseminated U. "
        "Sandstone-hosted uranium is another type. Uranium related to phosphorites "
        "and uranium in accessory minerals in granitoid and alkaline rocks are also listed."
    )
    genetic_answer = pipeline._direct_results_answer(
        "What genetic types of uranium mineralization were identified?",
        genetic,
    )
    assert all(term in genetic_answer.lower()
               for term in ["granite", "sandstone", "phosphorite", "accessory"])

    criteria = (
        "Favourable areas were selected using high eU/eTh values, high eUranium "
        "values, known uranium occurrences, organic-rich clastic rocks such as "
        "coal or bitumen, and known phosphorite occurrences."
    )
    criteria_answer = pipeline._direct_results_answer(
        "What criteria were used to select favourable areas for coal exploration?",
        criteria,
    )
    assert all(term in criteria_answer.lower()
               for term in ["euranium", "coal", "bitumen", "phosphorite"])

    genetic_model = (
        "Lower Cretaceous extension and rifting occurred first. Upper Cretaceous "
        "compression, magmatism and metamorphism followed. The Upper Cretaceous-"
        "Palaeocene stage marked the end of metamorphism."
    )
    model_answer = pipeline._direct_results_answer(
        "What is the genetic model for uranium mineralization?",
        genetic_model,
    )
    assert all(term in model_answer.lower() for term in [
               "extension", "compression", "magmatism", "metamorphism"])

    brine = (
        "Piper and Schoeller's plots both show Na and Cl as the dominant cation "
        "and anion, with elevated concentration of Ca in Lower Benue Trough. "
        "Durov's plot shows reverse ion exchange being responsible for variation "
        "in the hydro-geochemistry of the brines. Gibbs' plot identified "
        "evaporation-crystallization and rock weathering as mainly responsible."
    )
    assert "dominant cation and anion" in pipeline._direct_results_answer(
        "What did the Piper and Schoeller plots show for the brines?",
        brine,
    )
    plot_answer = pipeline._direct_results_answer(
        "What processes did the Durov and Gibbs plots indicate?",
        brine,
    )
    assert "reverse ion exchange" in plot_answer
    assert "evaporation-crystallization" in plot_answer
    assert "rock weathering" in plot_answer


def test_stage_1_7_targeted_grounding():
    """Extract requested facts without selecting nearby unrelated evidence."""
    pipeline = RAGPipeline.__new__(RAGPipeline)

    assert pipeline._direct_numeric_answer(
        "What is the total area of Target Uranium 1?",
        "Target Uranium 1\nTotal area: 769.18 km2.\nTarget Uranium 2\nTotal area: 577.09 km2.",
    ) == "769.18 km2"

    prospect_context = (
        "Target Uranium 2\n"
        "Mayo Lope prospect: 0.18% and 0.25% U in sandstones.\n"
        "Mika prospect: veins and disseminations of Uranium in a brecciated "
        "granite with grades of 0.63% U."
    )
    occurrences = pipeline._direct_results_answer(
        "What uranium occurrences were identified around the Mayo Lope and Mika prospects?",
        prospect_context,
    )
    assert "Mayo Lope" in occurrences
    assert "0.18%" in occurrences and "0.25% U" in occurrences
    assert "Mika" in occurrences and "0.63% U" in occurrences

    mika_answer = pipeline._direct_numeric_answer(
        "What uranium grade was reported at the Mika prospect?",
        prospect_context,
    )
    assert "0.63% U" in mika_answer
    assert "veins and disseminations" in mika_answer
    assert "brecciated granite" in mika_answer

    source_answer = pipeline._direct_results_answer(
        "What possible uranium source is suggested for Target Uranium 7?",
        "Target Uranium 7\nIntrusives in the basement complex with high eU values "
        "suggest a possible U source for the formation of sandstone hosted Uranium.",
    )
    assert "Intrusives in the basement complex" in source_answer

    scope_answer = pipeline._direct_results_answer(
        "What is the scope of the D6 Final Report project?",
        "The project covers mining investment facilitation activities on acquired "
        "geological information. The identification, ranking and selection of "
        "exploration targets D1 - D3 are included.",
    )
    assert "investment facilitation" in scope_answer
    assert "identification, ranking and selection" in scope_answer

    geology_answer = pipeline._direct_results_answer(
        "What geological information is described for Target Uranium 1?",
        "Target Uranium 1\nGeology: Sandstones, shales, mudstones and coal "
        "(Cnl and Cms - Late Cretaceous post-rift).",
    )
    assert "Sandstones, shales, mudstones and coal" in geology_answer

    assert pipeline._direct_purpose_answer(
        "What is the purpose of the mining investment facilitation activities?",
        "The project concerns mining investment facilitation activities on "
        "acquired geological information.",
    ) == "Mining investment facilitation activities on acquired geological information."

    assert pipeline._direct_results_answer(
        "What criteria were used to select favourable areas for coal exploration?",
        "Figure 183: Favourability map for target Pb-Zn 9. Figure 185: Coal mine "
        "in the selected target.",
    ) is None

    manganese_answer = pipeline._direct_location_answer(
        "Where are significant manganese occurrences located?",
        "Significant manganese occurrences are about 6 km north-northwest of "
        "Wasagu in the Western Basement Complex, including Duste Zagai Wasagu "
        "and Sabon Tunga Wasagu.",
    )
    assert "Western Basement Complex" in manganese_answer
    assert "Duste Zagai Wasagu" in manganese_answer
    assert "Sabon Tunga Wasagu" in manganese_answer

    brine_answer = pipeline._direct_numeric_answer(
        "How many brine samples were subjected to testing?",
        "The Twenty-one (21) brine samples were subjected to physico-chemical, "
        "mineralogical and evaporation tests.",
    )
    assert "Twenty-one (21) brine samples" in brine_answer


def test_rag_initialization():
    """Test RAG pipeline initialization."""
    try:
        # Create minimal test setup
        chunks_list = [
            __import__('src.chunker', fromlist=['Chunk']).Chunk(
                chunk_id="chunk_0",
                document_id="doc_0",
                text="Copper deposits occur in Nigeria.",
                chunk_index=0,
                start_char=0,
                end_char=50,
                metadata={"title": "Test", "source_file": "test.txt"}
            )
        ]
        embeddings = np.random.randn(1, 1536).astype(np.float32)
        retriever = VectorRetriever(chunks_list, embeddings)

        config = Config()
        rag = RAG(retriever, config)

        assert rag.retriever == retriever
        assert rag.config == config
    except ValueError as e:
        pytest.skip(f"OpenAI API key not configured: {e}")


def test_rag_query_format():
    """Test that RAG query returns expected format."""
    try:
        # This is a structural test, not a full integration test
        # Full integration testing requires actual embeddings/API calls
        from src.chunker import Chunk

        chunks_list = [
            Chunk(
                chunk_id="chunk_0",
                document_id="doc_0",
                text="Lithium occurrences in Ogun State are documented.",
                chunk_index=0,
                start_char=0,
                end_char=60,
                metadata={
                    "title": "Mineral Report",
                    "source_file": "minerals.txt",
                    "source_url": "http://example.com"
                }
            )
        ]
        embeddings = np.random.randn(1, 1536).astype(np.float32)
        retriever = VectorRetriever(chunks_list, embeddings)

        config = Config()
        rag = RAG(retriever, config)

        # The RAG.query method requires actual API calls
        # For now, we just verify the setup
        assert hasattr(rag, 'query')
        assert callable(rag.query)

    except ValueError as e:
        pytest.skip(f"OpenAI API key not configured: {e}")
