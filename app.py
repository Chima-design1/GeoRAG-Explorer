from pathlib import Path
import streamlit as st
import fitz
from src.rag_pipeline import RAGPipeline
from src.config import Config

st.set_page_config(page_title="GeoRAG Explorer",
                   page_icon="Geo", layout="wide")

st.title("GeoRAG Explorer")
st.caption(
    "AI-assisted geological document analysis using Retrieval-Augmented Generation")
st.divider()


@st.cache_resource
def load_rag():
    return RAGPipeline(Config(), top_k=5)


rag = load_rag()

with st.sidebar:
    st.header("About GeoRAG")
    st.write(
        "GeoRAG Explorer retrieves relevant evidence from geological "
        "reports and maps, then generates grounded answers using a "
        "local language model."
    )
    st.divider()
    st.subheader("Corpus")
    st.metric("Report documents", "2")
    st.metric("Report chunks", "551")
    st.metric("Map chunks", "189")
    st.metric("Total indexed chunks", "740")
    st.divider()
    st.subheader("Validated performance")
    st.metric("Questions evaluated", "30 / 30")
    st.metric("Excellent", "14")
    st.metric("Good", "9")
    st.metric("Fair", "7")
    st.metric("Poor", "0")

st.subheader("Project overview")
st.write(
    "GeoRAG Explorer is a retrieval-augmented generation system designed "
    "for question answering over geological reports and map-derived data. "
    "The system combines local semantic embeddings, lexical retrieval, "
    "hybrid ranking, reranking, evidence selection, and grounded answer "
    "generation."
)

st.subheader("System architecture")
architecture_image = Path("artifacts/georag_architecture.png")
if architecture_image.exists():
    st.image(
        str(architecture_image),
        caption="GeoRAG Explorer end-to-end retrieval and grounded generation pipeline",
        use_container_width=True
    )
else:
    st.warning("Architecture diagram not found.")

architecture = [
    ("1. Geological corpus", "Reports and geological map-derived text"),
    ("2. Document processing", "Extraction, chunking, and metadata preservation"),
    ("3. Local embeddings", "SentenceTransformer all-MiniLM-L6-v2"),
    ("4. Hybrid retrieval", "Semantic and lexical retrieval over 740 chunks"),
    ("5. Reranking", "Feature-based scoring, diversity control, and context expansion"),
    ("6. Evidence grounding", "Relevant factual evidence is selected for the answer"),
    ("7. Answer generation", "Local Qwen2.5-0.5B-Instruct model"),
]
for stage, description in architecture:
    with st.container(border=True):
        st.markdown(f"**{stage}**")
        st.write(description)

st.divider()
st.subheader("Technology stack")
tech_col1, tech_col2, tech_col3 = st.columns(3)
with tech_col1:
    st.markdown("**Core**")
    st.write("Python")
    st.write("PyTorch")
    st.write("Streamlit")
with tech_col2:
    st.markdown("**Retrieval and NLP**")
    st.write("Sentence Transformers")
    st.write("all-MiniLM-L6-v2")
    st.write("Hybrid semantic + lexical retrieval")
    st.write("Feature-based reranking")
with tech_col3:
    st.markdown("**Generation and data**")
    st.write("Qwen2.5-0.5B-Instruct")
    st.write("PDF document processing")
    st.write("Geological map OCR and processing")
    st.write("Metadata-aware chunking")

st.divider()
st.subheader("Key capabilities")
capabilities = [
    ("Grounded geological question answering",
     "Answers questions using evidence retrieved from the indexed geological corpus."),
    ("Hybrid information retrieval",
     "Combines semantic and lexical retrieval to improve evidence discovery."),
    ("Feature-based reranking",
     "Reranks retrieved candidates using multiple relevance signals."),
    ("Geological map processing",
     "Processes map-derived information alongside geological report content."),
    ("Factual evidence extraction",
     "Applies targeted extraction for questions involving geological facts, locations, grades, and project information."),
    ("Transparent evidence inspection",
     "Shows the retrieved document, ranking information, chunk identifier, and source text behind each answer."),
]
for title, description in capabilities:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(description)

st.divider()
st.subheader("Evaluation results")
st.caption("Latest validated benchmark across the 30-question evaluation set.")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Questions", "30 / 30")
with col2:
    st.metric("Answer quality", "0.7549")
with col3:
    st.metric("Coverage", "0.7612")
with col4:
    st.metric("Direct match", "0.4000")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Retrieval", "1.1298")
with col2:
    st.metric("Lexical", "0.6416")
with col3:
    st.metric("Keyword", "0.7831")
with col4:
    st.metric("Factual match", "0.2889")

st.markdown("**Answer quality distribution**")
quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
with quality_col1:
    st.metric("Excellent", "14")
with quality_col2:
    st.metric("Good", "9")
with quality_col3:
    st.metric("Fair", "7")
with quality_col4:
    st.metric("Poor", "0")
st.success("30/30 questions completed successfully with no Poor-rated answers.")

st.divider()

# -----------------------------
# Geological Map Explorer
# -----------------------------
st.subheader("Geological Map Explorer")
st.write(
    "Explore the original geological maps included in the GeoRAG corpus. "
    "Select a category and map to view the PDF directly in the application."
)

maps_root = Path("data/maps")
map_files = sorted(maps_root.glob("*/*.pdf"))

if map_files:
    categories = sorted({map_file.parent.name for map_file in map_files})

    map_col1, map_col2 = st.columns([1, 2])

    with map_col1:
        selected_category = st.selectbox("Map category", categories)

        category_maps = [
            map_file for map_file in map_files
            if map_file.parent.name == selected_category
        ]

        selected_map = st.selectbox(
            "Select map",
            category_maps,
            format_func=lambda p: p.stem.replace("_", " ")
        )

    with map_col2:
        st.markdown(f"**Selected map:** {selected_map.stem.replace('_', ' ')}")
        st.caption(f"Category: {selected_category}")

        doc = fitz.open(selected_map)
        st.write(f"Pages: {len(doc)}")

        if len(doc) > 1:
            page_number = st.number_input(
                "Map page", min_value=1, max_value=len(doc), value=1, step=1
            )
        else:
            page_number = 1

        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)

        st.image(
            pix.tobytes("png"),
            caption=selected_map.name,
            use_container_width=True
        )

        with open(selected_map, "rb") as pdf_file:
            st.download_button(
                "Download original map PDF",
                data=pdf_file.read(),
                file_name=selected_map.name,
                mime="application/pdf"
            )

        doc.close()
else:
    st.warning("No geological map PDFs were found in data/maps/.")

st.divider()
st.subheader("Ask a geological question")

examples = [
    "What geological information is described for Target Uranium 1?",
    "What uranium occurrences are identified at Target Uranium 2?",
    "What is the purpose of the mining investment facilitation activities?",
    "What is the scope of the D6 Final Report project?"
]
example = st.selectbox("Example questions", [
                       "Select an example..."] + examples)
question = st.text_area(
    "Question",
    value="" if example == "Select an example..." else example,
    height=100,
    placeholder="Ask a question about the geological corpus..."
)

if st.button("Search geological evidence", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and analyzing geological evidence..."):
            result = rag.query(question)

        st.divider()
        st.subheader("Answer")
        st.info(result["answer"])
        st.subheader("Retrieved evidence")
        st.caption(
            "The answer is generated from evidence retrieved from the indexed geological corpus."
        )

        for i, source in enumerate(result["sources"], 1):
            if isinstance(source, dict):
                document_id = source.get("document_id", "Unknown document")
                chunk_id = source.get("chunk_id", "Unknown chunk")
                rank = source.get("rank", i)
                score = source.get("score", None)
                text = source.get("text", "")
                score_text = f"{float(score):.3f}" if score is not None else "N/A"

                with st.expander(
                    f"Evidence {i} - {document_id}", expanded=(i == 1)
                ):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption("Retrieval rank")
                        st.write(rank)
                    with col2:
                        st.caption("Retrieval score")
                        st.write(score_text)
                    with col3:
                        st.caption("Chunk")
                        st.write(chunk_id)

                    st.markdown("**Retrieved geological text**")
                    st.write(text)

                    metadata = source.get("metadata")
                    if metadata:
                        with st.expander("Metadata"):
                            st.write(metadata)
            else:
                with st.expander(f"Evidence {i}", expanded=(i == 1)):
                    st.write(source)

st.divider()
st.caption(
    "GeoRAG Explorer | Local embeddings + hybrid retrieval + reranking + grounded generation"
)
