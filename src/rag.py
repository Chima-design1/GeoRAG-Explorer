"""RAG (Retrieval-Augmented Generation) pipeline.

Combines retrieval with LLM generation to produce grounded answers
with source citations.
"""

import logging
from typing import Dict, Any, List
import numpy as np
from openai import OpenAI
from src.config import Config
from src.embeddings import EmbeddingGenerator
from src.retriever import VectorRetriever, RetrievedResult
from src.logger import get_logger


class RAG:
    """RAG pipeline: retrieve evidence + generate grounded answer.
    
    Retrieves relevant chunks and generates answers using LLM,
    instructing the model to use only retrieved evidence and cite sources.
    """
    
    def __init__(
        self,
        retriever: VectorRetriever,
        config: Config,
        embedding_generator: EmbeddingGenerator | None = None,
    ):
        """Initialize RAG pipeline.
        
        Args:
            retriever: VectorRetriever instance.
            config: Config object.
            embedding_generator: Optional EmbeddingGenerator for query embedding.
                                If None, creates one from config.
        """
        self.retriever = retriever
        self.config = config
        self.embedding_generator = embedding_generator or EmbeddingGenerator(config)
        self.client = OpenAI(api_key=config.openai_api_key)
        self.logger = get_logger(__name__)
    
    def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        """Answer a geological question using RAG.
        
        Args:
            question: Geological question.
            top_k: Number of retrieved chunks to use. If None, uses retriever default.
        
        Returns:
            Dict with keys:
            - "question": The input question
            - "answer": Generated answer with grounding instructions
            - "sources": List of retrieved sources with metadata
            - "retrieved_chunks": Full retrieved chunks for inspection
        """
        if top_k is None:
            top_k = self.retriever.top_k
        
        self.logger.info(f"Processing query: {question[:100]}...")
        
        # Generate query embedding
        query_embedding = self._embed_query(question)
        
        # Retrieve relevant chunks
        retrieved = self.retriever.retrieve(query_embedding, top_k=top_k)
        self.logger.info(
            f"Retrieved {len(retrieved)} chunks. "
            f"Top score: {retrieved[0].score:.4f if retrieved else 'N/A'}"
        )
        
        # Generate grounded answer
        answer = self._generate_answer(question, retrieved)
        
        # Format sources
        sources = self._format_sources(retrieved)
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": [r.to_dict() for r in retrieved],
        }
    
    def _embed_query(self, question: str) -> np.ndarray:
        """Embed a query question.
        
        Args:
            question: Question text.
        
        Returns:
            Query embedding as NumPy array.
        """
        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=question,
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding
    
    def _generate_answer(
        self,
        question: str,
        retrieved: List[RetrievedResult],
    ) -> str:
        """Generate grounded answer using LLM.
        
        Args:
            question: Original question.
            retrieved: List of RetrievedResult objects.
        
        Returns:
            Generated answer with citations.
        """
        # Format retrieved evidence
        evidence = "\n\n".join(
            f"[Source {i+1}] ({r.chunk.metadata.get('source_file', 'Unknown')}): "
            f"{r.chunk.text[:500]}..."
            if len(r.chunk.text) > 500
            else f"[Source {i+1}] ({r.chunk.metadata.get('source_file', 'Unknown')}): "
                 f"{r.chunk.text}"
            for i, r in enumerate(retrieved)
        )
        
        # System prompt for grounded generation
        system_prompt = """You are a geological knowledge assistant. You answer questions about geology, 
minerals, geological formations, and geochemical data using ONLY the provided evidence.

IMPORTANT RULES:
1. Use ONLY the provided evidence to answer the question.
2. Do NOT invent geological facts or rely on general knowledge.
3. If the evidence is insufficient, say: "The provided geological data does not contain sufficient information to answer this question."
4. Cite your sources by referencing the source number [Source 1], [Source 2], etc.
5. Preserve geological terminology and accuracy.
6. Distinguish between explicit information in sources and any inference you make.
7. If multiple sources agree, mention them: e.g., "According to multiple sources..."
"""
        
        # User message with evidence and question
        user_message = f"""Evidence:
{evidence}

Question: {question}

Please answer using ONLY the above evidence. If the evidence is insufficient, state that clearly."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.chat_model,
                system_content=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,  # Lower temperature for factual consistency
                max_tokens=1000,
            )
            answer = response.choices[0].message.content
            return answer
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _format_sources(
        self,
        retrieved: List[RetrievedResult],
    ) -> List[Dict[str, Any]]:
        """Format retrieved results as sources.
        
        Args:
            retrieved: List of RetrievedResult objects.
        
        Returns:
            List of source dicts with metadata.
        """
        sources = []
        for i, result in enumerate(retrieved, start=1):
            source = {
                "rank": result.rank,
                "similarity_score": float(result.score),
                "source_number": i,
                "title": result.chunk.metadata.get("title", "Unknown"),
                "source_file": result.chunk.metadata.get("source_file", "Unknown"),
                "source_url": result.chunk.metadata.get("source_url", ""),
                "document_type": result.chunk.metadata.get("document_type", ""),
                "chunk_id": result.chunk.chunk_id,
                "text_preview": result.chunk.text[:200] + "..."
                if len(result.chunk.text) > 200
                else result.chunk.text,
            }
            sources.append(source)
        return sources
