"""Context-Aware Chunking - Semantic boundaries using embedding similarity"""
import numpy as np
from typing import List

from .base_strategy import BaseRAGStrategy


class ContextAwareChunkingStrategy(BaseRAGStrategy):
    """Context-aware chunking: chunks respect semantic boundaries."""
    
    def answer(self, query: str, **kwargs) -> str:
        """
        Answer query using context-aware chunking.
        Note: Chunking should be done during ingestion.
        This strategy uses the already-semantically-chunked documents.
        
        Args:
            query: User query
            **kwargs: Additional parameters
        """
        # Retrieve documents (chunks should already be semantically coherent)
        results = self._retrieve_documents(query, k=5)
        context = self._format_context(results)
        
        system_prompt = "You are a helpful RAG assistant. The context comes from semantically-coherent document chunks."
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.llm.generate(prompt, system_prompt=system_prompt)
        return answer
