"""Late Chunking - Embed full document first, then chunk token embeddings"""
from .base_strategy import BaseRAGStrategy


class LateChunkingStrategy(BaseRAGStrategy):
    """Late chunking: chunks preserve full document context in embeddings."""
    
    def answer(self, query: str, **kwargs) -> str:
        """
        Answer query using late chunking.
        Note: Late chunking should be done during ingestion.
        This strategy uses chunks that preserve full document context.
        
        Args:
            query: User query
            **kwargs: Additional parameters
        """
        # Retrieve documents (chunks should preserve full doc context)
        results = self._retrieve_documents(query, k=5)
        context = self._format_context(results)
        
        system_prompt = "You are a helpful RAG assistant. The context comes from chunks that preserve full document context."
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.llm.generate(prompt, system_prompt=system_prompt)
        return answer
