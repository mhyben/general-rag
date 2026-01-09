"""Contextual Retrieval - Add document context to chunks (Anthropic)"""
from .base_strategy import BaseRAGStrategy


class ContextualRetrievalStrategy(BaseRAGStrategy):
    """Contextual retrieval: chunks include document-level context."""
    
    def answer(self, query: str, **kwargs) -> str:
        """
        Answer query using contextual retrieval.
        Note: Contextual enrichment should be done during ingestion.
        This strategy uses the already-enriched chunks.
        
        Args:
            query: User query
            **kwargs: Additional parameters
        """
        # Retrieve documents (chunks should already have contextual prefixes)
        results = self._retrieve_documents(query, k=5)
        context = self._format_context(results)
        
        system_prompt = "You are a helpful RAG assistant. Answer based on the contextualized document chunks."
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.llm.generate(prompt, system_prompt=system_prompt)
        return answer
