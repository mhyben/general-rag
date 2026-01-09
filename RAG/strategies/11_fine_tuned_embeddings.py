"""Fine-tuned Embeddings - Custom embedding model for domain-specific retrieval"""
from .base_strategy import BaseRAGStrategy


class FineTunedEmbeddingsStrategy(BaseRAGStrategy):
    """Fine-tuned embeddings: uses domain-specific embedding model."""
    
    def answer(self, query: str, **kwargs) -> str:
        """
        Answer query using fine-tuned embeddings.
        Note: Fine-tuned embedding model should be used during ingestion.
        This strategy uses the fine-tuned model for retrieval.
        
        Args:
            query: User query
            **kwargs: Additional parameters
        """
        # Retrieve documents (using fine-tuned embeddings)
        results = self._retrieve_documents(query, k=5)
        context = self._format_context(results)
        
        system_prompt = "You are a helpful RAG assistant using fine-tuned embeddings for domain-specific retrieval."
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.llm.generate(prompt, system_prompt=system_prompt)
        return answer
