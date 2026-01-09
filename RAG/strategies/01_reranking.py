"""Re-ranking RAG - Two-stage retrieval with cross-encoder"""
from typing import List, Dict, Any
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from .base_strategy import BaseRAGStrategy


class RerankingStrategy(BaseRAGStrategy):
    """Two-stage retrieval: fast vector search + accurate reranking with cross-encoder."""
    
    def __init__(self, embedder, llm, vector_store, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", **kwargs):
        """
        Initialize reranking strategy.
        
        Args:
            embedder: Embedding model loader
            llm: LLM interface
            vector_store: Vector database
            reranker_model: Cross-encoder model for reranking
        """
        super().__init__(embedder, llm, vector_store, **kwargs)
        
        if CrossEncoder is None:
            raise ImportError("sentence-transformers required for reranking. Install: pip install sentence-transformers")
        
        self.reranker = CrossEncoder(reranker_model)
        self.candidate_multiplier = kwargs.get('candidate_multiplier', 4)  # Retrieve 4x candidates
    
    def answer(self, query: str, k: int = 5, **kwargs) -> str:
        """
        Answer query using two-stage retrieval with reranking.
        
        Args:
            query: User query
            k: Number of final results to return
            **kwargs: Additional parameters
        """
        # Stage 1: Fast vector search (retrieve more candidates)
        candidate_k = min(k * self.candidate_multiplier, 20)
        candidates = self._retrieve_documents(query, k=candidate_k)
        
        if not candidates:
            return "No relevant documents found."
        
        # Stage 2: Re-rank with cross-encoder
        query_doc_pairs = [[query, metadata['content']] for metadata, _ in candidates]
        rerank_scores = self.reranker.predict(query_doc_pairs)
        
        # Combine with original results and sort by rerank score
        reranked = list(zip(candidates, rerank_scores))
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k results
        top_results = [result for result, _ in reranked[:k]]
        
        # Format context
        context = self._format_context(top_results)
        
        # Generate answer
        system_prompt = "You are a helpful RAG assistant. Answer the question based on the provided context."
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.llm.generate(prompt, system_prompt=system_prompt)
        return answer
