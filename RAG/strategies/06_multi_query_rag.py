"""Multi-Query RAG - Parallel searches with multiple reformulations"""
from typing import List, Set, Dict, Any

from .base_strategy import BaseRAGStrategy


class MultiQueryRAGStrategy(BaseRAGStrategy):
    """Multi-query RAG: generate multiple query variations and search in parallel."""
    
    def _generate_query_variations(self, query: str, num_variations: int = 3) -> List[str]:
        """Generate multiple query variations."""
        variation_prompt = f"""Generate {num_variations} different variations of this query, each from a different perspective or angle.
Original query: {query}

Provide {num_variations} variations, one per line:"""
        
        variations_text = self.llm.generate(variation_prompt, temperature=0.7)
        
        # Parse variations (one per line)
        variations = [v.strip() for v in variations_text.split('\n') if v.strip()]
        variations = [v for v in variations if v and not v.startswith('#')]
        
        # Include original query
        all_queries = [query] + variations[:num_variations]
        
        return all_queries
    
    def answer(self, query: str, num_variations: int = 3, k: int = 5, **kwargs) -> str:
        """
        Answer query using multiple query variations.
        
        Args:
            query: User query
            num_variations: Number of query variations to generate
            k: Number of results per query
            **kwargs: Additional parameters
        """
        # Generate query variations
        queries = self._generate_query_variations(query, num_variations)
        
        # Search with each variation
        all_results: Dict[str, float] = {}  # content -> max similarity
        
        for q in queries:
            results = self._retrieve_documents(q, k=k)
            for metadata, similarity in results:
                content = metadata.get('content', '')
                # Keep highest similarity score for each unique content
                if content not in all_results or similarity > all_results[content]:
                    all_results[content] = similarity
        
        # Convert to list of (metadata, similarity) tuples
        unique_results = []
        for content, similarity in all_results.items():
            unique_results.append(({'content': content, 'source': 'multi-query'}, similarity))
        
        # Sort by similarity
        unique_results.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k
        top_results = unique_results[:k]
        context = self._format_context(top_results)
        
        system_prompt = "You are a helpful RAG assistant with multi-query retrieval capabilities."
        prompt = f"""Context (from multiple query perspectives):
{context}

Question: {query}

Answer:"""
        
        answer = self.llm.generate(prompt, system_prompt=system_prompt)
        return answer
