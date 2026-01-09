"""Query Expansion RAG - Generate multiple query variations for better retrieval"""
from typing import List, Set
from .base_strategy import BaseRAGStrategy


class QueryExpansionStrategy(BaseRAGStrategy):
    """Query expansion: enrich single query into more detailed version."""
    
    def _expand_query(self, query: str) -> str:
        """Expand a brief query into a more detailed, comprehensive version."""
        expansion_prompt = f"""Take this brief query and expand it into a more detailed, comprehensive version that:
1. Adds relevant context and clarifications
2. Includes related terminology and concepts
3. Specifies what aspects should be covered
4. Maintains the original intent
5. Keep it as a single, coherent question

Original query: {query}

Expanded query:"""
        
        expanded = self.llm.generate(expansion_prompt, temperature=0.3)
        return expanded.strip()
    
    def answer(self, query: str, **kwargs) -> str:
        """
        Answer query using query expansion.
        
        Args:
            query: User query
            **kwargs: Additional parameters
        """
        # Expand the query
        expanded_query = self._expand_query(query)
        
        # Retrieve using expanded query
        results = self._retrieve_documents(expanded_query, k=5)
        context = self._format_context(results)
        
        system_prompt = "You are a helpful RAG assistant with query expansion capabilities."
        prompt = f"""Context:
{context}

Original Question: {query}
Expanded Question: {expanded_query}

Answer:"""
        
        answer = self.llm.generate(prompt, system_prompt=system_prompt)
        return answer
