"""Self-Reflective RAG - Iteratively refine with self-assessment"""
from typing import List, Dict, Any

from .base_strategy import BaseRAGStrategy


class SelfReflectiveRAGStrategy(BaseRAGStrategy):
    """Self-reflective RAG: iteratively refine query and results with self-assessment."""
    
    def _grade_relevance(self, query: str, results: List[tuple]) -> float:
        """Grade the relevance of retrieved results."""
        if not results:
            return 0.0
        
        # Format results for grading
        results_text = "\n".join([
            f"- {metadata.get('content', '')[:200]}..." 
            for metadata, _ in results[:3]
        ])
        
        grade_prompt = f"""Query: {query}

Retrieved documents:
{results_text}

Rate the relevance of these documents to the query on a scale of 1-5, where:
1 = Not relevant at all
2 = Slightly relevant
3 = Moderately relevant
4 = Very relevant
5 = Perfectly relevant

Respond with only a number (1-5):"""
        
        grade_response = self.llm.generate(grade_prompt, temperature=0)
        
        # Extract number from response
        try:
            grade = float(grade_response.strip().split()[0])
            return max(1.0, min(5.0, grade)) / 5.0  # Normalize to 0-1
        except:
            return 0.5  # Default to medium relevance
    
    def _refine_query(self, original_query: str, results: List[tuple]) -> str:
        """Refine the query based on initial results."""
        results_text = "\n".join([
            f"- {metadata.get('content', '')[:200]}..." 
            for metadata, _ in results[:3]
        ])
        
        refine_prompt = f"""Original query: {original_query}

Initial results (may not be very relevant):
{results_text}

Suggest an improved, more specific query that would retrieve better results.
Respond with only the improved query:"""
        
        refined = self.llm.generate(refine_prompt, temperature=0.3)
        return refined.strip()
    
    def answer(self, query: str, max_iterations: int = 2, **kwargs) -> str:
        """
        Answer query using self-reflective retrieval.
        
        Args:
            query: User query
            max_iterations: Maximum number of refinement iterations
            **kwargs: Additional parameters
        """
        current_query = query
        results = None
        
        for iteration in range(max_iterations):
            # Retrieve documents
            results = self._retrieve_documents(current_query, k=5)
            
            if not results:
                break
            
            # Grade relevance
            relevance_score = self._grade_relevance(current_query, results)
            
            # If relevance is good enough, proceed to answer
            if relevance_score >= 0.6:  # 3/5 or higher
                break
            
            # Otherwise, refine query
            if iteration < max_iterations - 1:
                current_query = self._refine_query(query, results)
        
        if not results:
            return "No relevant documents found after refinement."
        
        # Format context
        context = self._format_context(results)
        
        # Generate answer
        system_prompt = "You are a helpful RAG assistant with self-reflective capabilities."
        prompt = f"""Context (retrieved with query: "{current_query}"):
{context}

Original Question: {query}

Answer:"""
        
        answer = self.llm.generate(prompt, system_prompt=system_prompt)
        return answer
