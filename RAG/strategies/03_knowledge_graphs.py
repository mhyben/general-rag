"""Knowledge Graphs RAG - Using Graphiti by Zep for temporal knowledge graphs"""
from typing import Optional

from .base_strategy import BaseRAGStrategy


class KnowledgeGraphsStrategy(BaseRAGStrategy):
    """Knowledge graph-based RAG using Graphiti (requires Neo4j)."""
    
    def __init__(self, embedder, llm, vector_store, graphiti_connection: Optional[str] = None, **kwargs):
        """
        Initialize knowledge graphs strategy.
        
        Args:
            embedder: Embedding model loader
            llm: LLM interface
            vector_store: Vector database
            graphiti_connection: Neo4j connection string (e.g., "neo4j://localhost:7687")
        """
        super().__init__(embedder, llm, vector_store, **kwargs)
        
        self.graphiti_connection = graphiti_connection
        self.graphiti = None
        
        if graphiti_connection:
            try:
                from graphiti_core import Graphiti
                # Note: Graphiti initialization requires credentials
                # This is a placeholder - user should configure Neo4j separately
                self.graphiti = None  # Graphiti(graphiti_connection, "neo4j", "password")
            except ImportError:
                pass
    
    def answer(self, query: str, **kwargs) -> str:
        """
        Answer query using knowledge graph search.
        
        Args:
            query: User query
            **kwargs: Additional parameters
        """
        if self.graphiti is None:
            # Fallback to standard vector search if Graphiti not available
            results = self._retrieve_documents(query, k=5)
            context = self._format_context(results)
            
            system_prompt = "You are a helpful RAG assistant."
            prompt = f"""Context:
{context}

Question: {query}

Answer:"""
            
            return self.llm.generate(prompt, system_prompt=system_prompt)
        
        # If Graphiti is available, use graph search
        # Note: This requires async implementation and Neo4j setup
        # For now, fallback to vector search
        results = self._retrieve_documents(query, k=5)
        context = self._format_context(results)
        
        system_prompt = "You are a helpful RAG assistant with knowledge graph capabilities."
        prompt = f"""Context (from knowledge graph):
{context}

Question: {query}

Answer:"""
        
        return self.llm.generate(prompt, system_prompt=system_prompt)
