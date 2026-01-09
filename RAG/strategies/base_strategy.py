"""
Base class for RAG strategies.
All RAG strategies should inherit from this class.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports when running as script
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from embedding_loader import EmbeddingLoader
from agent_llm import AgentLLM
from vector_store import FAISSVectorStore


class BaseRAGStrategy(ABC):
    """Base class for all RAG strategies."""
    
    def __init__(
        self,
        embedder: EmbeddingLoader,
        llm: AgentLLM,
        vector_store: FAISSVectorStore,
        **kwargs
    ):
        """
        Initialize the RAG strategy.
        
        Args:
            embedder: Embedding model loader
            llm: LLM interface (Ollama)
            vector_store: Vector database for document storage
            **kwargs: Strategy-specific parameters
        """
        self.embedder = embedder
        self.llm = llm
        self.vector_store = vector_store
        self.config = kwargs
    
    @abstractmethod
    def answer(self, query: str, **kwargs) -> str:
        """
        Answer a query using this RAG strategy.
        
        Args:
            query: User query
            **kwargs: Strategy-specific parameters
            
        Returns:
            Answer string
        """
        pass
    
    def _retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Basic retrieval method (can be overridden by strategies).
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            List of (metadata, similarity) tuples
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, k=k)
        return results
    
    def _format_context(self, results: List[tuple]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            results: List of (metadata, similarity) tuples
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, (metadata, similarity) in enumerate(results, 1):
            content = metadata.get('content', '')
            source = metadata.get('source', 'Unknown')
            context_parts.append(f"[Document {i} from {source}]\n{content}\n")
        
        return "\n".join(context_parts)

