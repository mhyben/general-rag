"""Agentic RAG - Agent dynamically chooses tools (vector, SQL, web)"""
from typing import List, Dict, Any
import os
from pathlib import Path

from .base_strategy import BaseRAGStrategy


class AgenticRAGStrategy(BaseRAGStrategy):
    """Agent autonomously chooses between multiple retrieval tools."""
    
    def __init__(self, embedder, llm, vector_store, documents_folder: str = None, **kwargs):
        """
        Initialize agentic RAG strategy.
        
        Args:
            embedder: Embedding model loader
            llm: LLM interface
            vector_store: Vector database
            documents_folder: Path to documents folder for full document retrieval
        """
        super().__init__(embedder, llm, vector_store, **kwargs)
        self.documents_folder = Path(documents_folder) if documents_folder else None
    
    def _vector_search(self, query: str, k: int = 3) -> str:
        """Search unstructured knowledge base using vector search."""
        results = self._retrieve_documents(query, k=k)
        if not results:
            return "No relevant chunks found."
        return self._format_context(results)
    
    def _retrieve_full_document(self, document_title: str) -> str:
        """Retrieve complete document when chunks lack context."""
        if self.documents_folder is None:
            return "Full document retrieval not available (documents_folder not set)."
        
        # Search for matching markdown files
        for md_file in self.documents_folder.glob("*.md"):
            if document_title.lower() in md_file.stem.lower():
                content = md_file.read_text(encoding='utf-8')
                return f"**{md_file.stem}**\n\n{content[:5000]}..."  # Limit length
        
        return f"Document '{document_title}' not found."
    
    def answer(self, query: str, **kwargs) -> str:
        """
        Answer query using agentic tool selection.
        
        Args:
            query: User query
            **kwargs: Additional parameters
        """
        system_prompt = """You are an agentic RAG assistant with access to multiple tools.
You can:
1. Use vector_search(query) to search document chunks
2. Use retrieve_full_document(title) to get complete documents

Choose the appropriate tool(s) based on the query. For specific questions, use vector_search.
For questions requiring full context, use retrieve_full_document."""
        
        # Let LLM decide which tools to use
        tool_prompt = f"""Query: {query}

Available tools:
1. vector_search(query) - Search document chunks
2. retrieve_full_document(title) - Get full document

First, decide which tool(s) to use. Then call them and provide an answer.

If you need to search chunks, respond with: TOOL: vector_search
If you need a full document, respond with: TOOL: retrieve_full_document <title>

Then I will execute the tool and provide results."""
        
        decision = self.llm.generate(tool_prompt, system_prompt=system_prompt)
        
        # Simple tool selection (can be enhanced with function calling)
        if "vector_search" in decision.lower() or "chunk" in decision.lower():
            context = self._vector_search(query, k=5)
        elif "retrieve_full_document" in decision.lower() or "full document" in decision.lower():
            # Extract document title from decision
            # Simple heuristic: use query keywords
            doc_title = query.split()[0] if query.split() else "document"
            context = self._retrieve_full_document(doc_title)
        else:
            # Default to vector search
            context = self._vector_search(query, k=5)
        
        # Generate final answer
        answer_prompt = f"""Context:
{context}

Question: {query}

Answer based on the context:"""
        
        answer = self.llm.generate(answer_prompt, system_prompt="You are a helpful assistant.")
        return answer
