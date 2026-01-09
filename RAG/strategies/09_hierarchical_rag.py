"""Hierarchical RAG - Search small chunks, return big parents with metadata"""
from typing import List, Dict, Any

from .base_strategy import BaseRAGStrategy


class HierarchicalRAGStrategy(BaseRAGStrategy):
    """Hierarchical RAG: search small chunks, return large parent chunks with context."""
    
    def answer(self, query: str, **kwargs) -> str:
        """
        Answer query using hierarchical retrieval.
        Note: Hierarchical structure should be set up during ingestion.
        This strategy searches child chunks but returns parent chunks.
        
        Args:
            query: User query
            **kwargs: Additional parameters
        """
        # Retrieve documents (should return parent chunks when child chunks match)
        results = self._retrieve_documents(query, k=5)
        
        # Format with hierarchical context (heading, section info)
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, (metadata, similarity) in enumerate(results, 1):
            content = metadata.get('content', '')
            source = metadata.get('source', 'Unknown')
            heading = metadata.get('heading', '')
            section_type = metadata.get('type', '')
            
            if heading:
                context_parts.append(f"[{heading}] ({section_type})\n{content}\n")
            else:
                context_parts.append(f"[Document {i} from {source}]\n{content}\n")
        
        context = "\n".join(context_parts)
        
        system_prompt = "You are a helpful RAG assistant with hierarchical retrieval capabilities."
        prompt = f"""Context (from hierarchical document structure):
{context}

Question: {query}

Answer:"""
        
        answer = self.llm.generate(prompt, system_prompt=system_prompt)
        return answer
