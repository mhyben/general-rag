"""
Structure-Aware Chunking for RAG applications.
Preserves document structure (headings, sections, tables) while chunking.
Based on advanced agentic RAG pipeline principles.
"""
from typing import List, Dict, Any, Optional
import re
from pathlib import Path


class StructureAwareChunker:
    """
    Structure-aware chunker that preserves document hierarchy and semantic boundaries.
    
    This chunker:
    - Detects document structure (headings, sections, paragraphs)
    - Preserves hierarchical relationships
    - Maintains context from parent sections
    - Chunks at semantic boundaries rather than fixed sizes
    """
    
    def __init__(self, max_chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        """
        Initialize the structure-aware chunker.
        
        Args:
            max_chunk_size: Maximum chunk size in characters (optional, for very large sections)
            overlap: Overlap size between chunks in characters (optional, rarely needed with structure-aware)
        """
        self.max_chunk_size = max_chunk_size  # Used as a limit, not a target
        self.overlap = overlap if overlap is not None else 0
        
        # Regex patterns for detecting document structure
        self.heading_patterns = [
            re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),  # Markdown headings
            re.compile(r'^\d+\.\s+[A-Z].+$', re.MULTILINE),  # Numbered headings
            re.compile(r'^[A-Z][A-Z\s]{3,}$', re.MULTILINE),  # ALL CAPS headings
        ]
    
    def detect_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect document structure (headings, sections, paragraphs).
        
        Args:
            text: Document text to analyze
            
        Returns:
            List of structure elements with their positions and types
        """
        structure = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Detect markdown headings
            if line_stripped.startswith('#'):
                level = len(line_stripped) - len(line_stripped.lstrip('#'))
                structure.append({
                    'type': 'heading',
                    'level': level,
                    'text': line_stripped.lstrip('#').strip(),
                    'line': i,
                    'position': sum(len(l) + 1 for l in lines[:i])  # Character position
                })
            # Detect numbered headings
            elif re.match(r'^\d+\.\s+[A-Z]', line_stripped):
                structure.append({
                    'type': 'heading',
                    'level': 2,
                    'text': line_stripped,
                    'line': i,
                    'position': sum(len(l) + 1 for l in lines[:i])
                })
            # Detect ALL CAPS headings (likely section headers)
            elif len(line_stripped) > 3 and line_stripped.isupper() and ' ' in line_stripped:
                structure.append({
                    'type': 'heading',
                    'level': 3,
                    'text': line_stripped,
                    'line': i,
                    'position': sum(len(l) + 1 for l in lines[:i])
                })
            # Detect tables (lines with multiple | characters)
            elif '|' in line_stripped and line_stripped.count('|') >= 2:
                structure.append({
                    'type': 'table',
                    'text': line_stripped,
                    'line': i,
                    'position': sum(len(l) + 1 for l in lines[:i])
                })
        
        return structure
    
    def get_section_context(self, position: int, structure: List[Dict[str, Any]]) -> str:
        """
        Get the hierarchical context (parent headings) for a given position.
        
        Args:
            position: Character position in document
            structure: List of detected structure elements
            
        Returns:
            Context string with parent headings
        """
        context_parts = []
        current_level = 999
        
        for elem in structure:
            if elem['position'] > position:
                break
            
            if elem['type'] == 'heading':
                # Only include headings that are parents (higher level)
                if elem['level'] < current_level:
                    context_parts.append(elem['text'])
                    current_level = elem['level']
        
        if context_parts:
            return ' > '.join(context_parts)
        return ''
    
    def chunk_document(self, text: str, source: str = "document") -> List[Dict[str, Any]]:
        """
        Chunk a document using structure-aware approach.
        
        Args:
            text: Document text to chunk
            source: Source document identifier
            
        Returns:
            List of chunks with metadata including structure context
        """
        # Detect document structure
        structure = self.detect_structure(text)
        
        # If no structure detected, fall back to semantic chunking
        if not structure:
            return self._semantic_chunk(text, source)
        
        chunks = []
        text_length = len(text)
        current_pos = 0
        
        # Group content by sections (between headings)
        sections = []
        for i, elem in enumerate(structure):
            if elem['type'] == 'heading':
                # Get section start
                start_pos = elem['position']
                
                # Get section end (next heading or end of document)
                if i + 1 < len(structure):
                    end_pos = structure[i + 1]['position']
                else:
                    end_pos = text_length
                
                sections.append({
                    'heading': elem,
                    'start': start_pos,
                    'end': end_pos,
                    'content': text[start_pos:end_pos]
                })
        
        # If we have sections, chunk within each section
        if sections:
            for section in sections:
                section_chunks = self._chunk_section(
                    section['content'],
                    section['heading'],
                    source,
                    structure
                )
                chunks.extend(section_chunks)
        else:
            # Fall back to semantic chunking
            chunks = self._semantic_chunk(text, source)
        
        return chunks
    
    def _chunk_section(self, section_text: str, heading: Dict[str, Any], 
                      source: str, structure: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a section while preserving its heading context.
        
        Args:
            section_text: Text of the section
            heading: Heading information for this section
            source: Source document identifier
            structure: Full document structure
            
        Returns:
            List of chunks from this section
        """
        chunks = []
        
        # Get full hierarchical context
        context = self.get_section_context(heading['position'], structure)
        if not context:
            context = heading['text']
        
        # Split section into paragraphs
        paragraphs = [p.strip() for p in section_text.split('\n\n') if p.strip()]
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If paragraph itself is larger than max_chunk_size, split it (only if max_chunk_size is set)
            if self.max_chunk_size and para_size > self.max_chunk_size:
                # Save current chunk if any
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append({
                        'content': chunk_text,
                        'source': source,
                        'context': context,
                        'heading': heading['text'],
                        'heading_level': heading['level']
                    })
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph
                para_chunks = self._split_large_paragraph(para, context, heading, source)
                chunks.extend(para_chunks)
            else:
                # Check if adding this paragraph would exceed max_chunk_size (only if set)
                # Otherwise, keep adding paragraphs to maintain semantic coherence
                if self.max_chunk_size and current_size + para_size + 2 > self.max_chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append({
                        'content': chunk_text,
                        'source': source,
                        'context': context,
                        'heading': heading['text'],
                        'heading_level': heading['level']
                    })
                    
                    # Start new chunk with optional overlap
                    # Note: Overlap is less important with structure-aware chunking
                    # since boundaries are natural, but can help with very large sections
                    if self.overlap and self.overlap > 0 and current_chunk:
                        # Include last paragraph(s) for overlap
                        overlap_text = '\n\n'.join(current_chunk[-1:])
                        if len(overlap_text) <= self.overlap:
                            current_chunk = [overlap_text]
                            current_size = len(overlap_text)
                        else:
                            current_chunk = []
                            current_size = 0
                    else:
                        current_chunk = []
                        current_size = 0
                
                # Add paragraph to current chunk
                current_chunk.append(para)
                current_size += para_size + 2  # +2 for \n\n
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'source': source,
                'context': context,
                'heading': heading['text'],
                'heading_level': heading['level']
            })
        
        return chunks
    
    def _split_large_paragraph(self, para: str, context: str, 
                               heading: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
        """Split a large paragraph into smaller chunks."""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', para)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sent_size = len(sentence)
            
            if self.max_chunk_size and current_size + sent_size > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'source': source,
                    'context': context,
                    'heading': heading['text'],
                    'heading_level': heading['level']
                })
                
                # Optional overlap handling
                if self.overlap and self.overlap > 0:
                    overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) >= 2 else current_chunk[-1]
                    if len(overlap_text) <= self.overlap:
                        current_chunk = [overlap_text]
                        current_size = len(overlap_text)
                    else:
                        current_chunk = []
                        current_size = 0
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sent_size + 1
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'source': source,
                'context': context,
                'heading': heading['text'],
                'heading_level': heading['level']
            })
        
        return chunks
    
    def _semantic_chunk(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        Fallback semantic chunking when no structure is detected.
        
        Args:
            text: Document text
            source: Source document identifier
            
        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        # Use max_chunk_size if set, otherwise use a reasonable default for fallback
        chunk_size = self.max_chunk_size if self.max_chunk_size else 1000
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence endings near the chunk boundary
                for boundary in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_boundary = text.rfind(boundary, start, end)
                    if last_boundary > start + self.chunk_size * 0.5:  # Don't break too early
                        end = last_boundary + len(boundary)
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'content': chunk_text,
                    'source': source,
                    'context': '',
                    'heading': '',
                    'heading_level': 0
                })
            
            # Move start position with overlap
            start = end - self.overlap if self.overlap > 0 else end
        
        return chunks
