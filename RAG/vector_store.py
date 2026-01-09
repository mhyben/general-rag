"""
FAISS vector database implementation for RAG.
Provides efficient similarity search over document embeddings.
"""
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")


class FAISSVectorStore:
    """FAISS-based vector store for document embeddings."""
    
    def __init__(self, dimension: int, index_path: Optional[str] = None):
        """
        Initialize the FAISS vector store.
        
        Args:
            dimension: Dimension of embedding vectors
            index_path: Optional path to save/load index
        """
        self.dimension = dimension
        self.index_path = index_path
        
        # Create FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(dimension)
        
        # Store document metadata (chunk text, source file, etc.)
        self.metadata: List[dict] = []
        
        # Track if index has been modified
        self._modified = False
    
    def add_documents(self, embeddings: np.ndarray, metadatas: List[dict]):
        """
        Add documents to the vector store.
        
        Args:
            embeddings: Array of embedding vectors (numpy array, shape: [n, dimension])
            metadatas: List of metadata dicts for each document
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )
        
        if len(embeddings) != len(metadatas):
            raise ValueError("Number of embeddings must match number of metadatas")
        
        # Convert to float32 (required by FAISS)
        embeddings = embeddings.astype('float32')
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata.extend(metadatas)
        
        self._modified = True
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[dict, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector (shape: [dimension])
            k: Number of results to return
            
        Returns:
            List of (metadata, distance) tuples, sorted by similarity
        """
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, "
                f"got {query_embedding.shape[0]}"
            )
        
        if self.index.ntotal == 0:
            return []
        
        # Reshape query to [1, dimension]
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Convert distances to similarities (lower distance = higher similarity)
        # FAISS returns L2 distances, convert to similarity scores
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                # Convert distance to similarity (1 / (1 + distance))
                similarity = 1.0 / (1.0 + float(dist))
                results.append((self.metadata[idx], similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def save(self, path: Optional[str] = None):
        """
        Save the index and metadata to disk.
        
        Args:
            path: Optional path to save (uses self.index_path if not provided)
        """
        save_path = path or self.index_path
        if save_path is None:
            raise ValueError("No path provided for saving")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path))
        
        # Save metadata
        metadata_path = save_path.with_suffix('.metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        self._modified = False
    
    def load(self, path: Optional[str] = None):
        """
        Load the index and metadata from disk.
        
        Args:
            path: Optional path to load (uses self.index_path if not provided)
        """
        load_path = path or self.index_path
        if load_path is None:
            raise ValueError("No path provided for loading")
        
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Index file not found: {load_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path))
        
        # Load metadata
        metadata_path = load_path.with_suffix('.metadata.pkl')
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = []
        
        self._modified = False
    
    def get_size(self) -> int:
        """Get the number of documents in the index."""
        return self.index.ntotal
    
    def clear(self):
        """Clear all documents from the index."""
        self.index.reset()
        self.metadata = []
        self._modified = True
    
    def is_modified(self) -> bool:
        """Check if the index has been modified since last save."""
        return self._modified

