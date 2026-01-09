"""
Embedding models loader for RAG application.
Supports HuggingFace / sentence-transformers models.
"""
from typing import List, Optional
import numpy as np
import os

# Set environment variables to avoid PyTorch CUDA timing issues
# This helps with the "fast_1 >= fast_0" error
os.environ.setdefault('TORCH_USE_CUDA_DSA', '1')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")


class EmbeddingLoader:
    """Load and manage embedding models from HuggingFace/sentence-transformers."""
    
    # Popular embedding models
    POPULAR_MODELS = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    ]
    
    def __init__(self, model_name: str):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            # Try to load with device preference (CPU first to avoid CUDA issues)
            import torch
            device = 'cpu'  # Use CPU by default to avoid CUDA timing issues
            if torch.cuda.is_available():
                try:
                    # Try CUDA but fallback to CPU if there are issues
                    device = 'cuda'
                except:
                    device = 'cpu'
            
            self.model = SentenceTransformer(self.model_name, device=device)
        except RuntimeError as e:
            # If CUDA fails, try CPU explicitly
            if 'cuda' in str(e).lower() or 'gpu' in str(e).lower():
                try:
                    self.model = SentenceTransformer(self.model_name, device='cpu')
                except Exception as e2:
                    raise RuntimeError(f"Failed to load embedding model '{self.model_name}' (tried CUDA and CPU): {str(e2)}")
            else:
                raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {str(e)}")
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple documents.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            
        Returns:
            Array of embedding vectors
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Get dimension by encoding a dummy text
        dummy_embedding = self.embed_query("test")
        return len(dummy_embedding)
    
    @classmethod
    def list_popular_models(cls) -> List[str]:
        """Return list of popular embedding models."""
        return cls.POPULAR_MODELS.copy()

