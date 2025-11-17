"""
Generate embeddings for models using sentence transformers.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import pickle
import os
from tqdm import tqdm


class ModelEmbedder:
    """Generate embeddings for model descriptions."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """
        Initialize embedder.
        
        Args:
            model_name: Sentence transformer model name
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 128,  # Increased default batch size for speed
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings (n_samples, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to disk."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from disk."""
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings

