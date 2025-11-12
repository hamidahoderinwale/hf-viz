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
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded!")
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
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
        if show_progress:
            print(f"Generating embeddings for {len(texts)} models...")
        
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
        print(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from disk."""
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Embeddings loaded from {filepath}")
        return embeddings

