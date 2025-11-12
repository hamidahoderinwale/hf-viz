"""
Dimensionality reduction for visualization (UMAP, t-SNE).
"""
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE
from typing import Optional
import pickle
import os


class DimensionReducer:
    """Reduce high-dimensional embeddings to 2D/3D for visualization."""
    
    def __init__(self, method: str = "umap", n_components: int = 2):
        """
        Initialize reducer.
        
        Args:
            method: 'umap' or 'tsne'
            n_components: Number of dimensions (2 or 3)
        """
        self.method = method.lower()
        self.n_components = n_components
        
        if self.method == "umap":
            self.reducer = UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
        elif self.method == "tsne":
            self.reducer = TSNE(
                n_components=n_components,
                perplexity=30,
                random_state=42,
                n_iter=1000
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'umap' or 'tsne'")
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit reducer and transform embeddings.
        
        Args:
            embeddings: High-dimensional embeddings (n_samples, embedding_dim)
            
        Returns:
            Reduced embeddings (n_samples, n_components)
        """
        print(f"Reducing dimensions using {self.method.upper()}...")
        reduced = self.reducer.fit_transform(embeddings)
        print(f"Reduced to {self.n_components}D: shape {reduced.shape}")
        return reduced
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings (only for UMAP, t-SNE doesn't support this)."""
        if self.method == "umap":
            return self.reducer.transform(embeddings)
        else:
            raise ValueError("t-SNE doesn't support transform. Use fit_transform instead.")
    
    def save_reducer(self, filepath: str):
        """Save fitted reducer to disk."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.reducer, f)
        print(f"Reducer saved to {filepath}")
    
    def load_reducer(self, filepath: str):
        """Load fitted reducer from disk."""
        with open(filepath, 'rb') as f:
            self.reducer = pickle.load(f)
        print(f"Reducer loaded from {filepath}")

