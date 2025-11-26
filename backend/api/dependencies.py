"""Shared dependencies for API routes."""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from utils.data_loader import ModelDataLoader
from utils.embeddings import ModelEmbedder
from utils.dimensionality_reduction import DimensionReducer
from utils.graph_embeddings import GraphEmbedder

# Global state (initialized in startup) - these are module-level variables
# that will be updated by main.py during startup
data_loader = ModelDataLoader()
embedder: Optional[ModelEmbedder] = None
graph_embedder: Optional[GraphEmbedder] = None
reducer: Optional[DimensionReducer] = None
df: Optional[pd.DataFrame] = None
embeddings: Optional[np.ndarray] = None
graph_embeddings_dict: Optional[Dict[str, np.ndarray]] = None
combined_embeddings: Optional[np.ndarray] = None
reduced_embeddings: Optional[np.ndarray] = None
reduced_embeddings_graph: Optional[np.ndarray] = None
cluster_labels: Optional[np.ndarray] = None

