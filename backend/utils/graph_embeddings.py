"""
Graph-aware embeddings for hierarchical model relationships.
Uses Node2Vec to create embeddings that respect family tree structure.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import networkx as nx
import pickle
import os
import logging

logger = logging.getLogger(__name__)

try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
except ImportError:
    NODE2VEC_AVAILABLE = False
    logger.warning("node2vec not available. Install with: pip install node2vec")


class GraphEmbedder:
    """
    Generate graph embeddings that respect hierarchical relationships.
    Combines text embeddings with graph structure embeddings.
    """
    
    def __init__(self, dimensions: int = 128, walk_length: int = 30, num_walks: int = 200):
        """
        Initialize graph embedder.
        
        Args:
            dimensions: Embedding dimensions
            walk_length: Length of random walks
            num_walks: Number of walks per node
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.graph: Optional[nx.DiGraph] = None
        self.embeddings: Optional[np.ndarray] = None
        self.model: Optional[Node2Vec] = None
    
    def build_family_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Build directed graph from family relationships.
        
        Args:
            df: DataFrame with model_id and parent_model columns
            
        Returns:
            NetworkX DiGraph
        """
        graph = nx.DiGraph()
        
        for idx, row in df.iterrows():
            model_id = str(row.get('model_id', idx))
            graph.add_node(model_id)
            
            parent_id = row.get('parent_model')
            if parent_id and pd.notna(parent_id):
                parent_str = str(parent_id)
                if parent_str != 'nan' and parent_str != '':
                    graph.add_edge(parent_str, model_id)
        
        self.graph = graph
        logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph
    
    def generate_graph_embeddings(
        self,
        graph: Optional[nx.DiGraph] = None,
        workers: int = 4
    ) -> Dict[str, np.ndarray]:
        """
        Generate Node2Vec embeddings for graph nodes.
        
        Args:
            graph: NetworkX graph (uses self.graph if None)
            workers: Number of parallel workers
            
        Returns:
            Dictionary mapping model_id to embedding vector
        """
        if not NODE2VEC_AVAILABLE:
            logger.warning("Node2Vec not available, returning empty embeddings")
            return {}
        
        if graph is None:
            graph = self.graph
        
        if graph is None or graph.number_of_nodes() == 0:
            logger.warning("No graph available for embedding generation")
            return {}
        
        try:
            node2vec = Node2Vec(
                graph,
                dimensions=self.dimensions,
                walk_length=self.walk_length,
                num_walks=self.num_walks,
                workers=workers
            )
            
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            self.model = model
            
            embeddings_dict = {}
            for node in graph.nodes():
                if node in model.wv:
                    embeddings_dict[node] = model.wv[node]
            
            logger.info(f"Generated graph embeddings for {len(embeddings_dict)} nodes")
            return embeddings_dict
            
        except Exception as e:
            logger.error(f"Error generating graph embeddings: {e}", exc_info=True)
            return {}
    
    def combine_embeddings(
        self,
        text_embeddings: np.ndarray,
        graph_embeddings: Dict[str, np.ndarray],
        model_ids: List[str],
        text_weight: float = 0.7,
        graph_weight: float = 0.3
    ) -> np.ndarray:
        """
        Combine text and graph embeddings with weighted average.
        
        Args:
            text_embeddings: Text-based embeddings (n_samples, text_dim)
            graph_embeddings: Graph embeddings dictionary
            model_ids: List of model IDs corresponding to text_embeddings
            text_weight: Weight for text embeddings
            graph_weight: Weight for graph embeddings
            
        Returns:
            Combined embeddings (n_samples, combined_dim)
        """
        if not graph_embeddings:
            return text_embeddings
        
        text_dim = text_embeddings.shape[1]
        graph_dim = next(iter(graph_embeddings.values())).shape[0]
        
        combined = np.zeros((len(model_ids), text_dim + graph_dim))
        
        for i, model_id in enumerate(model_ids):
            model_id_str = str(model_id)
            
            text_emb = text_embeddings[i]
            graph_emb = graph_embeddings.get(model_id_str, np.zeros(graph_dim))
            
            normalized_text = text_emb / (np.linalg.norm(text_emb) + 1e-8)
            normalized_graph = graph_emb / (np.linalg.norm(graph_emb) + 1e-8)
            
            combined[i] = np.concatenate([
                normalized_text * text_weight,
                normalized_graph * graph_weight
            ])
        
        return combined
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], filepath: str):
        """Save graph embeddings to disk."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
    
    def load_embeddings(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load graph embeddings from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


