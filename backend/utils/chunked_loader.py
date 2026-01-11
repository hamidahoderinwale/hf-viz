"""
Chunked embedding loader for scalable model embeddings.
Loads embeddings in chunks to reduce memory usage and startup time.
"""
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ChunkedEmbeddingLoader:
    """
    Load embeddings from chunked parquet files.
    Only loads chunks containing requested model IDs.
    """
    
    def __init__(self, data_dir: str = "precomputed_data", version: str = "v1", chunk_size: int = 50000):
        """
        Initialize chunked loader.
        
        Args:
            data_dir: Directory containing pre-computed files
            version: Version tag
            chunk_size: Number of models per chunk
        """
        self.data_dir = Path(data_dir)
        self.version = version
        self.chunk_size = chunk_size
        self.chunk_index: Optional[pd.DataFrame] = None
        self._chunk_cache: Dict[int, pd.DataFrame] = {}
        self._max_cache_size = 10  # Cache up to 10 chunks in memory
        
    def load_chunk_index(self) -> pd.DataFrame:
        """Load the chunk index mapping model_id to chunk_id."""
        index_file = self.data_dir / f"chunk_index_{self.version}.parquet"
        
        if not index_file.exists():
            raise FileNotFoundError(
                f"Chunk index not found: {index_file}\n"
                f"Run precompute_data.py with --chunked flag to generate chunked data."
            )
        
        logger.info(f"Loading chunk index from {index_file}...")
        self.chunk_index = pd.read_parquet(index_file)
        logger.info(f"Loaded chunk index: {len(self.chunk_index):,} models in {self.chunk_index['chunk_id'].nunique()} chunks")
        
        return self.chunk_index
    
    def _load_chunk(self, chunk_id: int) -> pd.DataFrame:
        """Load a single chunk file."""
        # Check cache first
        if chunk_id in self._chunk_cache:
            return self._chunk_cache[chunk_id]
        
        chunk_file = self.data_dir / f"embeddings_chunk_{chunk_id:03d}_{self.version}.parquet"
        
        if not chunk_file.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_file}")
        
        logger.debug(f"Loading chunk {chunk_id} from {chunk_file}...")
        chunk_df = pd.read_parquet(chunk_file)
        
        # Cache management: remove oldest if cache is full
        if len(self._chunk_cache) >= self._max_cache_size:
            oldest_chunk = min(self._chunk_cache.keys())
            del self._chunk_cache[oldest_chunk]
        
        self._chunk_cache[chunk_id] = chunk_df
        return chunk_df
    
    def load_embeddings_for_models(
        self, 
        model_ids: List[str],
        return_as_dict: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load embeddings only for specified model IDs.
        
        Args:
            model_ids: List of model IDs to load
            return_as_dict: If True, return dict mapping model_id to embedding
        
        Returns:
            Tuple of (embeddings_array, model_ids_found)
            If return_as_dict=True, returns (embeddings_dict, model_ids_found)
        """
        if self.chunk_index is None:
            self.load_chunk_index()
        
        # Convert to set for faster lookup
        requested_ids = set(model_ids)
        
        # Find which chunks contain these models
        model_chunks = self.chunk_index[
            self.chunk_index['model_id'].isin(requested_ids)
        ]
        
        if len(model_chunks) == 0:
            logger.warning(f"No embeddings found for {len(model_ids)} requested models")
            return (np.array([]), []) if not return_as_dict else ({}, [])
        
        # Group by chunk_id and load chunks
        embeddings_dict = {}
        found_ids = []
        
        for chunk_id in model_chunks['chunk_id'].unique():
            chunk_df = self._load_chunk(chunk_id)
            
            # Filter to requested models in this chunk
            chunk_model_ids = model_chunks[model_chunks['chunk_id'] == chunk_id]['model_id'].tolist()
            chunk_embeddings = chunk_df[chunk_df['model_id'].isin(chunk_model_ids)]
            
            for _, row in chunk_embeddings.iterrows():
                model_id = row['model_id']
                embedding = np.array(row['embedding'])
                embeddings_dict[model_id] = embedding
                found_ids.append(model_id)
        
        if return_as_dict:
            return embeddings_dict, found_ids
        
        # Convert to array maintaining order
        embeddings_list = [embeddings_dict[mid] for mid in model_ids if mid in embeddings_dict]
        found_ids_ordered = [mid for mid in model_ids if mid in embeddings_dict]
        
        if len(embeddings_list) == 0:
            return np.array([]), []
        
        embeddings_array = np.array(embeddings_list)
        return embeddings_array, found_ids_ordered
    
    def load_all_embeddings(self) -> Tuple[np.ndarray, pd.Series]:
        """
        Load all embeddings (for backward compatibility).
        Warning: This loads all chunks into memory!
        """
        if self.chunk_index is None:
            self.load_chunk_index()
        
        all_chunk_ids = sorted(self.chunk_index['chunk_id'].unique())
        logger.warning(f"Loading all {len(all_chunk_ids)} chunks - this may use significant memory!")
        
        all_embeddings = []
        all_model_ids = []
        
        for chunk_id in all_chunk_ids:
            chunk_df = self._load_chunk(chunk_id)
            all_embeddings.extend(chunk_df['embedding'].tolist())
            all_model_ids.extend(chunk_df['model_id'].tolist())
        
        embeddings_array = np.array(all_embeddings)
        model_ids_series = pd.Series(all_model_ids)
        
        return embeddings_array, model_ids_series
    
    def get_chunk_info(self) -> Dict:
        """Get information about chunks."""
        if self.chunk_index is None:
            self.load_chunk_index()
        
        chunk_counts = self.chunk_index['chunk_id'].value_counts().sort_index()
        
        return {
            'total_models': len(self.chunk_index),
            'total_chunks': self.chunk_index['chunk_id'].nunique(),
            'chunk_size': self.chunk_size,
            'chunk_counts': chunk_counts.to_dict(),
            'cached_chunks': list(self._chunk_cache.keys())
        }
    
    def clear_cache(self):
        """Clear the chunk cache."""
        self._chunk_cache.clear()
        logger.info("Chunk cache cleared")


def create_chunk_index(
    df: pd.DataFrame,
    chunk_size: int = 50000,
    output_dir: Path = None,
    version: str = "v1"
) -> pd.DataFrame:
    """
    Create chunk index from dataframe.
    
    Args:
        df: DataFrame with model_id column
        chunk_size: Number of models per chunk
        output_dir: Directory to save index
        version: Version tag
    
    Returns:
        DataFrame with columns: model_id, chunk_id, chunk_offset
    """
    model_ids = df['model_id'].astype(str).values
    
    # Assign chunk IDs based on position
    chunk_ids = (np.arange(len(model_ids)) // chunk_size).astype(int)
    chunk_offsets = np.arange(len(model_ids)) % chunk_size
    
    chunk_index = pd.DataFrame({
        'model_id': model_ids,
        'chunk_id': chunk_ids,
        'chunk_offset': chunk_offsets
    })
    
    if output_dir:
        index_file = output_dir / f"chunk_index_{version}.parquet"
        chunk_index.to_parquet(index_file, compression='snappy', index=False)
        logger.info(f"Saved chunk index: {index_file}")
    
    return chunk_index

