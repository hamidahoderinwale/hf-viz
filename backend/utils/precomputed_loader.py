"""
Loader for pre-computed embeddings and UMAP coordinates.
This module provides fast loading of pre-computed data from Parquet files.
Supports downloading from HuggingFace Hub if local files are not available.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# HuggingFace dataset for precomputed data
HF_PRECOMPUTED_DATASET = os.getenv("HF_PRECOMPUTED_DATASET", "modelbiome/hf-viz-precomputed")


class PrecomputedDataLoader:
    """Load pre-computed embeddings and coordinates from Parquet files."""
    
    def __init__(self, data_dir: str = "precomputed_data", version: str = "v1"):
        """
        Initialize the loader.
        
        Args:
            data_dir: Directory containing pre-computed files
            version: Version tag to load (default: v1)
        """
        self.data_dir = Path(data_dir)
        self.version = version
        self.metadata = None
        
    def load_metadata(self) -> Dict:
        """Load metadata about the pre-computed data."""
        metadata_file = self.data_dir / f"metadata_{self.version}.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}\n"
                f"Run scripts/precompute_data.py first to generate pre-computed data."
            )
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Loaded metadata for version {self.version}")
        logger.info(f"  Created: {self.metadata.get('created_at')}")
        logger.info(f"  Total models: {self.metadata.get('total_models'):,}")
        logger.info(f"  Embedding dim: {self.metadata.get('embedding_dim')}")
        
        return self.metadata
    
    def check_available(self) -> bool:
        """Check if pre-computed data is available."""
        metadata_file = self.data_dir / f"metadata_{self.version}.json"
        models_file = self.data_dir / f"models_{self.version}.parquet"
        
        # Embeddings file is optional - coordinates are in models file
        return (
            metadata_file.exists() and
            models_file.exists()
        )
    
    def load_models(self) -> pd.DataFrame:
        """
        Load pre-computed model data with coordinates.
        
        Returns:
            DataFrame with columns: model_id, library_name, pipeline_tag, downloads, likes,
                                   x_3d, y_3d, z_3d, x_2d, y_2d, etc.
        """
        models_file = self.data_dir / f"models_{self.version}.parquet"
        
        if not models_file.exists():
            raise FileNotFoundError(
                f"Models file not found: {models_file}\n"
                f"Run scripts/precompute_data.py first to generate pre-computed data."
            )
        
        logger.info(f"Loading pre-computed models from {models_file}...")
        df = pd.read_parquet(models_file)
        
        # Set model_id as index
        if 'model_id' in df.columns:
            df.set_index('model_id', drop=False, inplace=True)
        
        logger.info(f"Loaded {len(df):,} models with pre-computed coordinates")
        
        return df
    
    def load_embeddings(self) -> Tuple[np.ndarray, pd.Series]:
        """
        Load pre-computed embeddings.
        
        Returns:
            Tuple of (embeddings_array, model_ids_series)
        """
        embeddings_file = self.data_dir / f"embeddings_{self.version}.parquet"
        
        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_file}\n"
                f"Run scripts/precompute_data.py first to generate pre-computed data."
            )
        
        logger.info(f"Loading pre-computed embeddings from {embeddings_file}...")
        df = pd.read_parquet(embeddings_file)
        
        # Convert embeddings from list to numpy array
        embeddings = np.array(df['embedding'].tolist())
        model_ids = df['model_id']
        
        logger.info(f"Loaded embeddings: {embeddings.shape}")
        
        return embeddings, model_ids
    
    def load_all(self) -> Tuple[pd.DataFrame, Optional[np.ndarray], Dict]:
        """
        Load all pre-computed data.
        
        Returns:
            Tuple of (models_df, embeddings_array_or_None, metadata_dict)
        """
        metadata = self.load_metadata()
        df = self.load_models()
        
        # Try to load embeddings, but they're optional
        embeddings_file = self.data_dir / f"embeddings_{self.version}.parquet"
        if embeddings_file.exists():
            embeddings, _ = self.load_embeddings()
        else:
            logger.info("Embeddings file not found, skipping...")
            embeddings = None
        
        return df, embeddings, metadata


def download_from_hf_hub(data_dir: str, version: str = "v1") -> bool:
    """
    Download precomputed data from HuggingFace Hub.
    
    Args:
        data_dir: Directory to save downloaded files
        version: Version tag
    
    Returns:
        True if download successful, False otherwise
    """
    try:
        from huggingface_hub import hf_hub_download, HfApi
        
        dataset_id = HF_PRECOMPUTED_DATASET
        logger.info(f"Attempting to download precomputed data from {dataset_id}...")
        
        api = HfApi()
        
        # Check if the dataset exists
        try:
            api.dataset_info(dataset_id)
        except Exception:
            logger.info(f"Dataset {dataset_id} not found, skipping download.")
            return False
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Download metadata
        try:
            metadata_path = hf_hub_download(
                repo_id=dataset_id,
                filename=f"metadata_{version}.json",
                repo_type="dataset",
                local_dir=data_dir
            )
            logger.info(f"Downloaded metadata to {metadata_path}")
        except Exception as e:
            logger.warning(f"Could not download metadata: {e}")
            return False
        
        # Download models parquet
        try:
            models_path = hf_hub_download(
                repo_id=dataset_id,
                filename=f"models_{version}.parquet",
                repo_type="dataset",
                local_dir=data_dir
            )
            logger.info(f"Downloaded models to {models_path}")
        except Exception as e:
            logger.warning(f"Could not download models parquet: {e}")
            return False
        
        # Optionally download embeddings
        try:
            hf_hub_download(
                repo_id=dataset_id,
                filename=f"embeddings_{version}.parquet",
                repo_type="dataset",
                local_dir=data_dir
            )
            logger.info("Downloaded embeddings parquet")
        except Exception:
            logger.info("Embeddings file not available (optional)")
        
        return True
        
    except ImportError:
        logger.warning("huggingface_hub not installed, cannot download precomputed data")
        return False
    except Exception as e:
        logger.warning(f"Failed to download precomputed data: {e}")
        return False


def get_precomputed_loader(
    data_dir: Optional[str] = None,
    version: str = "v1"
) -> Optional[PrecomputedDataLoader]:
    """
    Get a PrecomputedDataLoader if pre-computed data is available.
    Will attempt to download from HuggingFace Hub if not found locally.
    
    Args:
        data_dir: Directory containing pre-computed files (default: auto-detect)
        version: Version tag to load
    
    Returns:
        PrecomputedDataLoader if available, None otherwise
    """
    if data_dir is None:
        # Try multiple locations
        backend_dir = Path(__file__).parent.parent
        root_dir = backend_dir.parent
        
        possible_dirs = [
            root_dir / "precomputed_data",
            backend_dir / "precomputed_data",
            Path("precomputed_data"),
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists():
                loader = PrecomputedDataLoader(data_dir=str(dir_path), version=version)
                if loader.check_available():
                    logger.info(f"Found pre-computed data in: {dir_path}")
                    return loader
        
        # Try to download from HF Hub
        download_dir = root_dir / "precomputed_data"
        if download_from_hf_hub(str(download_dir), version):
            loader = PrecomputedDataLoader(data_dir=str(download_dir), version=version)
            if loader.check_available():
                logger.info(f"Successfully loaded pre-computed data from HF Hub")
                return loader
        
        return None
    else:
        loader = PrecomputedDataLoader(data_dir=data_dir, version=version)
        if loader.check_available():
            return loader
        
        # Try to download
        if download_from_hf_hub(data_dir, version):
            loader = PrecomputedDataLoader(data_dir=data_dir, version=version)
            if loader.check_available():
                return loader
        
        return None





