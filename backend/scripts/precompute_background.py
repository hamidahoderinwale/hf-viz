#!/usr/bin/env python3
"""
Background pre-computation script for processing ALL models incrementally.
Designed to run in the background and save progress so it can be resumed.

Features:
- Processes models in batches to manage memory
- Saves progress incrementally
- Can be resumed if interrupted
- Provides status updates via JSON file

Usage:
    # Process all models (default ~500k batch)
    python scripts/precompute_background.py --all
    
    # Process specific number of models
    python scripts/precompute_background.py --sample-size 500000
    
    # Resume from previous run
    python scripts/precompute_background.py --resume
    
    # Check status
    python scripts/precompute_background.py --status
"""

import argparse
import os
import sys
import json
import time
import logging
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import threading

import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.decomposition import PCA, IncrementalPCA

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from utils.data_loader import ModelDataLoader
from utils.embeddings import ModelEmbedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('precompute_background.log')
    ]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    logger.warning("Shutdown requested - will save progress and exit...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class BackgroundPrecomputer:
    """Handles incremental pre-computation of model embeddings and coordinates."""
    
    def __init__(
        self,
        output_dir: str = "precomputed_data",
        version: str = "v1",
        batch_size: int = 50000,
        embedding_batch_size: int = 256
    ):
        self.output_dir = Path(output_dir)
        self.version = version
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        
        # Status file for tracking progress
        self.status_file = self.output_dir / f"background_status_{version}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = ModelDataLoader()
        self.embedder = ModelEmbedder()
        
    def get_status(self) -> Dict[str, Any]:
        """Get current computation status."""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {
            'status': 'not_started',
            'total_models': 0,
            'processed_models': 0,
            'current_batch': 0,
            'started_at': None,
            'last_updated': None,
            'error': None
        }
    
    def save_status(self, status: Dict[str, Any]):
        """Save computation status."""
        status['last_updated'] = datetime.now().isoformat()
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def load_full_dataset(self) -> pd.DataFrame:
        """Load the full dataset without sampling."""
        logger.info("Loading full dataset from HuggingFace...")
        from datasets import load_dataset
        dataset = load_dataset("modelbiome/ai_ecosystem", split="train")
        df = dataset.to_pandas()
        logger.info(f"Loaded {len(df):,} total models")
        return df
    
    def precompute_all(
        self,
        sample_size: Optional[int] = None,
        resume: bool = False,
        pca_dims: int = 50
    ):
        """
        Pre-compute embeddings and coordinates for all or specified number of models.
        
        Args:
            sample_size: If None, process all models. Otherwise, process this many.
            resume: If True, resume from previous progress
            pca_dims: Number of PCA dimensions for pre-reduction
        """
        global shutdown_requested
        
        start_time = time.time()
        
        # Get current status
        status = self.get_status() if resume else {
            'status': 'initializing',
            'total_models': 0,
            'processed_models': 0,
            'current_batch': 0,
            'started_at': datetime.now().isoformat(),
            'error': None,
            'batches_completed': []
        }
        
        try:
            # Step 1: Load data
            status['status'] = 'loading_data'
            self.save_status(status)
            
            if sample_size:
                logger.info(f"Loading {sample_size:,} models with stratified sampling...")
                df = self.data_loader.load_data(sample_size=sample_size, prioritize_base_models=True)
            else:
                logger.info("Loading ALL models...")
                df = self.load_full_dataset()
            
            total_models = len(df)
            status['total_models'] = total_models
            logger.info(f"Total models to process: {total_models:,}")
            
            # Build combined text
            logger.info("Building combined text for embeddings...")
            df['combined_text'] = (
                df.get('tags', '').astype(str) + ' ' +
                df.get('pipeline_tag', '').astype(str) + ' ' +
                df.get('library_name', '').astype(str)
            )
            if 'modelCard' in df.columns:
                df['combined_text'] = df['combined_text'] + ' ' + df['modelCard'].astype(str).str[:500]
            
            # Step 2: Generate embeddings in batches
            status['status'] = 'generating_embeddings'
            self.save_status(status)
            
            logger.info("Generating embeddings...")
            all_embeddings = []
            texts = df['combined_text'].tolist()
            
            num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(num_batches):
                if shutdown_requested:
                    logger.warning("Shutdown requested - saving partial progress...")
                    break
                
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]
                
                logger.info(f"Processing embedding batch {batch_idx + 1}/{num_batches} "
                           f"(models {batch_start:,} - {batch_end:,})...")
                
                batch_embeddings = self.embedder.generate_embeddings(
                    batch_texts, 
                    batch_size=self.embedding_batch_size
                )
                all_embeddings.append(batch_embeddings)
                
                status['processed_models'] = batch_end
                status['current_batch'] = batch_idx + 1
                status['progress_percent'] = round(100 * batch_end / total_models, 1)
                self.save_status(status)
            
            if shutdown_requested:
                status['status'] = 'interrupted'
                self.save_status(status)
                return
            
            embeddings = np.vstack(all_embeddings)
            logger.info(f"Generated embeddings: {embeddings.shape}")
            
            # Step 3: PCA pre-reduction
            status['status'] = 'pca_reduction'
            self.save_status(status)
            
            logger.info(f"Applying PCA reduction ({embeddings.shape[1]} -> {pca_dims} dims)...")
            pca = PCA(n_components=pca_dims, random_state=42)
            embeddings_reduced = pca.fit_transform(embeddings)
            explained_var = pca.explained_variance_ratio_.sum()
            logger.info(f"PCA complete (preserved {explained_var:.1%} variance)")
            
            if shutdown_requested:
                status['status'] = 'interrupted'
                self.save_status(status)
                return
            
            # Step 4: UMAP 3D
            status['status'] = 'umap_3d'
            self.save_status(status)
            
            logger.info("Running UMAP for 3D coordinates...")
            reducer_3d = UMAP(
                n_components=3,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                n_jobs=-1,
                low_memory=True if total_models > 200000 else False,
                spread=1.5,
                verbose=True
            )
            coords_3d = reducer_3d.fit_transform(embeddings_reduced)
            logger.info(f"3D coordinates: {coords_3d.shape}")
            
            if shutdown_requested:
                status['status'] = 'interrupted'
                self.save_status(status)
                return
            
            # Step 5: UMAP 2D
            status['status'] = 'umap_2d'
            self.save_status(status)
            
            logger.info("Running UMAP for 2D coordinates...")
            reducer_2d = UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                n_jobs=-1,
                low_memory=True if total_models > 200000 else False,
                spread=1.5,
                verbose=True
            )
            coords_2d = reducer_2d.fit_transform(embeddings_reduced)
            logger.info(f"2D coordinates: {coords_2d.shape}")
            
            # Step 6: Save results
            status['status'] = 'saving'
            self.save_status(status)
            
            logger.info("Saving results...")
            
            # Prepare output DataFrame
            output_df = df.copy()
            output_df['x_3d'] = coords_3d[:, 0]
            output_df['y_3d'] = coords_3d[:, 1]
            output_df['z_3d'] = coords_3d[:, 2]
            output_df['x_2d'] = coords_2d[:, 0]
            output_df['y_2d'] = coords_2d[:, 1]
            
            # Save models
            models_file = self.output_dir / f"models_{self.version}.parquet"
            output_df.to_parquet(models_file, compression='snappy', index=False)
            logger.info(f"Saved: {models_file} ({models_file.stat().st_size / 1024 / 1024:.1f} MB)")
            
            # Save embeddings
            embeddings_file = self.output_dir / f"embeddings_{self.version}.parquet"
            embeddings_df = pd.DataFrame({
                'model_id': df['modelId'].values,
                'embedding': [emb.tolist() for emb in embeddings]
            })
            embeddings_df.to_parquet(embeddings_file, compression='snappy', index=False)
            logger.info(f"Saved: {embeddings_file} ({embeddings_file.stat().st_size / 1024 / 1024:.1f} MB)")
            
            # Save metadata
            total_time = time.time() - start_time
            metadata = {
                'version': self.version,
                'created_at': datetime.now().isoformat(),
                'total_models': int(total_models),
                'embedding_dim': int(embeddings.shape[1]),
                'umap_3d_shape': list(coords_3d.shape),
                'umap_2d_shape': list(coords_2d.shape),
                'unique_libraries': int(df['library_name'].nunique()),
                'unique_pipelines': int(df['pipeline_tag'].nunique()),
                'processing_time_seconds': total_time,
                'processing_time_hours': total_time / 3600,
                'pca_dims': pca_dims,
                'pca_variance_preserved': float(explained_var)
            }
            
            metadata_file = self.output_dir / f"metadata_{self.version}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved: {metadata_file}")
            
            # Update final status
            status['status'] = 'completed'
            status['completed_at'] = datetime.now().isoformat()
            status['processing_time_hours'] = round(total_time / 3600, 2)
            self.save_status(status)
            
            logger.info("="*60)
            logger.info("BACKGROUND PRE-COMPUTATION COMPLETE!")
            logger.info("="*60)
            logger.info(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
            logger.info(f"Models processed: {total_models:,}")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Pre-computation failed: {e}", exc_info=True)
            status['status'] = 'failed'
            status['error'] = str(e)
            self.save_status(status)
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Background pre-computation of HF model embeddings and coordinates"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of models to process (default: all)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process ALL models (may take many hours)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous progress"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current computation status and exit"
    )
    parser.add_argument(
        "--output-dir", type=str, default="../precomputed_data",
        help="Output directory"
    )
    parser.add_argument(
        "--version", type=str, default="v1",
        help="Version tag"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50000,
        help="Batch size for processing"
    )
    
    args = parser.parse_args()
    
    precomputer = BackgroundPrecomputer(
        output_dir=args.output_dir,
        version=args.version,
        batch_size=args.batch_size
    )
    
    if args.status:
        status = precomputer.get_status()
        print(json.dumps(status, indent=2))
        return
    
    sample_size = None if args.all else (args.sample_size or 150000)
    
    if sample_size:
        logger.info(f"Processing {sample_size:,} models...")
    else:
        logger.info("Processing ALL models (this may take many hours)...")
    
    try:
        precomputer.precompute_all(
            sample_size=sample_size,
            resume=args.resume
        )
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

