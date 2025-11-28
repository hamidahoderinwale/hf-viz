#!/usr/bin/env python3
"""
FAST pre-computation script with speed optimizations.
~5-10x faster than standard version.

Optimizations:
- No random_state (enables parallel UMAP)
- PCA pre-reduction (384 -> 50 dims)
- Optimized UMAP parameters
- Larger batch sizes

Usage:
    python scripts/precompute_fast.py --sample-size 150000 --output-dir ../precomputed_data
"""

import argparse
import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from umap import UMAP
from sklearn.decomposition import PCA

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from utils.data_loader import ModelDataLoader
from utils.embeddings import ModelEmbedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def precompute_fast(
    sample_size: int = 150000,
    output_dir: str = "precomputed_data",
    version: str = "v1",
    pca_dims: int = 50,
    use_pca: bool = True
):
    """
    Pre-compute embeddings and UMAP coordinates with speed optimizations.
    
    Args:
        sample_size: Number of models to process
        output_dir: Directory to save output files
        version: Version tag for output files
        pca_dims: Number of PCA dimensions (if use_pca=True)
        use_pca: Whether to use PCA pre-reduction (much faster)
    """
    start_time = time.time()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("FAST PRE-COMPUTATION STARTED")
    logger.info("="*60)
    logger.info(f"Sample size: {sample_size:,}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Version: {version}")
    logger.info(f"PCA pre-reduction: {use_pca} ({pca_dims} dims)" if use_pca else "PCA: disabled")
    logger.info("="*60)
    
    # Step 1: Load data with methodological sampling
    logger.info("Step 1/5: Loading model data (prioritizing base models)...")
    step_start = time.time()
    
    data_loader = ModelDataLoader()
    df = data_loader.load_data(sample_size=sample_size, prioritize_base_models=True)
    
    step_time = time.time() - step_start
    logger.info(f"Loaded {len(df):,} models in {step_time:.1f} seconds")
    
    # Step 2: Generate embeddings
    logger.info("Step 2/5: Generating embeddings...")
    step_start = time.time()
    
    # Build combined text from available fields
    logger.info("Building combined text from model fields...")
    df['combined_text'] = (
        df.get('tags', '').astype(str) + ' ' +
        df.get('pipeline_tag', '').astype(str) + ' ' +
        df.get('library_name', '').astype(str)
    )
    
    # Add modelCard if available
    if 'modelCard' in df.columns:
        df['combined_text'] = df['combined_text'] + ' ' + df['modelCard'].astype(str).str[:500]
    
    embedder = ModelEmbedder()
    texts = df['combined_text'].tolist()
    
    # Use larger batch size for speed
    embeddings = embedder.generate_embeddings(texts, batch_size=256)
    
    step_time = time.time() - step_start
    logger.info(f"Generated embeddings: {embeddings.shape} in {step_time/60:.1f} minutes")
    
    # Optional: PCA pre-reduction for speed
    embeddings_for_umap = embeddings
    pca_model = None
    
    if use_pca and embeddings.shape[1] > pca_dims:
        logger.info(f"Step 2.5/5: PCA reduction ({embeddings.shape[1]} -> {pca_dims} dims)...")
        step_start = time.time()
        
        pca_model = PCA(n_components=pca_dims, random_state=42)
        embeddings_for_umap = pca_model.fit_transform(embeddings)
        
        explained_var = pca_model.explained_variance_ratio_.sum()
        step_time = time.time() - step_start
        logger.info(f"PCA complete in {step_time:.1f}s (preserved {explained_var:.1%} variance)")
        logger.info(f"Reduced embeddings: {embeddings_for_umap.shape}")
    
    # Step 3: Run UMAP for 3D (OPTIMIZED)
    logger.info("Step 3/5: Running OPTIMIZED UMAP for 3D coordinates...")
    step_start = time.time()
    
    reducer_3d = UMAP(
        n_components=3,
        n_neighbors=15,        # ↓ from 30 for speed
        min_dist=0.1,          # ↓ from 0.3 for speed
        metric='euclidean',    # faster than cosine
        n_jobs=-1,             # all cores (no random_state!)
        low_memory=False,      # faster if RAM available
        spread=1.5,
        verbose=True
    )
    coords_3d = reducer_3d.fit_transform(embeddings_for_umap)
    
    step_time = time.time() - step_start
    logger.info(f"Generated 3D coordinates: {coords_3d.shape} in {step_time/60:.1f} minutes")
    
    # Step 4: Run UMAP for 2D (OPTIMIZED)
    logger.info("Step 4/5: Running OPTIMIZED UMAP for 2D coordinates...")
    step_start = time.time()
    
    reducer_2d = UMAP(
        n_components=2,
        n_neighbors=15,        # ↓ from 30 for speed
        min_dist=0.1,          # ↓ from 0.3 for speed
        metric='euclidean',    # faster than cosine
        n_jobs=-1,             # all cores (no random_state!)
        low_memory=False,      # faster if RAM available
        spread=1.5,
        verbose=True
    )
    coords_2d = reducer_2d.fit_transform(embeddings_for_umap)
    
    step_time = time.time() - step_start
    logger.info(f"Generated 2D coordinates: {coords_2d.shape} in {step_time/60:.1f} minutes")
    
    # Step 5: Save to Parquet files
    logger.info("Step 5/5: Saving to Parquet files...")
    step_start = time.time()
    
    # Prepare DataFrame with all data
    output_df = df.copy()
    output_df['x_3d'] = coords_3d[:, 0]
    output_df['y_3d'] = coords_3d[:, 1]
    output_df['z_3d'] = coords_3d[:, 2]
    output_df['x_2d'] = coords_2d[:, 0]
    output_df['y_2d'] = coords_2d[:, 1]
    
    # Save main data
    models_file = output_path / f"models_{version}.parquet"
    output_df.to_parquet(models_file, compression='snappy', index=False)
    logger.info(f"Saved models data: {models_file} ({models_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Save embeddings separately
    embeddings_file = output_path / f"embeddings_{version}.parquet"
    embeddings_df = pd.DataFrame({
        'model_id': df['modelId'].values,
        'embedding': [emb.tolist() for emb in embeddings]
    })
    embeddings_df.to_parquet(embeddings_file, compression='snappy', index=False)
    logger.info(f"Saved embeddings: {embeddings_file} ({embeddings_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Save metadata
    total_time = time.time() - start_time
    metadata = {
        'version': version,
        'created_at': datetime.now().isoformat(),
        'total_models': len(df),
        'embedding_dim': embeddings.shape[1],
        'umap_3d_shape': coords_3d.shape,
        'umap_2d_shape': coords_2d.shape,
        'unique_libraries': int(df['library_name'].nunique()),
        'unique_pipelines': int(df['pipeline_tag'].nunique()),
        'processing_time_seconds': total_time,
        'processing_time_minutes': total_time / 60,
        'optimizations': {
            'pca_enabled': use_pca,
            'pca_dims': pca_dims if use_pca else None,
            'pca_variance_preserved': float(pca_model.explained_variance_ratio_.sum()) if pca_model else None,
            'umap_parallel': True,
            'umap_n_neighbors': 15,
            'umap_metric': 'euclidean',
            'batch_size': 256
        },
        'statistics': {
            'downloads': {
                'min': float(df['downloads'].min()) if 'downloads' in df else 0,
                'max': float(df['downloads'].max()) if 'downloads' in df else 0,
                'mean': float(df['downloads'].mean()) if 'downloads' in df else 0,
            },
            'likes': {
                'min': float(df['likes'].min()) if 'likes' in df else 0,
                'max': float(df['likes'].max()) if 'likes' in df else 0,
                'mean': float(df['likes'].mean()) if 'likes' in df else 0,
            }
        }
    }
    
    metadata_file = output_path / f"metadata_{version}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata: {metadata_file}")
    
    step_time = time.time() - step_start
    logger.info(f"Files saved in {step_time:.1f} seconds")
    
    # Final summary
    logger.info("="*60)
    logger.info("FAST PRE-COMPUTATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"Total time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    logger.info(f"Models processed: {len(df):,}")
    logger.info(f"Speedup estimate: ~3-5x faster than standard version")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Files created:")
    logger.info(f"  - {models_file.name} ({models_file.stat().st_size / 1024 / 1024:.1f} MB)")
    logger.info(f"  - {embeddings_file.name} ({embeddings_file.stat().st_size / 1024 / 1024:.1f} MB)")
    logger.info(f"  - {metadata_file.name}")
    logger.info("="*60)
    
    return output_df, embeddings, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast pre-computation of HF model embeddings and coordinates")
    parser.add_argument("--sample-size", type=int, default=150000, help="Number of models to process")
    parser.add_argument("--output-dir", type=str, default="../precomputed_data", help="Output directory")
    parser.add_argument("--version", type=str, default="v1", help="Version tag")
    parser.add_argument("--pca-dims", type=int, default=50, help="PCA dimensions for pre-reduction")
    parser.add_argument("--no-pca", action="store_true", help="Disable PCA pre-reduction")
    
    args = parser.parse_args()
    
    try:
        precompute_fast(
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            version=args.version,
            pca_dims=args.pca_dims,
            use_pca=not args.no_pca
        )
    except Exception as e:
        logger.error(f"Pre-computation failed: {e}", exc_info=True)
        sys.exit(1)

