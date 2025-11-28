#!/usr/bin/env python3
"""
Pre-compute embeddings and UMAP coordinates for HF models.
This script generates pre-computed data files that can be loaded instantly on server startup.

Usage:
    python scripts/precompute_data.py --sample-size 150000 --output-dir ../precomputed_data
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


def precompute_embeddings_and_umap(
    sample_size=150000,
    output_dir="precomputed_data",
    version="v1"
):
    """
    Pre-compute embeddings and UMAP coordinates.
    
    Args:
        sample_size: Number of models to process (None for all)
        output_dir: Directory to save pre-computed files
        version: Version tag for the data
    """
    start_time = time.time()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting pre-computation for {sample_size if sample_size else 'ALL'} models...")
    logger.info(f"Output directory: {output_path.absolute()}")
    
    # Step 1: Load data with methodological sampling
    logger.info("Step 1/5: Loading model data (prioritizing base models)...")
    data_loader = ModelDataLoader()
    df = data_loader.load_data(sample_size=sample_size, prioritize_base_models=True)
    df = data_loader.preprocess_for_embedding(df)
    
    if 'model_id' in df.columns:
        df.set_index('model_id', drop=False, inplace=True)
    
    # Ensure numeric columns
    for col in ['downloads', 'likes']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    logger.info(f"Loaded {len(df)} models")
    
    # Step 2: Generate embeddings
    logger.info("Step 2/5: Generating embeddings (this may take 10-30 minutes)...")
    embedder = ModelEmbedder()
    texts = df['combined_text'].tolist()
    embeddings = embedder.generate_embeddings(texts, batch_size=128)
    logger.info(f"Generated embeddings: {embeddings.shape}")
    
    # Step 3: Run UMAP for 3D
    logger.info("Step 3/5: Running UMAP for 3D coordinates (this may take 5-15 minutes)...")
    reducer_3d = UMAP(
        n_components=3,
        n_neighbors=30,
        min_dist=0.3,
        metric='cosine',
        random_state=42,
        n_jobs=-1,
        low_memory=True,
        spread=1.5,
        verbose=True
    )
    coords_3d = reducer_3d.fit_transform(embeddings)
    logger.info(f"Generated 3D coordinates: {coords_3d.shape}")
    
    # Step 4: Run UMAP for 2D
    logger.info("Step 4/5: Running UMAP for 2D coordinates (this may take 5-15 minutes)...")
    reducer_2d = UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.3,
        metric='cosine',
        random_state=42,
        n_jobs=-1,
        low_memory=True,
        spread=1.5,
        verbose=True
    )
    coords_2d = reducer_2d.fit_transform(embeddings)
    logger.info(f"Generated 2D coordinates: {coords_2d.shape}")
    
    # Step 5: Save to Parquet files
    logger.info("Step 5/5: Saving to Parquet files...")
    
    # Prepare DataFrame with all data
    result_df = pd.DataFrame({
        'model_id': df['model_id'].astype(str),
        'library_name': df.get('library_name', pd.Series([None] * len(df))),
        'pipeline_tag': df.get('pipeline_tag', pd.Series([None] * len(df))),
        'downloads': df.get('downloads', pd.Series([0] * len(df))),
        'likes': df.get('likes', pd.Series([0] * len(df))),
        'trendingScore': df.get('trendingScore', pd.Series([None] * len(df))),
        'tags': df.get('tags', pd.Series([None] * len(df))),
        'parent_model': df.get('parent_model', pd.Series([None] * len(df))),
        'licenses': df.get('licenses', pd.Series([None] * len(df))),
        'createdAt': df.get('createdAt', pd.Series([None] * len(df))),
        'x_3d': coords_3d[:, 0],
        'y_3d': coords_3d[:, 1],
        'z_3d': coords_3d[:, 2],
        'x_2d': coords_2d[:, 0],
        'y_2d': coords_2d[:, 1],
    })
    
    # Save main data file
    data_file = output_path / f"models_{version}.parquet"
    result_df.to_parquet(data_file, compression='snappy', index=False)
    logger.info(f"Saved main data: {data_file} ({data_file.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Save embeddings separately (for similarity search)
    embeddings_file = output_path / f"embeddings_{version}.parquet"
    embeddings_df = pd.DataFrame({
        'model_id': df['model_id'].astype(str),
        'embedding': [emb.tolist() for emb in embeddings]
    })
    embeddings_df.to_parquet(embeddings_file, compression='snappy', index=False)
    logger.info(f"Saved embeddings: {embeddings_file} ({embeddings_file.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Save metadata
    metadata = {
        'version': version,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'total_models': len(df),
        'sample_size': sample_size,
        'embedding_dim': embeddings.shape[1],
        'unique_libraries': int(df['library_name'].nunique()) if 'library_name' in df.columns else 0,
        'unique_pipelines': int(df['pipeline_tag'].nunique()) if 'pipeline_tag' in df.columns else 0,
        'files': {
            'models': f"models_{version}.parquet",
            'embeddings': f"embeddings_{version}.parquet"
        },
        'stats': {
            'avg_downloads': float(df['downloads'].mean()) if 'downloads' in df.columns else 0,
            'avg_likes': float(df['likes'].mean()) if 'likes' in df.columns else 0,
            'libraries': df['library_name'].value_counts().head(20).to_dict() if 'library_name' in df.columns else {},
            'pipelines': df['pipeline_tag'].value_counts().head(20).to_dict() if 'pipeline_tag' in df.columns else {}
        },
        'coordinates': {
            '3d': {
                'min': [float(coords_3d[:, i].min()) for i in range(3)],
                'max': [float(coords_3d[:, i].max()) for i in range(3)],
                'mean': [float(coords_3d[:, i].mean()) for i in range(3)]
            },
            '2d': {
                'min': [float(coords_2d[:, i].min()) for i in range(2)],
                'max': [float(coords_2d[:, i].max()) for i in range(2)],
                'mean': [float(coords_2d[:, i].mean()) for i in range(2)]
            }
        }
    }
    
    metadata_file = output_path / f"metadata_{version}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved metadata: {metadata_file}")
    
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Pre-computation complete!")
    logger.info(f"Total time: {elapsed / 60:.1f} minutes")
    logger.info(f"Models processed: {len(df):,}")
    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info(f"Files created:")
    logger.info(f"  - {data_file.name} ({data_file.stat().st_size / 1024 / 1024:.2f} MB)")
    logger.info(f"  - {embeddings_file.name} ({embeddings_file.stat().st_size / 1024 / 1024:.2f} MB)")
    logger.info(f"  - {metadata_file.name}")
    logger.info(f"{'='*60}\n")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Pre-compute embeddings and UMAP coordinates')
    parser.add_argument(
        '--sample-size',
        type=int,
        default=150000,
        help='Number of models to process (default: 150000, use 0 for all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='precomputed_data',
        help='Output directory for pre-computed files (default: precomputed_data)'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1',
        help='Version tag for the data (default: v1)'
    )
    
    args = parser.parse_args()
    
    sample_size = None if args.sample_size == 0 else args.sample_size
    
    try:
        precompute_embeddings_and_umap(
            sample_size=sample_size,
            output_dir=args.output_dir,
            version=args.version
        )
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during pre-computation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

