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
from utils.chunked_loader import create_chunk_index

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def precompute_embeddings_and_umap(
    sample_size=150000,
    output_dir="precomputed_data",
    version="v1",
    chunked=False,
    chunk_size=50000
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
    
    # Ensure df is reset and matches embeddings length
    df_aligned = df.reset_index(drop=True)
    n_models = len(embeddings)  # Use embeddings length as source of truth
    
    # Ensure all arrays match
    if len(df_aligned) != n_models:
        logger.warning(f"DataFrame length ({len(df_aligned)}) != embeddings length ({n_models}), truncating/aligning...")
        df_aligned = df_aligned.head(n_models).reset_index(drop=True)
    
    # Prepare DataFrame with all data
    result_df = pd.DataFrame({
        'model_id': df_aligned['model_id'].astype(str).values[:n_models],
        'library_name': df_aligned.get('library_name', pd.Series([None] * n_models)).values[:n_models],
        'pipeline_tag': df_aligned.get('pipeline_tag', pd.Series([None] * n_models)).values[:n_models],
        'downloads': df_aligned.get('downloads', pd.Series([0] * n_models)).values[:n_models],
        'likes': df_aligned.get('likes', pd.Series([0] * n_models)).values[:n_models],
        'trendingScore': df_aligned.get('trendingScore', pd.Series([None] * n_models)).values[:n_models],
        'tags': df_aligned.get('tags', pd.Series([None] * n_models)).values[:n_models],
        'parent_model': df_aligned.get('parent_model', pd.Series([None] * n_models)).values[:n_models],
        'licenses': df_aligned.get('licenses', pd.Series([None] * n_models)).values[:n_models],
        'createdAt': df_aligned.get('createdAt', pd.Series([None] * n_models)).values[:n_models],
        'x_3d': coords_3d[:n_models, 0],
        'y_3d': coords_3d[:n_models, 1],
        'z_3d': coords_3d[:n_models, 2],
        'x_2d': coords_2d[:n_models, 0],
        'y_2d': coords_2d[:n_models, 1],
    })
    
    # Save main data file
    data_file = output_path / f"models_{version}.parquet"
    result_df.to_parquet(data_file, compression='snappy', index=False)
    logger.info(f"Saved main data: {data_file} ({data_file.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Save embeddings separately (for similarity search)
    if chunked:
        # Save embeddings in chunks
        logger.info(f"Saving embeddings in chunks (chunk_size={chunk_size:,})...")
        # Create embeddings dataframe - ensure it matches embeddings array length
        embeddings_df = pd.DataFrame({
            'model_id': df_aligned['model_id'].astype(str).values[:n_models],
            'embedding': [emb.tolist() for emb in embeddings]
        })
        
        # Reset index to ensure proper alignment
        embeddings_df = embeddings_df.reset_index(drop=True)
        
        # Create chunk index using embeddings_df
        chunk_index = create_chunk_index(embeddings_df, chunk_size=chunk_size, output_dir=output_path, version=version)
        
        # Save chunks
        total_chunks = chunk_index['chunk_id'].nunique()
        for chunk_id in range(total_chunks):
            chunk_mask = chunk_index['chunk_id'] == chunk_id
            chunk_embeddings = embeddings_df[chunk_mask]
            
            chunk_file = output_path / f"embeddings_chunk_{chunk_id:03d}_{version}.parquet"
            chunk_embeddings.to_parquet(chunk_file, compression='snappy', index=False)
            logger.info(f"  Saved chunk {chunk_id}: {chunk_file.name} ({chunk_file.stat().st_size / 1024 / 1024:.2f} MB, {len(chunk_embeddings):,} models)")
        
        logger.info(f"Saved {total_chunks} embedding chunks")
        
        # Also save single file for backward compatibility (optional, can be skipped for very large datasets)
        if len(embeddings_df) <= 500000:  # Only if reasonable size
            embeddings_file = output_path / f"embeddings_{version}.parquet"
            embeddings_df.to_parquet(embeddings_file, compression='snappy', index=False)
            logger.info(f"Also saved single embeddings file: {embeddings_file.name} ({embeddings_file.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        # Save single embeddings file (original behavior)
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
        'total_models': n_models,
        'sample_size': sample_size,
        'embedding_dim': embeddings.shape[1],
        'unique_libraries': int(df_aligned['library_name'].nunique()) if 'library_name' in df_aligned.columns else 0,
        'unique_pipelines': int(df_aligned['pipeline_tag'].nunique()) if 'pipeline_tag' in df_aligned.columns else 0,
        'files': {
            'models': f"models_{version}.parquet",
            'embeddings': f"embeddings_{version}.parquet" if not chunked else f"embeddings_chunk_*_{version}.parquet",
            'chunk_index': f"chunk_index_{version}.parquet" if chunked else None
        },
        'chunked': chunked,
        'chunk_size': chunk_size if chunked else None,
        'stats': {
            'avg_downloads': float(df_aligned['downloads'].mean()) if 'downloads' in df_aligned.columns else 0,
            'avg_likes': float(df_aligned['likes'].mean()) if 'likes' in df_aligned.columns else 0,
            'libraries': df_aligned['library_name'].value_counts().head(20).to_dict() if 'library_name' in df_aligned.columns else {},
            'pipelines': df_aligned['pipeline_tag'].value_counts().head(20).to_dict() if 'pipeline_tag' in df_aligned.columns else {}
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
    logger.info(f"Models processed: {n_models:,}")
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
    parser.add_argument(
        '--chunked',
        action='store_true',
        help='Save embeddings in chunks for scalable loading (recommended for large datasets)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=50000,
        help='Number of models per chunk when using --chunked (default: 50000)'
    )
    
    args = parser.parse_args()
    
    sample_size = None if args.sample_size == 0 else args.sample_size
    
    try:
        precompute_embeddings_and_umap(
            sample_size=sample_size,
            output_dir=args.output_dir,
            version=args.version,
            chunked=args.chunked,
            chunk_size=args.chunk_size
        )
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during pre-computation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

