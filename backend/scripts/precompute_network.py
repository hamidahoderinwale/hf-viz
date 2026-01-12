"""
Pre-compute the full derivative network graph and save it to disk.
This allows the API to load the network instantly instead of building it on-demand.

Usage:
    python scripts/precompute_network.py [--output-dir precomputed_data] [--version v1]
"""

import os
import sys
import pickle
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from utils.network_analysis import ModelNetworkBuilder
from utils.precomputed_loader import PrecomputedDataLoader
from utils.data_loader import ModelDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def precompute_network(
    output_dir: str = "precomputed_data",
    version: str = "v1",
    include_edge_attributes: bool = False,
    min_downloads: int = 0,
    max_nodes: Optional[int] = None,
    load_from_hf: bool = False,
    sample_size: Optional[int] = None
):
    """
    Pre-compute the full derivative network graph for the force-directed visualization.
    
    Args:
        output_dir: Directory to save the network file
        version: Version tag for the data
        include_edge_attributes: Whether to calculate edge attributes
        min_downloads: Minimum downloads to include a model
        max_nodes: Maximum number of nodes (top N by downloads)
        load_from_hf: If True, load directly from HF dataset (includes parent relationships)
        sample_size: If load_from_hf=True, sample this many models (None = all models)
    """
    start_time = time.time()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PRE-COMPUTING FULL DERIVATIVE NETWORK")
    logger.info("=" * 60)
    
    # Step 1: Load model data
    logger.info("Step 1/3: Loading model data...")
    
    if load_from_hf:
        # Load directly from HF dataset (includes parent relationships)
        logger.info(f"Loading directly from Hugging Face dataset (sample_size={sample_size if sample_size else 'ALL'})...")
        data_loader = ModelDataLoader()
        df = data_loader.load_data(sample_size=sample_size, prioritize_base_models=False)
        df = data_loader.preprocess_for_embedding(df)
        
        # Ensure model_id is set as index
        if 'model_id' in df.columns:
            df.set_index('model_id', drop=False, inplace=True)
        
        # Ensure numeric columns
        for col in ['downloads', 'likes']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        logger.info(f"Loaded {len(df):,} models from HF dataset")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Check if parent_model column exists (needed for network edges)
        if 'parent_model' not in df.columns:
            logger.warning("'parent_model' column not found - network will have 0 edges!")
        else:
            parent_count = df['parent_model'].notna().sum()
            logger.info(f"Models with parent relationships: {parent_count:,}")
    else:
        # Load from pre-computed files
        loader = PrecomputedDataLoader(data_dir=output_dir, version=version)
        
        if not loader.check_available():
            logger.error(f"Pre-computed data not found in {output_dir}")
            logger.info("Please run precompute_data.py first, download from HF Hub, or use --load-from-hf flag")
            return False
        
        try:
            df, embeddings, metadata = loader.load_all()
            logger.info(f"Loaded {len(df):,} models from pre-computed data")
            
            # Check if parent_model column exists
            if 'parent_model' not in df.columns:
                logger.warning("'parent_model' column not found in pre-computed data - network will have 0 edges!")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    # Step 2: Filter data if needed
    if min_downloads > 0:
        df = df[df.get('downloads', 0) >= min_downloads]
        logger.info(f"Filtered to {len(df):,} models with >= {min_downloads} downloads")
    
    if max_nodes and len(df) > max_nodes:
        df = df.nlargest(max_nodes, 'downloads', keep='first')
        logger.info(f"Limited to top {max_nodes:,} models by downloads")
    
    # Step 3: Build network graph
    logger.info("Step 2/3: Building network graph (this may take 10-30 minutes)...")
    try:
        network_builder = ModelNetworkBuilder(df)
        graph = network_builder.build_full_derivative_network(
            include_edge_attributes=include_edge_attributes,
            filter_edge_types=None  # Include all edge types
        )
        
        logger.info(f"Graph built: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    except Exception as e:
        logger.error(f"Failed to build network graph: {e}", exc_info=True)
        return False
    
    # Step 4: Save network graph
    logger.info("Step 3/3: Saving network graph to disk...")
    network_file = output_path / "full_derivative_network.pkl"
    
    try:
        with open(network_file, 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size_mb = network_file.stat().st_size / (1024 * 1024)
        logger.info(f"Saved network graph to {network_file}")
        logger.info(f"File size: {file_size_mb:.2f} MB")
    except Exception as e:
        logger.error(f"Failed to save network graph: {e}", exc_info=True)
        return False
    
    # Save metadata
    metadata_file = output_path / "network_metadata.json"
    import json
    from datetime import datetime
    
    network_metadata = {
        "created_at": datetime.now().isoformat(),
        "version": version,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "include_edge_attributes": include_edge_attributes,
        "min_downloads": min_downloads,
        "max_nodes": max_nodes,
        "file_size_mb": round(file_size_mb, 2)
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(network_metadata, f, indent=2)
    
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"PRE-COMPUTATION COMPLETE in {total_time:.2f} seconds")
    logger.info(f"Network graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    logger.info(f"Saved to: {network_file}")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    import time
    
    parser = argparse.ArgumentParser(description="Pre-compute full derivative network graph")
    parser.add_argument("--output-dir", type=str, default="precomputed_data",
                       help="Output directory for pre-computed files")
    parser.add_argument("--version", type=str, default="v1",
                       help="Version tag for the data")
    parser.add_argument("--include-edge-attributes", action="store_true",
                       help="Include edge attributes (slower but more detailed)")
    parser.add_argument("--min-downloads", type=int, default=0,
                       help="Minimum downloads to include a model")
    parser.add_argument("--max-nodes", type=int, default=None,
                       help="Maximum number of nodes (top N by downloads)")
    parser.add_argument("--load-from-hf", action="store_true",
                       help="Load directly from HF dataset instead of pre-computed files (includes parent relationships)")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="If --load-from-hf, sample this many models (default: all models, use 0 for all)")
    
    args = parser.parse_args()
    
    success = precompute_network(
        output_dir=args.output_dir,
        version=args.version,
        include_edge_attributes=args.include_edge_attributes,
        min_downloads=args.min_downloads,
        max_nodes=args.max_nodes,
        load_from_hf=args.load_from_hf,
        sample_size=None if args.sample_size == 0 else args.sample_size
    )
    
    sys.exit(0 if success else 1)

