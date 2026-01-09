"""
Pre-compute force-directed layout positions for the full model network.
Uses graph-tool or networkx with Barnes-Hut optimization for large-scale layouts.

This script generates x, y, z coordinates for all nodes so the frontend
doesn't need to compute force simulation in real-time.

Usage:
    python precompute_force_layout.py [--output force_layout.pkl] [--3d]
"""

import os
import sys
import time
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_data() -> 'pd.DataFrame':
    """Load model data from precomputed parquet or CSV."""
    import pandas as pd
    
    backend_dir = Path(__file__).parent.parent
    root_dir = backend_dir.parent
    
    # Try precomputed data first
    precomputed_dir = root_dir / "precomputed_data"
    if precomputed_dir.exists():
        parquet_files = list(precomputed_dir.glob("*.parquet"))
        if parquet_files:
            logger.info(f"Loading from precomputed parquet: {parquet_files[0]}")
            return pd.read_parquet(parquet_files[0])
    
    # Try CSV data
    csv_path = precomputed_dir / "models.csv"
    if csv_path.exists():
        logger.info(f"Loading from CSV: {csv_path}")
        return pd.read_csv(csv_path)
    
    # Try data directory
    data_dir = root_dir / "data"
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        for csv_file in csv_files:
            if "model" in csv_file.name.lower():
                logger.info(f"Loading from {csv_file}")
                return pd.read_csv(csv_file)
    
    raise FileNotFoundError("No model data found")


def load_existing_graph(graph_path: str = None) -> Optional['nx.DiGraph']:
    """Load pre-existing networkx graph from pickle file."""
    import networkx as nx
    
    if graph_path and Path(graph_path).exists():
        logger.info(f"Loading existing graph from {graph_path}")
        with open(graph_path, 'rb') as f:
            return pickle.load(f)
    
    # Search for graph file
    search_paths = [
        Path(__file__).parent.parent.parent / "ai-ecosystem" / "data" / "ai_ecosystem_graph.pkl",
        Path(__file__).parent.parent.parent.parent / "ai-ecosystem" / "data" / "ai_ecosystem_graph.pkl",
        Path.home() / "ai-ecosystem-v2" / "ai-ecosystem" / "data" / "ai_ecosystem_graph.pkl",
    ]
    
    for path in search_paths:
        if path.exists():
            logger.info(f"Found existing graph at {path}")
            with open(path, 'rb') as f:
                return pickle.load(f)
    
    return None


def build_network_graph(df: 'pd.DataFrame') -> 'nx.DiGraph':
    """Build network graph from model dataframe."""
    import networkx as nx
    
    logger.info(f"Building network from {len(df):,} models...")
    G = nx.DiGraph()
    
    # Add all models as nodes
    for _, row in df.iterrows():
        model_id = str(row.get('model_id', row.get('modelId', '')))
        if not model_id:
            continue
            
        G.add_node(model_id, 
            downloads=row.get('downloads', 0),
            likes=row.get('likes', 0),
            library=row.get('library_name', row.get('library', '')),
            pipeline=row.get('pipeline_tag', '')
        )
    
    # Add edges based on parent relationships
    edge_count = 0
    for _, row in df.iterrows():
        model_id = str(row.get('model_id', row.get('modelId', '')))
        parent_id = row.get('parent_model', row.get('base_model', None))
        
        if not model_id:
            continue
            
        if pd.notna(parent_id) and str(parent_id).strip() and str(parent_id) != 'nan':
            parent_id = str(parent_id).strip()
            if parent_id in G.nodes:
                G.add_edge(parent_id, model_id, edge_type='derivative')
                edge_count += 1
    
    logger.info(f"Network: {G.number_of_nodes():,} nodes, {edge_count:,} edges")
    return G


def compute_force_layout_3d(
    G: 'nx.Graph',
    iterations: int = 100,
    seed: int = 42
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute 3D force-directed layout using networkx spring_layout.
    For very large graphs, uses Barnes-Hut approximation.
    """
    import networkx as nx
    
    n_nodes = G.number_of_nodes()
    logger.info(f"Computing 3D layout for {n_nodes:,} nodes...")
    
    if n_nodes == 0:
        return {}
    
    start_time = time.time()
    
    # For large graphs, compute layout on largest connected component first
    if n_nodes > 100000:
        logger.info("Large graph detected - using optimized approach...")
        
        # Get largest connected component (treat as undirected)
        if isinstance(G, nx.DiGraph):
            G_undirected = G.to_undirected()
        else:
            G_undirected = G
            
        components = list(nx.connected_components(G_undirected))
        components.sort(key=len, reverse=True)
        
        logger.info(f"Found {len(components):,} connected components")
        
        # Compute layouts for each component
        positions = {}
        offset_x = 0
        
        for i, component in enumerate(components):
            if len(component) < 2:
                # Isolated nodes - place randomly
                for node in component:
                    positions[node] = (
                        offset_x + np.random.randn() * 10,
                        np.random.randn() * 100,
                        np.random.randn() * 100
                    )
                continue
            
            subgraph = G_undirected.subgraph(component)
            
            # Use spring layout with reduced iterations for large components
            iter_count = min(iterations, max(20, 100 - len(component) // 10000))
            
            logger.info(f"  Component {i+1}/{len(components)}: {len(component):,} nodes, {iter_count} iterations")
            
            try:
                # 3D layout using spring_layout
                pos_2d = nx.spring_layout(
                    subgraph,
                    dim=3,
                    k=1.0 / np.sqrt(len(component)),
                    iterations=iter_count,
                    seed=seed + i,
                    scale=100 * np.log10(max(len(component), 10))
                )
                
                # Apply offset to separate components
                for node, (x, y, z) in pos_2d.items():
                    positions[node] = (x + offset_x, y, z)
                
                # Move offset for next component
                offset_x += 300 * np.log10(max(len(component), 10))
                
            except Exception as e:
                logger.warning(f"Layout failed for component {i}: {e}")
                # Fallback: random positions
                for node in component:
                    positions[node] = (
                        offset_x + np.random.randn() * 50,
                        np.random.randn() * 50,
                        np.random.randn() * 50
                    )
    else:
        # Standard approach for smaller graphs
        try:
            positions_raw = nx.spring_layout(
                G.to_undirected() if isinstance(G, nx.DiGraph) else G,
                dim=3,
                k=2.0 / np.sqrt(n_nodes) if n_nodes > 0 else 1.0,
                iterations=iterations,
                seed=seed,
                scale=200
            )
            positions = {node: tuple(pos) for node, pos in positions_raw.items()}
        except Exception as e:
            logger.warning(f"Spring layout failed: {e}, using random positions")
            np.random.seed(seed)
            positions = {
                node: (np.random.randn() * 100, np.random.randn() * 100, np.random.randn() * 100)
                for node in G.nodes()
            }
    
    elapsed = time.time() - start_time
    logger.info(f"Layout computed in {elapsed:.1f}s")
    
    return positions


def compute_force_layout_fa2(
    G: 'nx.Graph',
    iterations: int = 100,
    seed: int = 42
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute layout using ForceAtlas2 algorithm (faster for large graphs).
    Falls back to spring_layout if fa2 not available.
    """
    try:
        from fa2 import ForceAtlas2
        
        n_nodes = G.number_of_nodes()
        logger.info(f"Computing FA2 layout for {n_nodes:,} nodes...")
        
        if n_nodes == 0:
            return {}
        
        # Convert to undirected for layout
        if isinstance(G, nx.DiGraph):
            import networkx as nx
            G_layout = G.to_undirected()
        else:
            G_layout = G
        
        # Initialize ForceAtlas2
        fa2 = ForceAtlas2(
            outboundAttractionDistribution=True,
            linLogMode=False,
            adjustSizes=False,
            edgeWeightInfluence=1.0,
            jitterTolerance=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=False,
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,
            verbose=False
        )
        
        # Compute 2D positions
        positions_2d = fa2.forceatlas2_networkx_layout(
            G_layout,
            iterations=iterations
        )
        
        # Add 3rd dimension based on hierarchy/properties
        np.random.seed(seed)
        positions = {}
        for node, (x, y) in positions_2d.items():
            # Z based on downloads (popular models higher)
            downloads = G.nodes[node].get('downloads', 0) if node in G.nodes else 0
            z = np.log10(max(downloads, 1)) * 10 + np.random.randn() * 5
            positions[node] = (x * 100, y * 100, z)
        
        return positions
        
    except ImportError:
        logger.warning("fa2 not installed, falling back to spring_layout")
        return compute_force_layout_3d(G, iterations, seed)


def save_layout(
    positions: Dict[str, Tuple[float, float, float]],
    output_path: str,
    graph: 'nx.Graph' = None
):
    """Save layout positions to pickle file."""
    
    data = {
        'positions': positions,
        'n_nodes': len(positions),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    if graph is not None:
        data['n_edges'] = graph.number_of_edges()
    
    # Calculate bounds
    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        zs = [p[2] for p in positions.values()]
        data['bounds'] = {
            'x_min': min(xs), 'x_max': max(xs),
            'y_min': min(ys), 'y_max': max(ys),
            'z_min': min(zs), 'z_max': max(zs),
        }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Saved layout to {output_path}")
    logger.info(f"  Nodes: {len(positions):,}")
    if 'bounds' in data:
        b = data['bounds']
        logger.info(f"  Bounds: X[{b['x_min']:.1f}, {b['x_max']:.1f}], Y[{b['y_min']:.1f}, {b['y_max']:.1f}], Z[{b['z_min']:.1f}, {b['z_max']:.1f}]")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute force-directed layout')
    parser.add_argument('--output', '-o', type=str, default='force_layout_3d.pkl',
                       help='Output pickle file path')
    parser.add_argument('--iterations', '-i', type=int, default=100,
                       help='Number of layout iterations')
    parser.add_argument('--algorithm', '-a', choices=['spring', 'fa2'], default='spring',
                       help='Layout algorithm to use')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--graph', '-g', type=str, default=None,
                       help='Path to existing networkx graph pickle file')
    
    args = parser.parse_args()
    
    # Determine output path
    backend_dir = Path(__file__).parent.parent
    root_dir = backend_dir.parent
    precomputed_dir = root_dir / "precomputed_data"
    precomputed_dir.mkdir(exist_ok=True)
    
    output_path = precomputed_dir / args.output
    
    logger.info("=" * 60)
    logger.info("Pre-computing Force-Directed Layout")
    logger.info("=" * 60)
    
    # Try to load existing graph first (faster)
    G = load_existing_graph(args.graph)
    
    if G is None:
        # Load data and build graph
        df = load_model_data()
        logger.info(f"Loaded {len(df):,} models")
        G = build_network_graph(df)
    else:
        logger.info(f"Using existing graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Compute layout
    if args.algorithm == 'fa2':
        positions = compute_force_layout_fa2(G, args.iterations, args.seed)
    else:
        positions = compute_force_layout_3d(G, args.iterations, args.seed)
    
    # Save
    save_layout(positions, str(output_path), G)
    
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

