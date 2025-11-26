"""
Export minimal dataset to binary format for fast client-side loading.
This creates a compact binary representation optimized for WebGL rendering.
"""
import struct
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import ModelDataLoader
from utils.dimensionality_reduction import DimensionReducer
from utils.embeddings import ModelEmbedder


def calculate_family_depths(df: pd.DataFrame) -> dict:
    """Calculate depth of each model in its family tree."""
    depths = {}
    
    def get_depth(model_id: str, visited: set = None) -> int:
        if visited is None:
            visited = set()
        if model_id in visited:
            return 0  # Cycle detected
        visited.add(model_id)
        
        if model_id in depths:
            return depths[model_id]
        
        parent_col = df.get('parent_model', pd.Series([None] * len(df), index=df.index))
        model_row = df[df['model_id'] == model_id]
        
        if model_row.empty:
            depths[model_id] = 0
            return 0
        
        parent = model_row.iloc[0].get('parent_model')
        if pd.isna(parent) or parent == '' or str(parent) == 'nan':
            depths[model_id] = 0
            return 0
        
        parent_depth = get_depth(str(parent), visited.copy())
        depth = parent_depth + 1
        depths[model_id] = depth
        return depth
    
    for model_id in df['model_id'].unique():
        if model_id not in depths:
            get_depth(str(model_id))
    
    return depths


def export_binary_dataset(df: pd.DataFrame, reduced_embeddings: np.ndarray, output_dir: Path):
    """
    Export minimal dataset to binary format for fast client-side loading.
    
    Binary format:
    - Header (64 bytes): magic, version, counts, lookup table sizes
    - Domain lookup table (32 bytes per domain)
    - License lookup table (32 bytes per license)
    - Family lookup table (32 bytes per family)
    - Model records (16 bytes each): x, y, z, domain_id, license_id, family_id, flags
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting {len(df)} models to binary format...")
    
    # Ensure we have coordinates
    if 'x' not in df.columns or 'y' not in df.columns:
        if reduced_embeddings is None or len(reduced_embeddings) != len(df):
            raise ValueError("Need reduced embeddings to generate coordinates")
        
        df['x'] = reduced_embeddings[:, 0] if reduced_embeddings.shape[1] > 0 else 0.0
        df['y'] = reduced_embeddings[:, 1] if reduced_embeddings.shape[1] > 1 else 0.0
        df['z'] = reduced_embeddings[:, 2] if reduced_embeddings.shape[1] > 2 else 0.0
    
    # Create lookup tables
    # Domain = library_name
    domains = sorted(df['library_name'].dropna().astype(str).unique())
    domains = [d for d in domains if d and d != 'nan'][:255]  # Limit to 255
    
    # License
    licenses = sorted(df['license'].dropna().astype(str).unique())
    licenses = [l for l in licenses if l and l != 'nan'][:255]  # Limit to 255
    
    # Family ID mapping (use parent_model to create family groups)
    family_depths = calculate_family_depths(df)
    
    # Create family mapping: group models by root parent
    def get_root_parent(model_id: str) -> str:
        visited = set()
        current = str(model_id)
        while current in visited == False:
            visited.add(current)
            model_row = df[df['model_id'] == current]
            if model_row.empty:
                return current
            parent = model_row.iloc[0].get('parent_model')
            if pd.isna(parent) or parent == '' or str(parent) == 'nan':
                return current
            current = str(parent)
        return current
    
    root_parents = {}
    family_counter = 0
    for model_id in df['model_id'].unique():
        root = get_root_parent(str(model_id))
        if root not in root_parents:
            root_parents[root] = family_counter
            family_counter += 1
    
    # Map each model to its family
    model_to_family = {}
    for model_id in df['model_id'].unique():
        root = get_root_parent(str(model_id))
        model_to_family[str(model_id)] = root_parents.get(root, 65535)
    
    # Limit families to 65535 (u16 max)
    if len(root_parents) > 65535:
        # Use hash-based family IDs
        import hashlib
        for model_id in df['model_id'].unique():
            root = get_root_parent(str(model_id))
            family_hash = int(hashlib.md5(root.encode()).hexdigest()[:4], 16) % 65535
            model_to_family[str(model_id)] = family_hash
    
    # Prepare model records
    records = []
    model_ids = []
    
    for idx, row in df.iterrows():
        model_id = str(row['model_id'])
        model_ids.append(model_id)
        
        # Get coordinates
        x = float(row.get('x', 0.0))
        y = float(row.get('y', 0.0))
        z = float(row.get('z', 0.0))
        
        # Encode domain (library_name)
        domain_str = str(row.get('library_name', ''))
        domain_id = domains.index(domain_str) if domain_str in domains else 255
        
        # Encode license
        license_str = str(row.get('license', ''))
        license_id = licenses.index(license_str) if license_str in licenses else 255
        
        # Encode family
        family_id = model_to_family.get(model_id, 65535)
        
        # Encode flags
        flags = 0
        parent = row.get('parent_model')
        if pd.isna(parent) or parent == '' or str(parent) == 'nan':
            flags |= 0x01  # is_base_model
        
        # Check if has children (simple check - could be improved)
        children = df[df['parent_model'] == model_id]
        if len(children) > 0:
            flags |= 0x04  # has_children
        elif not pd.isna(parent) and parent != '' and str(parent) != 'nan':
            flags |= 0x02  # has_parent
        
        # Pack record: f32 x, f32 y, f32 z, u8 domain, u8 license, u16 family, u8 flags
        records.append(struct.pack('fffBBBH', x, y, z, domain_id, license_id, family_id, flags))
    
    num_models = len(records)
    
    # Write binary file
    with open(output_dir / 'embeddings.bin', 'wb') as f:
        # Header (64 bytes)
        header = struct.pack('5sBIIIBBH50s',
            b'HFVIZ',  # magic (5 bytes)
            1,  # version (1 byte)
            num_models,  # num_models (4 bytes)
            len(domains),  # num_domains (4 bytes)
            len(licenses),  # num_licenses (4 bytes)
            len(set(model_to_family.values())),  # num_families (4 bytes)
            0,  # reserved (1 byte)
            0,  # reserved (1 byte)
            0,  # reserved (2 bytes)
            b'\x00' * 50  # padding (50 bytes)
        )
        f.write(header)
        
        # Domain lookup table (32 bytes per domain, null-terminated)
        for domain in domains:
            domain_bytes = domain.encode('utf-8')[:31]
            f.write(domain_bytes.ljust(32, b'\x00'))
        
        # License lookup table (32 bytes per license)
        for license in licenses:
            license_bytes = license.encode('utf-8')[:31]
            f.write(license_bytes.ljust(32, b'\x00'))
        
        # Model records
        f.write(b''.join(records))
    
    # Write model IDs JSON (separate file for string table)
    with open(output_dir / 'model_ids.json', 'w') as f:
        json.dump(model_ids, f)
    
    # Write metadata JSON
    metadata = {
        'domains': domains,
        'licenses': licenses,
        'num_models': num_models,
        'num_families': len(set(model_to_family.values())),
        'version': 1
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    binary_size = (output_dir / 'embeddings.bin').stat().st_size
    json_size = (output_dir / 'model_ids.json').stat().st_size
    
    print(f"✓ Exported {num_models} models")
    print(f"✓ Binary size: {binary_size / 1024 / 1024:.2f} MB")
    print(f"✓ Model IDs JSON: {json_size / 1024 / 1024:.2f} MB")
    print(f"✓ Total: {(binary_size + json_size) / 1024 / 1024:.2f} MB")
    print(f"✓ Domains: {len(domains)}")
    print(f"✓ Licenses: {len(licenses)}")
    print(f"✓ Families: {len(set(model_to_family.values()))}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Export dataset to binary format')
    parser.add_argument('--output', type=str, default='backend/cache/binary', help='Output directory')
    parser.add_argument('--sample-size', type=int, default=None, help='Sample size (for testing)')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    # Load data
    print("Loading dataset...")
    data_loader = ModelDataLoader()
    df = data_loader.load_data(sample_size=args.sample_size)
    df = data_loader.preprocess_for_embedding(df)
    
    # Generate embeddings and reduce dimensions if needed
    if 'x' not in df.columns or 'y' not in df.columns:
        print("Generating embeddings...")
        embedder = ModelEmbedder()
        embeddings = embedder.generate_embeddings(df['combined_text'].tolist())
        
        print("Reducing dimensions...")
        reducer = DimensionReducer()
        reduced_embeddings = reducer.reduce_dimensions(embeddings, n_components=3, method='umap')
    else:
        reduced_embeddings = None
    
    # Export
    export_binary_dataset(df, reduced_embeddings, output_dir)
    print("Done!")

