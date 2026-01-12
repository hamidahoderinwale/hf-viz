#!/usr/bin/env python3
"""
Upload precomputed chunked data to Hugging Face Dataset.
Run this after generating chunked embeddings locally.
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, login
from tqdm import tqdm

def upload_chunked_data(
    dataset_id: str = "modelbiome/hf-viz-precomputed",
    data_dir: str = "precomputed_data",
    version: str = "v1",
    token: str = None
):
    """
    Upload chunked embeddings and metadata to HF Dataset.
    
    Args:
        dataset_id: Hugging Face dataset ID
        data_dir: Local directory containing precomputed data
        version: Version tag
        token: HF token (or use login())
    """
    if token:
        login(token=token)
    else:
        login()  # Will prompt for token or use cached
    
    api = HfApi()
    data_path = Path(data_dir)
    
    # Required files
    required_files = [
        f"metadata_{version}.json",
        f"models_{version}.parquet",
        f"chunk_index_{version}.parquet",
    ]
    
    # Chunk files
    chunk_files = []
    chunk_id = 0
    while True:
        chunk_file = data_path / f"embeddings_chunk_{chunk_id:03d}_{version}.parquet"
        if chunk_file.exists():
            chunk_files.append(f"embeddings_chunk_{chunk_id:03d}_{version}.parquet")
            chunk_id += 1
        else:
            break
    
    print(f"Found {len(chunk_files)} chunk files")
    
    # Upload required files
    print("\nUploading required files...")
    for filename in tqdm(required_files, desc="Required files"):
        filepath = data_path / filename
        if filepath.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(filepath),
                    path_in_repo=filename,
                    repo_id=dataset_id,
                    repo_type="dataset",
                    commit_message=f"Upload {filename}"
                )
                print(f"✓ Uploaded {filename}")
            except Exception as e:
                print(f"✗ Failed to upload {filename}: {e}")
        else:
            print(f"⚠ {filename} not found, skipping")
    
    # Upload chunk files
    print(f"\nUploading {len(chunk_files)} chunk files...")
    for filename in tqdm(chunk_files, desc="Chunk files"):
        filepath = data_path / filename
        try:
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=dataset_id,
                repo_type="dataset",
                commit_message=f"Upload {filename}"
            )
        except Exception as e:
            print(f"✗ Failed to upload {filename}: {e}")
            break
    
    print(f"\n✓ Upload complete!")
    print(f"  Dataset: {dataset_id}")
    print(f"  Files uploaded: {len(required_files) + len(chunk_files)}")
    print(f"  Chunks: {len(chunk_files)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload chunked data to HF Dataset")
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="modelbiome/hf-viz-precomputed",
        help="Hugging Face dataset ID"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="precomputed_data",
        help="Local directory with precomputed data"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version tag"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or use login())"
    )
    
    args = parser.parse_args()
    
    upload_chunked_data(
        dataset_id=args.dataset_id,
        data_dir=args.data_dir,
        version=args.version,
        token=args.token
    )


