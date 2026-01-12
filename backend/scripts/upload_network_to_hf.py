"""
Upload pre-computed network graph to Hugging Face Hub dataset.

Usage:
    python scripts/upload_network_to_hf.py [--network-file precomputed_data/full_derivative_network.pkl] [--version v1]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.precomputed_loader import HF_PRECOMPUTED_DATASET

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def upload_network_to_hf(
    network_file: str,
    version: str = "v1",
    dataset_id: str = None
):
    """
    Upload pre-computed network graph to Hugging Face Hub.
    
    Args:
        network_file: Path to the network pickle file
        version: Version tag for the data
        dataset_id: HF dataset ID (defaults to HF_PRECOMPUTED_DATASET)
    """
    try:
        from huggingface_hub import HfApi, upload_file
        
        if dataset_id is None:
            dataset_id = HF_PRECOMPUTED_DATASET
        
        network_path = Path(network_file)
        if not network_path.exists():
            logger.error(f"Network file not found: {network_file}")
            return False
        
        logger.info(f"Uploading network graph to {dataset_id}...")
        logger.info(f"File: {network_file}")
        logger.info(f"Version: {version}")
        
        api = HfApi()
        
        # Check if repository exists, create if it doesn't
        try:
            api.dataset_info(dataset_id)
            logger.info(f"Repository {dataset_id} exists")
        except Exception:
            logger.info(f"Repository {dataset_id} not found. Creating it...")
            try:
                api.create_repo(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    exist_ok=True
                )
                logger.info(f"Created repository {dataset_id}")
            except Exception as create_error:
                logger.error(f"Could not create repository: {create_error}")
                logger.info("You may need to create it manually at https://huggingface.co/new-dataset")
                return False
        
        # Upload network file
        filename = f"full_derivative_network_{version}.pkl"
        upload_file(
            path_or_fileobj=str(network_path),
            path_in_repo=filename,
            repo_id=dataset_id,
            repo_type="dataset",
            commit_message=f"Upload pre-computed network graph (version {version})"
        )
        
        logger.info(f"Successfully uploaded {filename} to {dataset_id}")
        
        # Try to upload metadata if it exists
        metadata_file = network_path.parent / "network_metadata.json"
        if metadata_file.exists():
            try:
                upload_file(
                    path_or_fileobj=str(metadata_file),
                    path_in_repo=f"network_metadata_{version}.json",
                    repo_id=dataset_id,
                    repo_type="dataset",
                    commit_message=f"Upload network metadata (version {version})"
                )
                logger.info("Successfully uploaded network metadata")
            except Exception as e:
                logger.warning(f"Could not upload metadata: {e}")
        
        return True
        
    except ImportError:
        logger.error("huggingface_hub not installed. Install it with: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"Error uploading network to HF Hub: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload pre-computed network graph to HF Hub")
    parser.add_argument("--network-file", type=str, 
                       default="precomputed_data/full_derivative_network.pkl",
                       help="Path to network pickle file")
    parser.add_argument("--version", type=str, default="v1",
                       help="Version tag for the data")
    parser.add_argument("--dataset-id", type=str, default=None,
                       help="HF dataset ID (defaults to HF_PRECOMPUTED_DATASET)")
    
    args = parser.parse_args()
    
    success = upload_network_to_hf(
        network_file=args.network_file,
        version=args.version,
        dataset_id=args.dataset_id
    )
    
    sys.exit(0 if success else 1)

